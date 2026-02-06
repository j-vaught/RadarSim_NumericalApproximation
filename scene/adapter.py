"""Scene Adapter — bridge between scene YAML objects and Tier2 PointTargets."""

from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np

from .config import ObjectGroupConfig, PhysicsConfig, SceneConfig
from .intensity import IntensityEngine
from .motion import generate_path, resolve_position
from ..core import NUM_RANGE_BINS
from ..models.targets import PointTarget, SwerlingCase


# Size-to-RCS fallback map (m²) when physics.rcs_m2 is not specified
SIZE_RCS_MAP = {
    "small": (0.5, 5.0),
    "medium": (5.0, 50.0),
    "large": (50.0, 500.0),
}


class NumpyRngAdapter:
    """Wraps np.random.RandomState to match the stdlib random.Random interface
    expected by generate_path() and IntensityEngine."""

    def __init__(self, rng: np.random.RandomState):
        self._rng = rng

    def randint(self, a: int, b: int) -> int:
        """Return random int in [a, b] inclusive (stdlib convention)."""
        return int(self._rng.randint(a, b + 1))

    def choice(self, seq):
        idx = self._rng.randint(0, len(seq))
        return seq[idx]

    def random(self) -> float:
        return float(self._rng.random())

    def uniform(self, a: float, b: float) -> float:
        return float(self._rng.uniform(a, b))


@dataclass
class SceneTrack:
    """One tracked instance from an object group."""
    group: ObjectGroupConfig
    instance_idx: int
    positions: List[Tuple[int, int]]
    start_frame: int
    end_frame: int
    intensity_engine: IntensityEngine
    target: PointTarget
    rcs_m2: float


def _resolve_rcs(physics: PhysicsConfig, size: Any, rng: NumpyRngAdapter) -> float:
    """Resolve RCS from physics config or fall back to size map."""
    if physics.rcs_m2 is not None:
        if isinstance(physics.rcs_m2, (list, tuple)) and len(physics.rcs_m2) == 2:
            lo, hi = float(physics.rcs_m2[0]), float(physics.rcs_m2[1])
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        return float(physics.rcs_m2)
    # Fall back to size map
    size_key = str(size) if isinstance(size, str) else "medium"
    lo, hi = SIZE_RCS_MAP.get(size_key, SIZE_RCS_MAP["medium"])
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


class SceneAdapter:
    """Bridge between scene config objects and Tier2 PointTargets.

    Builds motion tracks from the scene YAML, creates IntensityEngines,
    and yields (PointTarget, intensity_scale) pairs per frame.
    """

    def __init__(self, scene_config: SceneConfig, radar, water_mask,
                 rng: np.random.RandomState):
        self.scene = scene_config
        self.radar = radar
        self.water_mask = water_mask
        self._rng_adapter = NumpyRngAdapter(rng)
        self._np_rng = rng
        self.tracks: List[SceneTrack] = []

        # Build valid/edge positions from water mask
        self._valid_positions, self._edge_positions = self._build_positions()
        self._build_tracks()

    def _build_positions(self):
        """Extract valid water positions and edge positions from the water mask."""
        num_pulses = self.water_mask.data.shape[0]
        num_bins = self.water_mask.data.shape[1]

        valid = []
        edges = []
        for p in range(num_pulses):
            for b in range(num_bins):
                if self.water_mask.get(p, b):
                    valid.append((p, b))
                    # Check if this is an edge position (near coverage boundary
                    # or adjacent to land)
                    if b < 60 or b > num_bins - 60:
                        edges.append((p, b))
                    elif (not self.water_mask.get(p, b + 1) or
                          not self.water_mask.get(p, b - 1)):
                        edges.append((p, b))

        if not valid:
            # Fallback: all positions are valid (no mask constraint)
            for p in range(0, num_pulses, 10):
                for b in range(50, num_bins - 50, 10):
                    valid.append((p, b))
                    if b < 60 or b > num_bins - 60:
                        edges.append((p, b))

        return valid, edges

    def _build_tracks(self):
        """Build motion tracks for all object groups."""
        total_frames = self.scene.count
        num_pulses = self.water_mask.data.shape[0]

        for group in self.scene.objects:
            for i in range(group.count):
                try:
                    positions, start_frame, end_frame = generate_path(
                        path_config=group.path,
                        total_frames=total_frames,
                        valid_positions=self._valid_positions,
                        edge_positions=self._edge_positions,
                        water_mask=self.water_mask,
                        num_pulses=num_pulses,
                        rng=self._rng_adapter,
                        allow_land=group.path.allow_land,
                    )
                except (ValueError, IndexError):
                    continue

                if not positions:
                    continue

                engine = IntensityEngine(
                    config=group.intensity,
                    flicker=group.flicker,
                    total_frames=total_frames,
                    rng=self._rng_adapter,
                )

                rcs = _resolve_rcs(group.physics, group.size, self._rng_adapter)

                # Create the PointTarget (position will be updated per frame)
                first_pos = positions[0]
                range_m, az_deg = self._pixel_to_physical(first_pos[0], first_pos[1])
                target = PointTarget(
                    range_m=range_m,
                    azimuth_deg=az_deg,
                    mean_rcs_m2=rcs,
                    swerling_case=SwerlingCase(group.physics.swerling_case),
                    target_class=group.physics.target_class,
                    target_id=len(self.tracks),
                    speed_mps=0.0,  # Motion handled by adapter, not target
                )

                self.tracks.append(SceneTrack(
                    group=group,
                    instance_idx=i,
                    positions=positions,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    intensity_engine=engine,
                    target=target,
                    rcs_m2=rcs,
                ))

    def _pixel_to_physical(self, pulse: int, bin_idx: int) -> Tuple[float, float]:
        """Convert pixel space (pulse, bin) to physical (range_m, azimuth_deg)."""
        range_m = bin_idx * self.radar.range_resolution
        azimuth_deg = (pulse / self.radar.samples_per_revolution) * 360.0
        return range_m, azimuth_deg

    def get_targets_for_frame(self, frame_idx: int) -> Iterator[Tuple[PointTarget, float]]:
        """Yield (PointTarget, intensity_scale) for each active track at this frame."""
        for track in self.tracks:
            if frame_idx < track.start_frame or frame_idx >= track.end_frame:
                continue

            local_idx = frame_idx - track.start_frame
            if local_idx >= len(track.positions):
                continue

            pulse, bin_idx = track.positions[local_idx]
            range_m, az_deg = self._pixel_to_physical(pulse, bin_idx)

            # Update target position
            track.target.range_m = range_m
            track.target.azimuth_deg = az_deg

            # Get intensity scale from engine
            intensity_scale = track.intensity_engine.get_scale(frame_idx)

            yield track.target, intensity_scale
