"""Intensity Pipeline — profiles, jitter, and flicker state machine."""

import math
from typing import List

from .config import FlickerConfig, IntensityConfig
from .expressions import safe_eval


class IntensityEngine:
    """Computes per-frame intensity scale factors combining profile, jitter, and flicker."""

    def __init__(self, config: IntensityConfig, flicker: FlickerConfig,
                 total_frames: int, rng):
        self._config = config
        self._flicker = flicker
        self._total_frames = total_frames
        self._rng = rng

        # Pre-compute flicker state timeline
        self._flicker_active = [False] * total_frames
        if flicker.enabled:
            self._build_flicker_timeline()

    def _build_flicker_timeline(self):
        f = self._flicker
        rng = self._rng
        frame = 0
        is_flickering = False
        while frame < self._total_frames:
            if is_flickering:
                duration = rng.randint(f.flicker_frames[0], f.flicker_frames[1])
                end = min(frame + duration, self._total_frames)
                for i in range(frame, end):
                    self._flicker_active[i] = True
                frame = end
                is_flickering = False
            else:
                duration = rng.randint(f.primary_frames[0], f.primary_frames[1])
                frame += duration
                # Transition: chance to enter flicker
                if frame < self._total_frames and rng.random() < f.rate:
                    is_flickering = True

    def get_scale(self, frame_idx: int) -> float:
        """Get intensity multiplier for a given frame index."""
        cfg = self._config

        # Profile base value
        if cfg.profile == "constant":
            value = cfg.base
        elif cfg.profile == "ramp":
            t_frac = frame_idx / max(self._total_frames - 1, 1)
            value = cfg.ramp_start + (cfg.ramp_end - cfg.ramp_start) * t_frac
        elif cfg.profile == "sine":
            value = cfg.base + cfg.amplitude * math.sin(2.0 * math.pi * frame_idx / cfg.period)
        elif cfg.profile == "custom":
            value = safe_eval(cfg.expression, {"t": frame_idx})
        else:
            value = cfg.base

        # Jitter
        if cfg.variability > 0:
            jitter = 1.0 + self._rng.uniform(-cfg.variability, cfg.variability)
            value *= jitter

        # Flicker
        if self._flicker.enabled and frame_idx < len(self._flicker_active):
            if self._flicker_active[frame_idx]:
                value *= self._flicker.intensity_drop

        return max(0.0, value)


def scale_echo_data(echo_data: List[List[int]], scale_factor: float) -> List[List[int]]:
    """Scale echo data by a factor, clamping to 0-255."""
    result = []
    for row in echo_data:
        scaled_row = []
        for val in row:
            scaled_row.append(int(min(255.0, max(0.0, val * scale_factor))))
        result.append(scaled_row)
    return result
