"""
Tier 2 Analytical Radar Simulation Pipeline

Fast statistical radar data generation using:
- K-distribution / Weibull sea clutter
- Swerling fluctuating target models
- Analytical radar equation (no ray tracing)
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml

# Radar system configuration
from .radar import RadarConfig, FurunoNXT

# Core data structures
from .core import (
    NUM_RANGE_BINS,
    Pulse,
    RadarFrame,
    RadarCSVHandler,
    RadarConverter,
    ConversionConfig,
    radians_to_ticks,
    PolarMask,
)

# Statistical models
from .models import (
    SeaClutterModel,
    RainClutterModel,
    SwerlingTarget,
    PointTarget,
    create_target_ensemble,
    AnalyticalPropagation,
    PropagationConfig,
    LandGenerator,
    LandConfig,
    create_harbor_coastline,
)

# Scene system
from .scene import SceneConfig, SceneAdapter, load_scene

# Land / background loading
from .land_loader import LandLoader, LandLoadResult


@dataclass
class Tier2Config:
    """Configuration for Tier 2 simulation."""
    num_frames: int = 100
    seed: Optional[int] = None

    # Environment
    sea_state: int = 3
    rain_rate_mmhr: float = 0.0
    land_enabled: bool = False
    land_config: Optional[LandConfig] = None
    render_mode: str = "auto"  # "radar_equation" | "max_blend" | "auto"

    # Output
    output_dir: Path = Path("output")
    output_polar_csv: bool = True
    output_cartesian_png: bool = True
    output_yolo_labels: bool = True

    # Radar (use FurunoNXT preset with Tier3-compatible range bins)
    # Max range = range_resolution * num_range_bins
    # For 0.5 NM (926m): 926 / 868 = 1.07 m/bin
    radar: RadarConfig = field(default_factory=lambda: RadarConfig(
        frequency=9.41e9,
        peak_power=25.0,
        pulse_width=0.5e-6,
        prf=2100.0,
        beamwidth_h=3.9,
        beamwidth_v=25.0,
        gain_db=26.0,
        rotation_rate=24.0,
        antenna_height=4.0,
        noise_figure_db=4.0,
        bandwidth=25e6,
        range_resolution=1.07,  # ~0.5 NM max range
        num_range_bins=NUM_RANGE_BINS,  # 868 for Tier3 compatibility
        samples_per_revolution=720,
    ))

    @classmethod
    def from_yaml(cls, path: Path) -> 'Tier2Config':
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        radar_data = data.get('radar', {})
        radar = RadarConfig(
            frequency=radar_data.get('frequency_ghz', 9.41) * 1e9,
            peak_power=radar_data.get('peak_power_w', 25),
            antenna_height=radar_data.get('antenna_height_m', 4),
            gain_db=radar_data.get('antenna_gain_db', 26),
            range_resolution=radar_data.get('range_resolution_m', 6),
            num_range_bins=radar_data.get('num_range_bins', NUM_RANGE_BINS),
            samples_per_revolution=radar_data.get('num_azimuths', 720),
            beamwidth_h=radar_data.get('beamwidth_h_deg', 3.9),
            beamwidth_v=radar_data.get('beamwidth_v_deg', 25),
        )

        return cls(
            num_frames=data.get('simulation', {}).get('num_frames', 100),
            seed=data.get('simulation', {}).get('seed'),
            sea_state=data.get('environment', {}).get('sea_state', 3),
            rain_rate_mmhr=data.get('environment', {}).get('rain_rate_mmhr', 0),
            output_dir=Path(data.get('output', {}).get('directory', './output')),
            radar=radar,
        )

    @classmethod
    def from_scene(cls, scene_config: SceneConfig) -> 'Tier2Config':
        """Build Tier2Config from a scene YAML's environment block."""
        env = scene_config.environment
        land_enabled = env.land is not None
        land_config = None
        if land_enabled and env.land.type == "parametric":
            land_config = LandConfig(
                coastline_range=env.land.coastline_range,
                land_start_az=env.land.land_start_az,
                land_end_az=env.land.land_end_az,
                roughness=env.land.roughness,
                intensity=env.land.intensity,
                bay_enabled=env.land.bay_enabled,
                bay_center_az=env.land.bay_center_az,
                bay_width_az=env.land.bay_width_az,
                bay_depth=env.land.bay_depth,
            )

        # Resolve render_mode: "auto" picks based on land type
        render_mode = env.render_mode
        if render_mode == "auto":
            if land_enabled and env.land.land_frames_dir:
                render_mode = "max_blend"
            else:
                render_mode = "radar_equation"

        return cls(
            num_frames=scene_config.count,
            seed=scene_config.seed,
            sea_state=env.sea_state,
            rain_rate_mmhr=env.rain_rate_mmhr,
            land_enabled=land_enabled,
            land_config=land_config,
            render_mode=render_mode,
        )


class Tier2Pipeline:
    """
    Analytical radar simulation pipeline.

    Generates radar frames using:
    - Statistical clutter models (K-distribution sea clutter)
    - Swerling fluctuating target models
    - Analytical radar equation (no ray tracing)
    """

    def __init__(self, config: Tier2Config):
        self.config = config
        self.radar = config.radar

        if config.seed is not None:
            np.random.seed(config.seed)

        # Initialize models
        self.clutter_model = SeaClutterModel(sea_state=config.sea_state)
        self.rain_model = RainClutterModel(config.rain_rate_mmhr) if config.rain_rate_mmhr > 0 else None

        self.propagation = AnalyticalPropagation(PropagationConfig(
            wavelength_m=self.radar.wavelength,
            antenna_height_m=self.radar.antenna_height,
        ))

        # Range array for computations
        self.range_bins = np.arange(self.radar.num_range_bins) * self.radar.range_resolution
        self.range_bins[0] = self.radar.range_resolution

        # Land mask (if enabled)
        self.land_mask = None
        self.land_generator = None
        self._land_full_depth = False  # True for annotation/mask land sources
        if config.land_enabled:
            land_cfg = config.land_config or LandConfig()
            self.land_generator = LandGenerator(land_cfg)
            self.land_mask = self.land_generator.generate_coastline(
                self.radar.samples_per_revolution,
                self.radar.num_range_bins,
                seed=config.seed
            )

        # Converter for polar->Cartesian (lazy init)
        self._converter = None

    @property
    def converter(self) -> RadarConverter:
        """Lazy-initialize the Cartesian converter."""
        if self._converter is None:
            self._converter = RadarConverter(ConversionConfig(
                num_pulses=self.radar.samples_per_revolution,
            ))
        return self._converter

    def generate_clutter(self) -> np.ndarray:
        """Generate sea clutter background for one frame."""
        shape = (self.radar.samples_per_revolution, self.radar.num_range_bins)
        clutter = self.clutter_model.correlated_clutter(shape, correlation_range=3.0)

        if self.rain_model is not None:
            rain = self.rain_model.generate(shape, self.range_bins, self.radar.range_resolution)
            clutter += rain

        return clutter

    def generate_noise(self) -> np.ndarray:
        """Generate thermal noise floor."""
        shape = (self.radar.samples_per_revolution, self.radar.num_range_bins)
        noise_power = self.radar.thermal_noise_power
        return np.random.rayleigh(np.sqrt(noise_power / 2), shape)

    def add_target(self, frame: np.ndarray, target: PointTarget,
                   time_s: float, scan_index: int,
                   intensity_scale: float = 1.0) -> Optional[Dict[str, Any]]:
        """Add a single target return to the frame.

        Args:
            frame: Radar frame array to modify in-place.
            target: PointTarget to render.
            time_s: Current simulation time in seconds.
            scan_index: Current scan number for Swerling fluctuation.
            intensity_scale: Multiplier applied after multipath (default 1.0).
        """
        range_m, az_deg, _ = target.get_position(time_s)

        if range_m < 0 or range_m > self.radar.max_range:
            return None

        rcs = target.get_rcs(num_pulses=1, scan_index=scan_index)[0]

        # Compute SNR-based intensity (relative to background)
        # Use radar equation for R^4 falloff, but scale to visible intensity
        snr = self.radar.snr(range_m, rcs)

        # Scale intensity: base level that's visible above clutter
        # Larger RCS and closer range = brighter target
        base_intensity = 1e-7  # Tuned to be visible above sea state 4 clutter
        intensity = base_intensity * snr * (rcs / 10.0)  # Normalize to 10 m² reference

        # Apply multipath fading
        F4 = self.propagation.two_ray_propagation_factor(
            np.array([range_m]), target_height_m=5.0
        )[0]
        intensity *= np.clip(F4, 0.1, 10.0)  # Limit extreme multipath swings

        # Apply scene intensity scale (from IntensityEngine)
        intensity *= intensity_scale

        range_bin = self.radar.range_to_bin(range_m)
        az_bin = self.radar.azimuth_to_spoke(az_deg)

        # Minimum blob size + scaling with RCS
        min_blob = 4  # Minimum radar return size
        base_size = max(min_blob, int(np.sqrt(rcs) * 1.2)) + np.random.randint(0, 3)

        # In Cartesian, azimuth spread = range * delta_angle
        # To look circular, reduce azimuth bins at longer ranges
        range_norm = range_bin / self.radar.num_range_bins
        az_scale = max(0.4, 1.0 - range_norm * 0.6)  # Shrink azimuth at longer range

        blob_size_r = base_size + np.random.randint(0, 4)
        blob_size_az = max(2, int(base_size * az_scale))

        # Add some irregularity
        for da in range(-blob_size_az - 1, blob_size_az + 2):
            for dr in range(-blob_size_r - 1, blob_size_r + 2):
                # Irregular blob boundary
                az_radius = blob_size_az * (0.8 + 0.4 * np.random.random())
                r_radius = blob_size_r * (0.8 + 0.4 * np.random.random())

                dist_sq = (da / max(az_radius, 0.5))**2 + (dr / max(r_radius, 0.5))**2
                if dist_sq > 1.0:
                    continue

                # Soft falloff with speckle
                weight = np.exp(-2.0 * dist_sq) * (0.5 + 0.5 * np.random.random())

                ai = (az_bin + da) % self.radar.samples_per_revolution
                ri = range_bin + dr

                if 0 <= ri < self.radar.num_range_bins:
                    frame[ai, ri] += intensity * weight

        return {
            'target_id': target.target_id,
            'class': target.target_class,
            'range_m': range_m,
            'azimuth_deg': az_deg,
            'range_bin': range_bin,
            'azimuth_bin': az_bin,
            'rcs_m2': rcs,
            'intensity': intensity,
        }

    def generate_land_returns(self) -> np.ndarray:
        """Generate radar returns from land."""
        if self.land_mask is None:
            return np.zeros((self.radar.samples_per_revolution, self.radar.num_range_bins))

        base_intensity = 5e-3 if self._land_full_depth else 5e-4
        returns = self.land_generator.generate_land_returns(
            self.land_mask,
            base_intensity=base_intensity,
            full_depth=self._land_full_depth,
        )
        return returns

    def generate_frame(self, targets: List[PointTarget],
                       frame_idx: int, scan_index: int) -> tuple:
        """Generate one complete radar frame."""
        time_s = frame_idx * self.radar.rotation_period

        frame = self.generate_clutter() + self.generate_noise()

        # Add land returns
        if self.land_mask is not None:
            frame += self.generate_land_returns()

        labels = []
        for target in targets:
            label = self.add_target(frame, target, time_s, scan_index)
            if label is not None:
                labels.append(label)

        # Convert to uint8 with 4-bit quantization (16 levels)
        frame_db = 10 * np.log10(frame + 1e-20)

        # Threshold so most background is 0 - only strong returns show
        # Use a higher floor so weak clutter maps to 0
        noise_floor = np.percentile(frame_db, 85)  # 85th percentile as floor
        db_min = noise_floor
        db_max = noise_floor + 25  # 25 dB dynamic range above floor

        frame_norm = np.clip((frame_db - db_min) / (db_max - db_min), 0, 1)

        # 4-bit quantization: 16 discrete levels (0, 17, 34, ..., 255)
        frame_quantized = np.floor(frame_norm * 15).astype(np.uint8)  # 0-15
        frame_quantized = (frame_quantized * 17).astype(np.uint8)     # Scale to 0-255

        return frame_quantized, labels

    def array_to_radar_frame(self, data: np.ndarray, frame_idx: int) -> RadarFrame:
        """Convert numpy array to RadarFrame for output compatibility."""
        num_pulses = data.shape[0]
        frame = RadarFrame(
            source_path=f"tier2_frame_{frame_idx:04d}",
            timestamp=f"{frame_idx:04d}",
        )

        angle_step = 2.0 * np.pi / num_pulses
        for i in range(num_pulses):
            angle_rad = i * angle_step
            pulse = Pulse(
                status=1,
                scale=0,
                range_=0,
                gain=0,
                angle_ticks=radians_to_ticks(angle_rad),
                angle_rad=angle_rad,
                echoes=data[i, :].astype(np.uint8),
            )
            frame.pulses.append(pulse)

        frame.update_unique_angles()
        return frame

    def write_outputs(self, frame: np.ndarray, labels: List[Dict], frame_idx: int):
        """Write frame to all configured output formats."""
        num_azimuths = self.radar.samples_per_revolution
        num_bins = self.radar.num_range_bins

        if self.config.output_polar_csv:
            csv_dir = self.config.output_dir / 'csv'
            csv_dir.mkdir(parents=True, exist_ok=True)
            radar_frame = self.array_to_radar_frame(frame, frame_idx)
            RadarCSVHandler.write_csv(radar_frame, str(csv_dir / f'frame_{frame_idx:04d}.csv'))

        if self.config.output_cartesian_png:
            img_dir = self.config.output_dir / 'images'
            img_dir.mkdir(parents=True, exist_ok=True)

            radar_frame = self.array_to_radar_frame(frame, frame_idx)
            gray_img = self.converter.polar_to_cartesian_gray(radar_frame)

            from PIL import Image
            img = Image.fromarray(gray_img, mode='L')
            img.save(str(img_dir / f'frame_{frame_idx:04d}.png'))

        if self.config.output_yolo_labels:
            label_dir = self.config.output_dir / 'labels'
            label_dir.mkdir(parents=True, exist_ok=True)

            with open(label_dir / f'frame_{frame_idx:04d}.txt', 'w') as f:
                for lbl in labels:
                    x_norm = lbl['range_bin'] / num_bins
                    y_norm = lbl['azimuth_bin'] / num_azimuths
                    w_norm = 0.02
                    h_norm = 0.01
                    class_id = 0
                    f.write(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    def run(self, targets: Optional[List[PointTarget]] = None):
        """Run the full simulation."""
        if targets is None:
            targets = create_target_ensemble(
                num_targets=10,
                range_limits=(500, self.radar.max_range * 0.8),
            )
            print(f"Generated {len(targets)} random targets")

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Generating {self.config.num_frames} frames to {self.config.output_dir}")

        for i in range(self.config.num_frames):
            frame, labels = self.generate_frame(targets, i, scan_index=i)
            self.write_outputs(frame, labels, i)

            if (i + 1) % 50 == 0:
                print(f"  Frame {i+1}/{self.config.num_frames}")

        print("Done!")

    def generate_scene_frame(self, adapter: SceneAdapter,
                             frame_idx: int, scan_index: int) -> tuple:
        """Generate one radar frame driven by a SceneAdapter."""
        frame = self.generate_clutter() + self.generate_noise()

        # Add land returns
        if self.land_mask is not None:
            frame += self.generate_land_returns()

        labels = []
        time_s = frame_idx * self.radar.rotation_period

        for target, intensity_scale in adapter.get_targets_for_frame(frame_idx):
            label = self.add_target(frame, target, time_s, scan_index,
                                    intensity_scale=intensity_scale)
            if label is not None:
                labels.append(label)

        # Convert to uint8 with 4-bit quantization
        frame_db = 10 * np.log10(frame + 1e-20)

        # Compute noise floor from water pixels only (if land mask present)
        # so land-dominant frames don't shift the dynamic range
        if self.land_mask is not None:
            water_pixels = frame_db[~self.land_mask]
            if water_pixels.size > 0:
                noise_floor = np.percentile(water_pixels, 65)
            else:
                noise_floor = np.percentile(frame_db, 65)
            db_range = 35
        else:
            noise_floor = np.percentile(frame_db, 85)
            db_range = 25

        db_min = noise_floor
        db_max = noise_floor + db_range

        frame_norm = np.clip((frame_db - db_min) / (db_max - db_min), 0, 1)
        frame_quantized = np.floor(frame_norm * 15).astype(np.uint8)
        frame_quantized = (frame_quantized * 17).astype(np.uint8)

        return frame_quantized, labels

    def generate_land_frame(self, land_frames: List[np.ndarray],
                            frame_idx: int, adapter: SceneAdapter,
                            scan_index: int) -> tuple:
        """Generate one radar frame using a real land frame as background."""
        # Cycle through land frames
        bg = land_frames[frame_idx % len(land_frames)].copy().astype(np.float32)

        labels = []
        time_s = frame_idx * self.radar.rotation_period

        for target, intensity_scale in adapter.get_targets_for_frame(frame_idx):
            # For land-frame mode, add targets via max-blending (like Tier3)
            label = self._add_target_to_land_frame(
                bg, target, time_s, scan_index, intensity_scale)
            if label is not None:
                labels.append(label)

        # Already uint8 data — just clamp
        frame_quantized = np.clip(bg, 0, 255).astype(np.uint8)
        return frame_quantized, labels

    def _add_target_to_land_frame(self, frame: np.ndarray, target: PointTarget,
                                   time_s: float, scan_index: int,
                                   intensity_scale: float = 1.0) -> Optional[Dict[str, Any]]:
        """Add a target to a real land frame using max-blending (Tier3 style)."""
        range_m, az_deg, _ = target.get_position(time_s)

        if range_m < 0 or range_m > self.radar.max_range:
            return None

        rcs = target.get_rcs(num_pulses=1, scan_index=scan_index)[0]

        range_bin = self.radar.range_to_bin(range_m)
        az_bin = self.radar.azimuth_to_spoke(az_deg)

        # Scale target brightness based on RCS — map to uint8 range
        # Must be bright enough to stand out against land returns (up to 252)
        base_brightness = min(252, max(230, int(15 * np.log10(rcs + 1) + 235)))
        # Clamp intensity_scale so targets always remain visible
        effective_scale = max(0.85, intensity_scale)
        brightness = max(220, min(252, int(base_brightness * effective_scale)))

        # Blob size from RCS
        min_blob = 4
        base_size = max(min_blob, int(np.sqrt(rcs) * 1.2)) + np.random.randint(0, 3)

        range_norm = range_bin / self.radar.num_range_bins
        az_scale = max(0.4, 1.0 - range_norm * 0.6)

        blob_size_r = base_size + np.random.randint(0, 4)
        blob_size_az = max(2, int(base_size * az_scale))

        for da in range(-blob_size_az - 1, blob_size_az + 2):
            for dr in range(-blob_size_r - 1, blob_size_r + 2):
                az_radius = blob_size_az * (0.8 + 0.4 * np.random.random())
                r_radius = blob_size_r * (0.8 + 0.4 * np.random.random())

                dist_sq = (da / max(az_radius, 0.5))**2 + (dr / max(r_radius, 0.5))**2
                if dist_sq > 1.0:
                    continue

                weight = np.exp(-0.8 * dist_sq) * (0.85 + 0.15 * np.random.random())
                pixel_val = int(brightness * weight)

                ai = (az_bin + da) % self.radar.samples_per_revolution
                ri = range_bin + dr

                if 0 <= ri < self.radar.num_range_bins:
                    # Max-blend: take the brighter of background or target
                    frame[ai, ri] = max(frame[ai, ri], pixel_val)

        return {
            'target_id': target.target_id,
            'class': target.target_class,
            'range_m': range_m,
            'azimuth_deg': az_deg,
            'range_bin': range_bin,
            'azimuth_bin': az_bin,
            'rcs_m2': rcs,
            'intensity': brightness,
        }

    def run_scene(self, scene_config: SceneConfig, scene_path: Path):
        """Run simulation driven by a scene YAML file."""
        env = scene_config.environment

        # Load land data via LandLoader (replaces inline if/elif chain)
        if env.land is not None:
            result = LandLoader.load(env.land, scene_path, self.converter, self.radar)
            if result.land_mask is not None:
                self.land_mask = result.land_mask
                self._land_full_depth = result.full_depth
            if result.land_generator is not None:
                self.land_generator = result.land_generator
            land_frames = result.land_frames
            water_polar = result.water_mask
        else:
            land_frames = None
            num_pulses = self.radar.samples_per_revolution
            num_bins = self.radar.num_range_bins
            water_polar = PolarMask(num_pulses, fill=True)
            water_polar.data = np.ones((num_pulses, num_bins), dtype=bool)

        # Build scene adapter
        rng = np.random.RandomState(scene_config.seed)
        adapter = SceneAdapter(scene_config, self.radar, water_polar, rng)
        print(f"Scene: {len(adapter.tracks)} object tracks built")

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        total = scene_config.count
        render = self.config.render_mode
        print(f"Generating {total} scene frames to {self.config.output_dir} (render_mode={render})")

        for i in range(total):
            if render == "max_blend" and land_frames is not None:
                frame, labels = self.generate_land_frame(
                    land_frames, i, adapter, scan_index=i)
            else:
                frame, labels = self.generate_scene_frame(adapter, i, scan_index=i)
            self.write_outputs(frame, labels, i)

            if (i + 1) % 50 == 0:
                print(f"  Frame {i+1}/{total}")

        print("Done!")


def main():
    """Command-line interface."""
    import argparse

    parser = argparse.ArgumentParser(description='Tier 2 Analytical Radar Simulation')
    parser.add_argument('--config', '-c', type=Path, help='YAML configuration file')
    parser.add_argument('--scene', type=Path, help='Scene YAML file (unified Tier2/Tier3 format)')
    parser.add_argument('--frames', '-n', type=int, default=100, help='Number of frames')
    parser.add_argument('--sea-state', '-s', type=int, default=3, help='Sea state (0-7)')
    parser.add_argument('--output', '-o', type=Path, default=Path('./output'), help='Output directory')
    parser.add_argument('--seed', type=int, help='Random seed')

    args = parser.parse_args()

    if args.scene:
        scene_config = load_scene(str(args.scene))
        config = Tier2Config.from_scene(scene_config)
        if args.output != Path('./output'):
            config.output_dir = args.output
        pipeline = Tier2Pipeline(config)
        pipeline.run_scene(scene_config, args.scene)
    elif args.config:
        config = Tier2Config.from_yaml(args.config)
        pipeline = Tier2Pipeline(config)
        pipeline.run()
    else:
        config = Tier2Config(
            num_frames=args.frames,
            sea_state=args.sea_state,
            output_dir=args.output,
            seed=args.seed,
        )
        pipeline = Tier2Pipeline(config)
        pipeline.run()


if __name__ == '__main__':
    main()
