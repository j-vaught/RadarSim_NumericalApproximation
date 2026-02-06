"""
Synthetic land/coastline generation for radar simulation.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class LandConfig:
    """Configuration for land generation."""
    # Coastline position (fraction of radar range, 0=center, 1=edge)
    coastline_range: float = 0.6
    # Azimuth range where land exists (degrees)
    land_start_az: float = 30.0
    land_end_az: float = 150.0
    # Coastline roughness (0=smooth, 1=very jagged)
    roughness: float = 0.3
    # Land return intensity (relative to max)
    intensity: float = 0.9
    # Add a bay/inlet
    bay_enabled: bool = True
    bay_center_az: float = 90.0
    bay_width_az: float = 30.0
    bay_depth: float = 0.15  # How far bay cuts into land


class LandGenerator:
    """Generate synthetic land returns for radar simulation."""

    def __init__(self, config: LandConfig = None):
        self.config = config or LandConfig()
        # Override azimuth boundaries (set by auto-detect from mask)
        self._override_land_start_az: Optional[float] = None
        self._override_land_end_az: Optional[float] = None

    def set_azimuth_bounds_from_mask(self, land_mask: np.ndarray):
        """Auto-detect land azimuth extent from a boolean mask and set fade boundaries."""
        num_az = land_mask.shape[0]
        land_azimuths = []
        for az_idx in range(num_az):
            if np.any(land_mask[az_idx, :]):
                land_azimuths.append(az_idx / num_az * 360.0)
        if land_azimuths:
            self._override_land_start_az = min(land_azimuths)
            self._override_land_end_az = max(land_azimuths)

    def generate_coastline(self, num_azimuths: int, num_range_bins: int,
                           seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a coastline mask in polar coordinates.

        Returns:
            Boolean mask where True = land, shape (num_azimuths, num_range_bins)
        """
        if seed is not None:
            np.random.seed(seed)

        cfg = self.config
        mask = np.zeros((num_azimuths, num_range_bins), dtype=bool)

        # Base coastline distance (in range bins)
        base_range = int(cfg.coastline_range * num_range_bins)

        # Generate fractal coastline variation
        coastline = self._generate_fractal_coastline(num_azimuths, cfg.roughness)
        coastline = coastline * num_range_bins * 0.15  # Scale roughness

        for az_idx in range(num_azimuths):
            az_deg = az_idx / num_azimuths * 360

            # Check if this azimuth has land
            if not self._is_land_azimuth(az_deg):
                continue

            # Calculate coastline distance at this azimuth
            coast_dist = base_range + int(coastline[az_idx])

            # Apply bay if enabled
            if cfg.bay_enabled:
                coast_dist = self._apply_bay(az_deg, coast_dist, num_range_bins)

            # Fill from coastline to edge with land
            coast_dist = max(0, min(coast_dist, num_range_bins - 1))
            mask[az_idx, coast_dist:] = True

        return mask

    def _is_land_azimuth(self, az_deg: float) -> bool:
        """Check if azimuth angle is in land sector."""
        cfg = self.config
        # Handle wrap-around
        if cfg.land_start_az <= cfg.land_end_az:
            return cfg.land_start_az <= az_deg <= cfg.land_end_az
        else:
            return az_deg >= cfg.land_start_az or az_deg <= cfg.land_end_az

    def _get_azimuth_fade(self, az_deg: float) -> float:
        """Get fade factor for azimuthal edges - no sharp cutoffs."""
        cfg = self.config
        fade_width = 15.0  # degrees to fade over

        # Use overridden bounds (from annotation/mask auto-detect) if available
        start_az = self._override_land_start_az if self._override_land_start_az is not None else cfg.land_start_az
        end_az = self._override_land_end_az if self._override_land_end_az is not None else cfg.land_end_az

        # Distance from start edge
        if start_az <= end_az:
            dist_from_start = az_deg - start_az
            dist_from_end = end_az - az_deg
        else:
            # Wrap-around case
            if az_deg >= start_az:
                dist_from_start = az_deg - start_az
                dist_from_end = (360 - az_deg) + end_az
            else:
                dist_from_start = (360 - start_az) + az_deg
                dist_from_end = end_az - az_deg

        # Fade at edges
        fade = 1.0
        if dist_from_start < fade_width:
            fade = min(fade, dist_from_start / fade_width)
        if dist_from_end < fade_width:
            fade = min(fade, dist_from_end / fade_width)

        return max(0, fade)

    def _apply_bay(self, az_deg: float, coast_dist: int, num_range_bins: int) -> int:
        """Cut a bay into the coastline."""
        cfg = self.config
        bay_center = cfg.bay_center_az
        bay_half_width = cfg.bay_width_az / 2

        # Distance from bay center
        delta = abs(az_deg - bay_center)
        if delta > 180:
            delta = 360 - delta

        if delta < bay_half_width:
            # Inside bay - push coastline outward
            bay_factor = np.cos(delta / bay_half_width * np.pi / 2) ** 2
            bay_push = int(cfg.bay_depth * num_range_bins * bay_factor)
            coast_dist += bay_push

        return coast_dist

    def _generate_fractal_coastline(self, length: int, roughness: float) -> np.ndarray:
        """Generate fractal coastline with organic blobby variation."""
        from scipy.ndimage import gaussian_filter

        coastline = np.zeros(length)

        # Multiple octaves of noise for natural look
        for octave in range(6):
            freq = 2 ** octave
            amp = roughness / (octave + 0.5)
            phase = np.random.random() * 2 * np.pi
            coastline += amp * np.sin(np.linspace(0, freq * 2 * np.pi, length) + phase)

        # Add medium-scale bumps (blobby protrusions)
        num_bumps = np.random.randint(3, 8)
        for _ in range(num_bumps):
            center = np.random.randint(0, length)
            width = np.random.randint(20, 60)
            height = np.random.randn() * roughness * 0.8
            bump = height * np.exp(-0.5 * ((np.arange(length) - center) / width) ** 2)
            coastline += bump

        # Smooth to make it blobby, not jagged
        coastline = gaussian_filter(coastline, sigma=3)

        return coastline

    def generate_land_returns(self, land_mask: np.ndarray,
                               base_intensity: float = 1e-6) -> np.ndarray:
        """
        Generate radar returns from land - solid bright like target blobs.

        Args:
            land_mask: Boolean mask of land positions
            base_intensity: Base intensity for land returns

        Returns:
            Intensity array for land returns
        """
        from scipy.ndimage import gaussian_filter

        cfg = self.config
        num_az, num_range = land_mask.shape
        returns = np.zeros((num_az, num_range), dtype=float)

        land_intensity = base_intensity * cfg.intensity

        # Generate blobby shadow edge variation (not straight)
        base_depth = np.random.randint(90, 130)

        # Multiple scales of variation for organic look
        depth_variation = np.zeros(num_az)
        for scale in [60, 30, 15]:
            freq = num_az / scale
            phase = np.random.random() * 2 * np.pi
            amp = 20 * (scale / 60)
            depth_variation += amp * np.sin(np.linspace(0, freq * 2 * np.pi, num_az) + phase)

        # Add some blobby bumps to shadow edge
        num_bumps = np.random.randint(4, 10)
        for _ in range(num_bumps):
            center = np.random.randint(0, num_az)
            width = np.random.randint(15, 50)
            height = np.random.randn() * 25
            bump = height * np.exp(-0.5 * ((np.arange(num_az) - center) / width) ** 2)
            depth_variation += bump

        depth_variation = gaussian_filter(depth_variation, sigma=8)
        visible_depths = (base_depth + depth_variation).astype(int)
        visible_depths = np.clip(visible_depths, 60, 180)

        # Fill visible land with solid intensity, fading at azimuthal edges
        for az in range(num_az):
            land_bins = np.where(land_mask[az, :])[0]
            if len(land_bins) == 0:
                continue

            az_deg = az / num_az * 360
            az_fade = self._get_azimuth_fade(az_deg)

            if az_fade <= 0:
                continue

            # Reduce visible depth at azimuthal edges (land tapers off)
            effective_depth = int(visible_depths[az] * az_fade)

            for i, r_bin in enumerate(land_bins):
                if i < effective_depth:
                    # Also fade intensity at edges
                    intensity = land_intensity * (0.5 + 0.5 * az_fade)
                    returns[az, r_bin] = intensity

        # Light blur for soft edges - same style as targets
        returns = gaussian_filter(returns, sigma=(1.0, 1.2))

        return returns


def create_harbor_coastline(num_azimuths: int = 720, num_range_bins: int = 868) -> Tuple[np.ndarray, LandConfig]:
    """Create a realistic harbor/bay coastline."""
    config = LandConfig(
        coastline_range=0.55,
        land_start_az=20.0,
        land_end_az=160.0,
        roughness=0.25,
        intensity=0.95,
        bay_enabled=True,
        bay_center_az=90.0,
        bay_width_az=40.0,
        bay_depth=0.2,
    )
    generator = LandGenerator(config)
    mask = generator.generate_coastline(num_azimuths, num_range_bins, seed=123)
    return mask, config


def create_peninsula_coastline(num_azimuths: int = 720, num_range_bins: int = 868) -> Tuple[np.ndarray, LandConfig]:
    """Create a peninsula jutting into the water."""
    config = LandConfig(
        coastline_range=0.75,
        land_start_az=60.0,
        land_end_az=120.0,
        roughness=0.2,
        intensity=0.9,
        bay_enabled=False,
    )
    generator = LandGenerator(config)
    mask = generator.generate_coastline(num_azimuths, num_range_bins, seed=456)
    return mask, config


def create_island(num_azimuths: int = 720, num_range_bins: int = 868,
                  center_az: float = 180.0, center_range: float = 0.4,
                  size: float = 0.1) -> np.ndarray:
    """Create a small island."""
    mask = np.zeros((num_azimuths, num_range_bins), dtype=bool)

    center_r_bin = int(center_range * num_range_bins)
    az_center_idx = int(center_az / 360 * num_azimuths)

    # Island radius in bins
    r_radius = int(size * num_range_bins)
    az_radius = int(size * num_azimuths * 0.5)  # Adjust for polar distortion

    for da in range(-az_radius, az_radius + 1):
        for dr in range(-r_radius, r_radius + 1):
            # Elliptical shape with noise
            dist = np.sqrt((da / max(az_radius, 1))**2 + (dr / max(r_radius, 1))**2)
            if dist < 0.8 + 0.2 * np.random.random():
                az_idx = (az_center_idx + da) % num_azimuths
                r_idx = center_r_bin + dr
                if 0 <= r_idx < num_range_bins:
                    mask[az_idx, r_idx] = True

    return mask
