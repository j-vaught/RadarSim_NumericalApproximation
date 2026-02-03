"""
Swerling fluctuating target models and point target representation.

References:
- Swerling, P. "Probability of Detection for Fluctuating Targets" (1960)
- Mahafza, B.R. "Radar Systems Analysis and Design Using MATLAB"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from enum import IntEnum


class SwerlingCase(IntEnum):
    """Swerling target fluctuation cases."""
    CASE_0 = 0   # Non-fluctuating (Marcum)
    CASE_1 = 1   # Slow fluctuation, many equal scatterers (scan-to-scan)
    CASE_2 = 2   # Fast fluctuation, many equal scatterers (pulse-to-pulse)
    CASE_3 = 3   # Slow fluctuation, one dominant + many small (scan-to-scan)
    CASE_4 = 4   # Fast fluctuation, one dominant + many small (pulse-to-pulse)


@dataclass
class SwerlingTarget:
    """
    Fluctuating target RCS model based on Swerling cases.

    Physical interpretation:
    - Cases 1 & 2: Many scatterers of comparable size (complex target)
    - Cases 3 & 4: One dominant scatterer + many small (simple target)
    - Slow (1, 3): RCS constant during dwell, changes scan-to-scan
    - Fast (2, 4): RCS fluctuates pulse-to-pulse
    """

    case: SwerlingCase
    mean_rcs_m2: float

    # Internal state for slow-fluctuating cases
    _current_rcs: float = field(default=None, repr=False)
    _scan_index: int = field(default=-1, repr=False)

    def __post_init__(self):
        if isinstance(self.case, int):
            self.case = SwerlingCase(self.case)

    def sample_rcs(self, num_pulses: int = 1, scan_index: int = 0) -> np.ndarray:
        """
        Generate RCS samples based on Swerling case.

        Args:
            num_pulses: Number of pulses in the dwell
            scan_index: Current scan number (for slow-fluctuating cases)

        Returns:
            Array of RCS values in m^2, shape (num_pulses,)
        """
        sigma = self.mean_rcs_m2

        if self.case == SwerlingCase.CASE_0:
            # Non-fluctuating: constant RCS
            return np.full(num_pulses, sigma)

        elif self.case == SwerlingCase.CASE_1:
            # Exponential (Rayleigh amplitude), constant within scan
            if scan_index != self._scan_index:
                self._current_rcs = np.random.exponential(sigma)
                self._scan_index = scan_index
            return np.full(num_pulses, self._current_rcs)

        elif self.case == SwerlingCase.CASE_2:
            # Exponential, pulse-to-pulse variation
            return np.random.exponential(sigma, num_pulses)

        elif self.case == SwerlingCase.CASE_3:
            # Chi-squared (4 DOF), constant within scan
            # Mean of chi-sq(4) with scale s is 2s, so use sigma/2
            if scan_index != self._scan_index:
                self._current_rcs = np.random.gamma(2, sigma/2)
                self._scan_index = scan_index
            return np.full(num_pulses, self._current_rcs)

        elif self.case == SwerlingCase.CASE_4:
            # Chi-squared (4 DOF), pulse-to-pulse
            return np.random.gamma(2, sigma/2, num_pulses)

        else:
            raise ValueError(f"Unknown Swerling case: {self.case}")

    def get_pdf(self, rcs_values: np.ndarray) -> np.ndarray:
        """
        Compute PDF of RCS for this Swerling case.

        Args:
            rcs_values: RCS values to evaluate PDF at

        Returns:
            PDF values
        """
        sigma = self.mean_rcs_m2
        x = np.maximum(rcs_values, 1e-20)  # Avoid division by zero

        if self.case in [SwerlingCase.CASE_0]:
            # Delta function at mean
            return np.where(np.isclose(x, sigma), np.inf, 0)

        elif self.case in [SwerlingCase.CASE_1, SwerlingCase.CASE_2]:
            # Exponential: p(x) = (1/sigma) * exp(-x/sigma)
            return (1/sigma) * np.exp(-x/sigma)

        elif self.case in [SwerlingCase.CASE_3, SwerlingCase.CASE_4]:
            # Chi-squared (4 DOF): p(x) = (4x/sigma^2) * exp(-2x/sigma)
            return (4*x/sigma**2) * np.exp(-2*x/sigma)


@dataclass
class PointTarget:
    """
    A point target in the radar scene.

    Combines position, RCS model, and optional motion.
    """

    # Position (polar coordinates relative to radar)
    range_m: float
    azimuth_deg: float
    elevation_deg: float = 0.0

    # RCS characteristics
    mean_rcs_m2: float = 10.0
    swerling_case: SwerlingCase = SwerlingCase.CASE_1

    # Target classification (for labeling)
    target_class: str = "unknown"
    target_id: int = 0

    # Motion parameters (optional)
    speed_mps: float = 0.0
    heading_deg: float = 0.0

    # Internal
    _swerling: SwerlingTarget = field(default=None, repr=False)

    def __post_init__(self):
        self._swerling = SwerlingTarget(self.swerling_case, self.mean_rcs_m2)

    def get_position(self, time_s: float = 0.0) -> Tuple[float, float, float]:
        """
        Get target position at given time.

        Args:
            time_s: Time in seconds from start

        Returns:
            (range_m, azimuth_deg, elevation_deg)
        """
        if self.speed_mps == 0:
            return self.range_m, self.azimuth_deg, self.elevation_deg

        # Convert to Cartesian, apply motion, convert back
        r0, az0 = self.range_m, np.radians(self.azimuth_deg)
        x0 = r0 * np.sin(az0)
        y0 = r0 * np.cos(az0)

        # Apply velocity
        hdg = np.radians(self.heading_deg)
        vx = self.speed_mps * np.sin(hdg)
        vy = self.speed_mps * np.cos(hdg)

        x = x0 + vx * time_s
        y = y0 + vy * time_s

        # Back to polar
        r = np.sqrt(x**2 + y**2)
        az = np.degrees(np.arctan2(x, y)) % 360

        return r, az, self.elevation_deg

    def get_rcs(self, num_pulses: int = 1, scan_index: int = 0) -> np.ndarray:
        """Get fluctuating RCS samples."""
        return self._swerling.sample_rcs(num_pulses, scan_index)

    def get_range_bin(self, range_resolution_m: float) -> int:
        """Convert range to bin index."""
        return int(self.range_m / range_resolution_m)

    def get_azimuth_bin(self, num_azimuths: int) -> int:
        """Convert azimuth to bin index."""
        return int(self.azimuth_deg / 360 * num_azimuths) % num_azimuths


# Common target RCS values (approximate, X-band)
TYPICAL_RCS = {
    'small_boat': 5.0,        # m^2, 5-10m vessel
    'medium_vessel': 100.0,   # m^2, 20-50m vessel
    'large_ship': 10000.0,    # m^2, cargo ship
    'buoy': 1.0,              # m^2, navigation buoy
    'corner_reflector_3in': 0.08,  # m^2, 3" corner reflector
    'person_in_water': 0.5,   # m^2
    'kayak': 2.0,             # m^2
    'jet_ski': 3.0,           # m^2
    'sailboat': 20.0,         # m^2, mast effect
}


def create_target_ensemble(num_targets: int,
                           range_limits: Tuple[float, float] = (500, 5000),
                           azimuth_limits: Optional[Tuple[float, float]] = None,
                           rcs_range: Tuple[float, float] = (1, 100),
                           swerling_cases: List[int] = [1, 3]) -> List[PointTarget]:
    """
    Create a random ensemble of point targets.

    Args:
        num_targets: Number of targets to create
        range_limits: (min, max) range in meters
        azimuth_limits: (min, max) azimuth in degrees (supports wrap-around, e.g., (270, 90))
        rcs_range: (min, max) mean RCS in m^2
        swerling_cases: List of Swerling cases to sample from

    Returns:
        List of PointTarget objects
    """
    targets = []

    for i in range(num_targets):
        # Handle azimuth limits with wrap-around
        if azimuth_limits is None:
            az = np.random.uniform(0, 360)
        else:
            az_min, az_max = azimuth_limits
            if az_min <= az_max:
                az = np.random.uniform(az_min, az_max)
            else:
                # Wrap-around case (e.g., 270 to 90 means 270-360 and 0-90)
                if np.random.random() < (360 - az_min) / (360 - az_min + az_max):
                    az = np.random.uniform(az_min, 360)
                else:
                    az = np.random.uniform(0, az_max)

        target = PointTarget(
            range_m=np.random.uniform(*range_limits),
            azimuth_deg=az % 360,
            mean_rcs_m2=np.exp(np.random.uniform(np.log(rcs_range[0]), np.log(rcs_range[1]))),
            swerling_case=SwerlingCase(np.random.choice(swerling_cases)),
            target_id=i,
            speed_mps=np.random.uniform(0, 15),  # 0-30 knots
            heading_deg=np.random.uniform(0, 360),
        )
        targets.append(target)

    return targets
