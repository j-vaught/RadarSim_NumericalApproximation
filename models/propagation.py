"""
Analytical electromagnetic propagation models.

Fast approximations for path loss and multipath without ray tracing.

References:
- Blake, L.V. "Radar Range-Performance Analysis"
- Barton, D.K. "Modern Radar System Analysis"
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PropagationConfig:
    """Configuration for propagation model."""
    wavelength_m: float = 0.032  # X-band ~9.4 GHz
    antenna_height_m: float = 10.0
    earth_radius_m: float = 6.371e6
    effective_earth_factor: float = 4/3  # Standard atmosphere


class AnalyticalPropagation:
    """
    Fast analytical propagation models.

    Provides:
    - Free space loss
    - Two-ray (flat earth) multipath
    - Spherical earth horizon
    - Ducting effects (simplified)
    """

    def __init__(self, config: Optional[PropagationConfig] = None):
        self.config = config or PropagationConfig()
        self.k = 2 * np.pi / self.config.wavelength_m  # Wavenumber

    @property
    def effective_earth_radius(self) -> float:
        """Effective earth radius accounting for refraction."""
        return self.config.earth_radius_m * self.config.effective_earth_factor

    def free_space_loss(self, range_m: np.ndarray) -> np.ndarray:
        """
        Free-space path loss (one-way).

        L_fs = (4 * pi * R / lambda)^2

        Args:
            range_m: Range in meters

        Returns:
            Path loss factor (linear, > 1)
        """
        return (4 * np.pi * range_m / self.config.wavelength_m) ** 2

    def free_space_loss_db(self, range_m: np.ndarray) -> np.ndarray:
        """Free-space loss in dB."""
        return 10 * np.log10(self.free_space_loss(range_m))

    def two_ray_propagation_factor(self,
                                    range_m: np.ndarray,
                                    target_height_m: float,
                                    surface_reflection_coeff: complex = -1.0) -> np.ndarray:
        """
        Two-ray (flat earth) propagation factor.

        Accounts for direct ray + surface-reflected ray interference.

        F = |1 + rho * exp(j * delta_phi)|

        For radar: F^4 factor (two-way, squared for power)

        Args:
            range_m: Slant range in meters
            target_height_m: Target height above surface
            surface_reflection_coeff: Surface reflection coefficient (complex)
                                     -1 for perfect conductor (smooth sea)

        Returns:
            Propagation factor F^4 (linear)
        """
        h1 = self.config.antenna_height_m
        h2 = target_height_m

        # Path length difference
        # delta_R = 2 * h1 * h2 / R (small angle approximation)
        range_m = np.maximum(range_m, 1.0)  # Avoid division by zero
        delta_R = 2 * h1 * h2 / range_m

        # Phase difference
        delta_phi = self.k * delta_R

        # Propagation factor (one-way)
        rho = surface_reflection_coeff
        F = np.abs(1 + rho * np.exp(1j * delta_phi))

        # Return F^4 for two-way radar propagation
        return F ** 4

    def grazing_angle(self, range_m: np.ndarray) -> np.ndarray:
        """
        Compute grazing angle at given range.

        Uses flat earth approximation for simplicity.

        Args:
            range_m: Slant range in meters

        Returns:
            Grazing angle in radians
        """
        h = self.config.antenna_height_m
        return np.arctan(h / np.maximum(range_m, 1.0))

    def horizon_range(self, target_height_m: float = 0.0) -> float:
        """
        Compute radar horizon range.

        R_horizon = sqrt(2 * Re * h1) + sqrt(2 * Re * h2)

        Args:
            target_height_m: Target height above surface

        Returns:
            Horizon range in meters
        """
        Re = self.effective_earth_radius
        h1 = self.config.antenna_height_m
        h2 = target_height_m

        return np.sqrt(2 * Re * h1) + np.sqrt(2 * Re * h2)

    def radar_equation(self,
                       power_w: float,
                       gain: float,
                       rcs_m2: np.ndarray,
                       range_m: np.ndarray,
                       losses: float = 1.0,
                       include_multipath: bool = True,
                       target_height_m: float = 5.0) -> np.ndarray:
        """
        Standard radar equation with optional multipath.

        Pr = (Pt * G^2 * lambda^2 * sigma * F^4) / ((4*pi)^3 * R^4 * L)

        Args:
            power_w: Transmit power in watts
            gain: Antenna gain (linear)
            rcs_m2: Target RCS in m^2
            range_m: Range in meters
            losses: System losses (linear, > 1)
            include_multipath: Whether to include two-ray multipath
            target_height_m: Target height for multipath calculation

        Returns:
            Received power in watts
        """
        lam = self.config.wavelength_m

        # Basic radar equation
        numerator = power_w * (gain ** 2) * (lam ** 2) * rcs_m2
        denominator = ((4 * np.pi) ** 3) * (range_m ** 4) * losses

        Pr = numerator / denominator

        # Apply multipath factor
        if include_multipath:
            F4 = self.two_ray_propagation_factor(range_m, target_height_m)
            Pr = Pr * F4

        return Pr

    def snr_single_pulse(self,
                         power_w: float,
                         gain: float,
                         rcs_m2: np.ndarray,
                         range_m: np.ndarray,
                         noise_figure: float = 4.0,
                         bandwidth_hz: float = 25e6,
                         losses: float = 2.0,
                         temp_k: float = 290.0) -> np.ndarray:
        """
        Single-pulse SNR.

        Args:
            power_w: Transmit power (watts)
            gain: Antenna gain (linear)
            rcs_m2: Target RCS (m^2)
            range_m: Range (meters)
            noise_figure: Receiver noise figure (linear)
            bandwidth_hz: Receiver bandwidth (Hz)
            losses: System losses (linear)
            temp_k: System temperature (Kelvin)

        Returns:
            SNR (linear)
        """
        # Received signal power
        Pr = self.radar_equation(power_w, gain, rcs_m2, range_m, losses)

        # Noise power
        k_boltz = 1.38e-23  # Boltzmann constant
        Pn = k_boltz * temp_k * noise_figure * bandwidth_hz

        return Pr / Pn

    def atmospheric_attenuation(self, range_m: np.ndarray,
                                 attenuation_db_per_km: float = 0.01) -> np.ndarray:
        """
        One-way atmospheric attenuation.

        Args:
            range_m: Range in meters
            attenuation_db_per_km: Attenuation rate (typical X-band clear air: 0.01)

        Returns:
            Attenuation factor (linear, < 1)
        """
        range_km = range_m / 1000
        atten_db = attenuation_db_per_km * range_km
        return 10 ** (-atten_db / 10)


class DuctingModel:
    """
    Simplified atmospheric ducting model.

    Models anomalous propagation due to temperature inversions.
    """

    def __init__(self,
                 duct_height_m: float = 50.0,
                 duct_strength: float = 0.5):
        """
        Args:
            duct_height_m: Height of ducting layer
            duct_strength: 0 = no ducting, 1 = strong ducting
        """
        self.duct_height = duct_height_m
        self.strength = np.clip(duct_strength, 0, 1)

    def trapping_factor(self,
                        antenna_height_m: float,
                        target_height_m: float) -> float:
        """
        Compute trapping factor for targets within duct.

        Returns enhancement factor for propagation (> 1 if trapped).
        """
        # Both antenna and target must be within duct
        if antenna_height_m > self.duct_height or target_height_m > self.duct_height:
            return 1.0

        # Enhancement based on how deep in duct
        depth_factor = min(antenna_height_m, target_height_m) / self.duct_height

        return 1.0 + self.strength * (1 - depth_factor) * 10  # Up to 10x enhancement

    def extended_horizon(self, base_horizon_m: float) -> float:
        """
        Compute extended horizon range due to ducting.
        """
        return base_horizon_m * (1 + self.strength * 2)  # Up to 3x horizon extension
