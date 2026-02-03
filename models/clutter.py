"""
Statistical sea and rain clutter models for X-band radar.

References:
- Ward, K.D., Tough, R.J.A., Watts, S. "Sea Clutter: Scattering, the K Distribution and Radar Performance"
- Sekine, M., Mao, Y. "Weibull Radar Clutter"
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ClutterParams:
    """Parameters for clutter distribution."""
    shape: float      # Distribution shape parameter
    scale: float      # Mean power / scale
    correlation: float = 0.0  # Spatial correlation length (range bins)


class SeaClutterModel:
    """
    K-distribution and Weibull sea clutter models.

    The K-distribution models sea clutter as compound process:
    - Slow-varying texture (gamma distributed) representing wave bunching
    - Fast-varying speckle (exponential) representing individual scatterers
    """

    # Empirical parameters by sea state (X-band, low grazing angle)
    # Scaled to produce realistic signal-to-clutter ratios
    # Scale is in normalized intensity units (0-255 output range)
    SEA_STATE_PARAMS = {
        0: ClutterParams(shape=50.0, scale=1e-12),   # Glassy calm
        1: ClutterParams(shape=20.0, scale=5e-12),   # Calm
        2: ClutterParams(shape=10.0, scale=2e-11),   # Smooth
        3: ClutterParams(shape=5.0,  scale=1e-10),   # Slight
        4: ClutterParams(shape=2.0,  scale=5e-10),   # Moderate
        5: ClutterParams(shape=1.0,  scale=2e-9),    # Rough
        6: ClutterParams(shape=0.5,  scale=1e-8),    # Very rough
        7: ClutterParams(shape=0.2,  scale=5e-8),    # High
    }

    def __init__(self, sea_state: int = 3, grazing_angle_deg: float = 1.0):
        """
        Initialize sea clutter model.

        Args:
            sea_state: Sea state 0-7 (Douglas scale)
            grazing_angle_deg: Radar grazing angle in degrees
        """
        self.sea_state = np.clip(sea_state, 0, 7)
        self.grazing_angle_deg = grazing_angle_deg
        self.params = self.SEA_STATE_PARAMS[self.sea_state]

        # Adjust for grazing angle (clutter increases at low angles)
        self._apply_grazing_correction()

    def _apply_grazing_correction(self):
        """Adjust parameters based on grazing angle."""
        # Empirical correction: clutter power ~ sin^(-n)(grazing)
        # n typically 1-2 for X-band
        psi = np.radians(max(self.grazing_angle_deg, 0.1))
        correction = (np.sin(np.radians(5.0)) / np.sin(psi)) ** 1.5
        self.params.scale *= correction

    def k_distribution(self, shape: Tuple[int, ...],
                       nu: Optional[float] = None,
                       mean_power: Optional[float] = None) -> np.ndarray:
        """
        Generate K-distributed clutter samples.

        The K-distribution PDF is:
        p(x) = (2/Gamma(nu)) * (nu*x/mu)^(nu/2) * K_{nu-1}(2*sqrt(nu*x/mu))

        where K_n is modified Bessel function of second kind.

        Args:
            shape: Output array shape (azimuths, range_bins)
            nu: Shape parameter (None = use sea state default)
            mean_power: Mean clutter power (None = use default)

        Returns:
            K-distributed intensity values
        """
        nu = nu if nu is not None else self.params.shape
        mu = mean_power if mean_power is not None else self.params.scale

        # Compound model: texture * speckle
        # Texture ~ Gamma(nu, 1/nu) so E[texture] = 1
        texture = np.random.gamma(nu, 1.0/nu, shape)

        # Speckle ~ Exponential(mu * texture)
        # Conditioned on texture, this gives K-distribution
        clutter = np.random.exponential(mu * texture)

        return clutter

    def weibull(self, shape: Tuple[int, ...],
                c: Optional[float] = None,
                b: Optional[float] = None) -> np.ndarray:
        """
        Generate Weibull-distributed clutter samples.

        PDF: p(x) = (c/b) * (x/b)^(c-1) * exp(-(x/b)^c)

        Args:
            shape: Output array shape
            c: Shape parameter (None = derive from sea state)
            b: Scale parameter (None = derive from sea state)

        Returns:
            Weibull-distributed intensity values
        """
        # Convert K-distribution params to approximate Weibull
        if c is None:
            c = 1.0 + 0.1 * self.params.shape  # Empirical mapping
        if b is None:
            b = self.params.scale

        return b * np.random.weibull(c, shape)

    def correlated_clutter(self, shape: Tuple[int, int],
                           correlation_range: float = 5.0) -> np.ndarray:
        """
        Generate spatially correlated K-distributed clutter.

        Uses convolution with exponential correlation kernel.

        Args:
            shape: (num_azimuths, num_range_bins)
            correlation_range: Correlation length in range bins

        Returns:
            Correlated K-distributed clutter
        """
        num_az, num_range = shape

        # Generate texture with correlation
        texture_uncorr = np.random.gamma(
            self.params.shape, 1.0/self.params.shape, shape
        )

        # Exponential correlation kernel (range direction)
        kernel_size = int(3 * correlation_range)
        kernel = np.exp(-np.abs(np.arange(-kernel_size, kernel_size+1)) / correlation_range)
        kernel /= kernel.sum()

        # Apply correlation in range
        from scipy.ndimage import convolve1d
        texture = convolve1d(texture_uncorr, kernel, axis=1, mode='wrap')

        # Speckle (uncorrelated)
        clutter = np.random.exponential(self.params.scale * texture)

        return clutter

    def get_sigma0(self, range_m: np.ndarray, wind_speed_mps: float = 10.0) -> np.ndarray:
        """
        Get normalized radar cross section (sigma0) vs range.

        Uses GIT model for X-band sea clutter.

        Args:
            range_m: Range values in meters
            wind_speed_mps: Wind speed in m/s

        Returns:
            sigma0 in linear units (m^2/m^2)
        """
        # Grazing angle vs range (flat earth approximation)
        # psi = arctan(h/R) for antenna height h
        antenna_height = 10.0  # meters, typical
        psi = np.arctan(antenna_height / range_m)

        # GIT model: sigma0 = a * psi^b * U^c
        # X-band coefficients (approximate)
        a = 1e-6
        b = 1.5
        c = 2.0

        sigma0 = a * (psi ** b) * (wind_speed_mps ** c)
        return sigma0


class RainClutterModel:
    """
    Rain clutter model for X-band radar.

    Rain rate -> reflectivity -> received power
    """

    def __init__(self, rain_rate_mmhr: float = 10.0):
        """
        Args:
            rain_rate_mmhr: Rain rate in mm/hour
        """
        self.rain_rate = rain_rate_mmhr

    def reflectivity_factor(self) -> float:
        """
        Compute reflectivity factor Z (mm^6/m^3).

        Uses Marshall-Palmer Z-R relationship:
        Z = 200 * R^1.6
        """
        return 200 * (self.rain_rate ** 1.6)

    def attenuation_db_per_km(self) -> float:
        """
        X-band specific attenuation through rain.

        Approximately 0.01 * R^1.2 dB/km for X-band
        """
        return 0.01 * (self.rain_rate ** 1.2)

    def generate(self, shape: Tuple[int, ...],
                 range_bins: np.ndarray,
                 range_resolution_m: float) -> np.ndarray:
        """
        Generate rain clutter volume.

        Args:
            shape: Output shape
            range_bins: Range bin indices
            range_resolution_m: Range resolution

        Returns:
            Rain clutter power
        """
        Z = self.reflectivity_factor()

        # Convert to linear reflectivity
        eta = Z * 1e-18  # Convert from mm^6/m^3

        # Volume clutter: P ~ eta * range_res / R^2
        range_m = range_bins * range_resolution_m + range_resolution_m
        volume_factor = range_resolution_m / (range_m ** 2)

        # Add random variation (log-normal)
        variation = np.random.lognormal(0, 0.5, shape)

        return eta * volume_factor * variation
