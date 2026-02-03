"""
Radar system configuration.

Contains all parameters defining a radar system, including transmitter,
antenna, receiver, and signal processing settings.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class RadarConfig:
    """Complete radar system configuration.
    
    All parameters needed to simulate radar operation, from transmission
    through reception and signal processing.
    
    Attributes:
        frequency: Operating frequency in Hz (e.g., 9.41e9 for X-band)
        peak_power: Transmitter peak power in Watts
        pulse_width: Pulse width in seconds
        prf: Pulse repetition frequency in Hz
        
        beamwidth_h: Horizontal beamwidth in degrees
        beamwidth_v: Vertical beamwidth in degrees
        gain_db: Antenna gain in dB
        sidelobe_level_db: First sidelobe level in dB (negative, below main lobe)
        rotation_rate: Antenna rotation rate in RPM
        antenna_height: Antenna height above waterline in meters
        
        noise_figure_db: Receiver noise figure in dB
        bandwidth: Receiver bandwidth in Hz
        dynamic_range_db: Receiver dynamic range in dB
        
        range_resolution: Range resolution in meters
        num_range_bins: Number of range bins
        samples_per_revolution: Number of azimuth samples per rotation
        
        sea_clutter_filter: Sea clutter mitigation ("none", "STC", "CFAR")
        rain_clutter_filter: Rain clutter mitigation ("none", "FTC")
        interference_rejection: Whether interference rejection is enabled
    """
    
    # Transmitter
    frequency: float = 9.41e9  # Hz, X-band default
    peak_power: float = 25.0  # Watts (solid-state)
    pulse_width: float = 0.5e-6  # seconds
    prf: float = 2100.0  # Hz
    
    # Antenna
    beamwidth_h: float = 3.9  # degrees
    beamwidth_v: float = 25.0  # degrees
    gain_db: float = 26.0  # dB
    sidelobe_level_db: float = -18.0  # dB below main lobe
    rotation_rate: float = 24.0  # RPM
    antenna_height: float = 4.0  # meters above waterline
    
    # Receiver
    noise_figure_db: float = 4.0  # dB
    bandwidth: float = 25e6  # Hz
    dynamic_range_db: float = 80.0  # dB
    
    # Signal processing
    range_resolution: float = 6.0  # meters
    num_range_bins: int = 2048
    samples_per_revolution: int = 720
    
    # Signal Synthesis
    adc_sample_rate: Optional[float] = None # Defaults to 2*bandwidth

    
    # Clutter mitigation
    sea_clutter_filter: str = "none"
    rain_clutter_filter: str = "none"
    interference_rejection: bool = False
    
    # Derived properties
    @property
    def wavelength(self) -> float:
        """Wavelength in meters."""
        return 3.0e8 / self.frequency
    
    @property
    def gain(self) -> float:
        """Antenna gain as linear ratio."""
        return 10.0 ** (self.gain_db / 10.0)
    
    @property
    def noise_figure(self) -> float:
        """Noise figure as linear ratio."""
        return 10.0 ** (self.noise_figure_db / 10.0)
    
    @property
    def pulse_interval(self) -> float:
        """Time between pulses in seconds (1/PRF)."""
        return 1.0 / self.prf
    
    @property
    def azimuth_resolution(self) -> float:
        """Azimuth resolution in degrees."""
        return 360.0 / self.samples_per_revolution
    
    @property
    def sample_rate(self) -> float:
        """ADC sample rate in Hz."""
        if self.adc_sample_rate is not None:
             return self.adc_sample_rate
        return 2.0 * self.bandwidth

    
    @property
    def max_range(self) -> float:
        """Maximum instrumented range in meters."""
        return self.range_resolution * self.num_range_bins
    
    @property
    def max_unambiguous_range(self) -> float:
        """Maximum unambiguous range based on PRF in meters."""
        return 3.0e8 / (2.0 * self.prf)
    
    @property
    def rotation_period(self) -> float:
        """Time for one complete rotation in seconds."""
        return 60.0 / self.rotation_rate
    
    @property
    def dwell_time(self) -> float:
        """Time the beam illuminates a point target in seconds."""
        return (self.beamwidth_h / 360.0) * self.rotation_period
    
    @property
    def pulses_per_dwell(self) -> int:
        """Number of pulses during beam dwell on target."""
        return int(self.dwell_time * self.prf)
    
    @property
    def spoke_interval(self) -> float:
        """Time between adjacent azimuth samples (spokes) in seconds."""
        return self.rotation_period / self.samples_per_revolution
    
    @property
    def thermal_noise_power(self) -> float:
        """Thermal noise power in Watts.
        
        N = kTB * F where:
        - k = 1.38e-23 J/K (Boltzmann constant)
        - T = 290 K (standard temperature)
        - B = bandwidth
        - F = noise figure
        """
        k_boltzmann = 1.38e-23
        T_standard = 290.0  # Kelvin
        return k_boltzmann * T_standard * self.bandwidth * self.noise_figure
    
    def radar_equation(self, range_m: float, rcs_m2: float) -> float:
        """Calculate received power using radar equation.
        
        P_r = (P_t * G^2 * λ^2 * σ) / ((4π)^3 * R^4)
        
        Args:
            range_m: Range to target in meters
            rcs_m2: Radar cross section in m²
            
        Returns:
            Received power in Watts
        """
        if range_m <= 0:
            return 0.0
        
        numerator = self.peak_power * (self.gain ** 2) * (self.wavelength ** 2) * rcs_m2
        denominator = ((4.0 * np.pi) ** 3) * (range_m ** 4)
        
        return numerator / denominator
    
    def snr(self, range_m: float, rcs_m2: float) -> float:
        """Calculate signal-to-noise ratio.
        
        Args:
            range_m: Range to target in meters
            rcs_m2: Radar cross section in m²
            
        Returns:
            SNR as linear ratio
        """
        p_received = self.radar_equation(range_m, rcs_m2)
        return p_received / self.thermal_noise_power
    
    def snr_db(self, range_m: float, rcs_m2: float) -> float:
        """Calculate signal-to-noise ratio in dB.
        
        Args:
            range_m: Range to target in meters
            rcs_m2: Radar cross section in m²
            
        Returns:
            SNR in dB
        """
        snr_linear = self.snr(range_m, rcs_m2)
        if snr_linear <= 0:
            return -np.inf
        return 10.0 * np.log10(snr_linear)
    
    def range_to_bin(self, range_m: float) -> int:
        """Convert range to range bin index.
        
        Args:
            range_m: Range in meters
            
        Returns:
            Range bin index (0-indexed)
        """
        return int(range_m / self.range_resolution)
    
    def bin_to_range(self, bin_idx: int) -> float:
        """Convert range bin index to range.
        
        Args:
            bin_idx: Range bin index (0-indexed)
            
        Returns:
            Range in meters (center of bin)
        """
        return (bin_idx + 0.5) * self.range_resolution
    
    def azimuth_to_spoke(self, azimuth_deg: float) -> int:
        """Convert azimuth angle to spoke index.
        
        Args:
            azimuth_deg: Azimuth in degrees (0=North, clockwise)
            
        Returns:
            Spoke index (0-indexed)
        """
        azimuth_normalized = azimuth_deg % 360.0
        return int(azimuth_normalized / self.azimuth_resolution) % self.samples_per_revolution
    
    def spoke_to_azimuth(self, spoke_idx: int) -> float:
        """Convert spoke index to azimuth angle.
        
        Args:
            spoke_idx: Spoke index (0-indexed)
            
        Returns:
            Azimuth in degrees (0=North, clockwise)
        """
        return (spoke_idx + 0.5) * self.azimuth_resolution


# Preset configurations for common radars

FurunoNXT = RadarConfig(
    # Furuno DRS4D-NXT Solid-State Doppler Radar
    frequency=9.41e9,  # X-band
    peak_power=25.0,  # 25W solid-state
    pulse_width=0.5e-6,  # 0.5 μs short pulse
    prf=2100.0,  # Approximate
    
    beamwidth_h=3.9,  # Standard radome
    beamwidth_v=25.0,
    gain_db=26.0,
    sidelobe_level_db=-18.0,
    rotation_rate=24.0,  # 24 or 48 RPM
    antenna_height=4.0,  # Typical yacht installation
    
    noise_figure_db=4.0,
    bandwidth=25e6,
    dynamic_range_db=80.0,
    
    range_resolution=6.0,  # Depends on range scale
    num_range_bins=2048,
    samples_per_revolution=720,  # 0.5° per sample
    
    sea_clutter_filter="none",  # No interference rejection on this model
    rain_clutter_filter="none",
    interference_rejection=False,  # Critical: this is OFF per user specs
)
