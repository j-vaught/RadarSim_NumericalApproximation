"""
Radar Waveform Generation.

Defines transmit waveforms for high-fidelity signal simulation.
"""

from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

@dataclass
class Waveform(ABC):
    """Base class for radar waveforms."""
    
    @abstractmethod
    def sample(self, fs: float) -> np.ndarray:
        """Generate time-domain samples of the waveform.
        
        Args:
            fs: Sampling frequency in Hz
            
        Returns:
            Complex IQ samples
        """
        pass
    
    @property
    @abstractmethod
    def bandwidth(self) -> float:
        pass
    
    @property
    @abstractmethod
    def pulse_width(self) -> float:
        pass


@dataclass
class LFMChirp(Waveform):
    """Linear Frequency Modulation (LFM) Chirp."""
    bw: float          # Bandwidth in Hz
    pw: float          # Pulse width in seconds
    center_freq: float = 0.0 # Baseband center frequency
    
    def sample(self, fs: float) -> np.ndarray:
        """Generate LFM Chirp samples."""
        num_samples = int(self.pw * fs)
        t = np.arange(num_samples) / fs - (self.pw / 2) # Centered at 0
        
        # Chirp rate
        alpha = self.bw / self.pw
        
        # Signal: exp(j * pi * alpha * t^2)
        # Add center freq offset if needed
        phase = np.pi * alpha * t**2 + 2 * np.pi * self.center_freq * t
        
        return np.exp(1j * phase)

    @property
    def bandwidth(self) -> float:
        return self.bw
        
    @property
    def pulse_width(self) -> float:
        return self.pw
