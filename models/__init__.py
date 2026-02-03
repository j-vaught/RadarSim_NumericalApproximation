"""
Tier 2 Statistical Models

- clutter: K-distribution, Weibull sea clutter, rain clutter
- targets: Swerling RCS fluctuation models (cases 0-4), point targets
- propagation: Analytical path loss, 2-ray multipath, radar equation
"""

from .clutter import SeaClutterModel, RainClutterModel, ClutterParams
from .targets import SwerlingTarget, PointTarget, SwerlingCase, create_target_ensemble, TYPICAL_RCS
from .propagation import AnalyticalPropagation, PropagationConfig, DuctingModel
from .land import LandGenerator, LandConfig, create_harbor_coastline, create_peninsula_coastline, create_island

__all__ = [
    # Clutter
    'SeaClutterModel',
    'RainClutterModel',
    'ClutterParams',
    # Targets
    'SwerlingTarget',
    'SwerlingCase',
    'PointTarget',
    'create_target_ensemble',
    'TYPICAL_RCS',
    # Propagation
    'AnalyticalPropagation',
    'PropagationConfig',
    'DuctingModel',
]
