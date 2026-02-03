"""
Tier 2: Analytical Radar Simulation

Fast statistical radar data generation using:
- K-distribution / Weibull sea clutter
- Swerling fluctuating target models
- Analytical radar equation (no ray tracing)

Usage:
    from Tier2 import Tier2Pipeline, Tier2Config
    from Tier2.models import SeaClutterModel, PointTarget

    config = Tier2Config(num_frames=100, sea_state=4)
    pipeline = Tier2Pipeline(config)
    pipeline.run()

CLI:
    python -m Tier2.pipeline --frames 100 --sea-state 4 --output ./output
"""

from .pipeline import Tier2Pipeline, Tier2Config
from .models import SeaClutterModel, SwerlingTarget, PointTarget, AnalyticalPropagation
from .radar import RadarConfig, FurunoNXT
from .core import RadarFrame, RadarCSVHandler, RadarConverter

__all__ = [
    'Tier2Pipeline',
    'Tier2Config',
    'SeaClutterModel',
    'SwerlingTarget',
    'PointTarget',
    'AnalyticalPropagation',
    'RadarConfig',
    'FurunoNXT',
    'RadarFrame',
    'RadarCSVHandler',
    'RadarConverter',
]
