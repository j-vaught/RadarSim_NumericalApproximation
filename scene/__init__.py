"""
Scene generation: motion paths, intensity profiles, and configuration.
"""

from .config import (
    SceneConfig,
    ObjectGroupConfig,
    PathConfig,
    IntensityConfig,
    FlickerConfig,
    LabelConfig,
    PlacementConfig,
    PhysicsConfig,
    EnvironmentConfig,
    LandSceneConfig,
    DuctingConfig,
    load_scene,
)
from .intensity import IntensityEngine, scale_echo_data
from .motion import generate_path, safe_eval, eval_speed, resolve_position
from .adapter import SceneAdapter, NumpyRngAdapter, SceneTrack

__all__ = [
    # config
    'SceneConfig', 'ObjectGroupConfig', 'PathConfig', 'IntensityConfig',
    'FlickerConfig', 'LabelConfig', 'PlacementConfig', 'PhysicsConfig',
    'EnvironmentConfig', 'LandSceneConfig', 'DuctingConfig', 'load_scene',
    # intensity
    'IntensityEngine', 'scale_echo_data',
    # motion
    'generate_path', 'safe_eval', 'eval_speed', 'resolve_position',
    # adapter
    'SceneAdapter', 'NumpyRngAdapter', 'SceneTrack',
]
