"""
Core data structures for radar data handling.
"""

from .csv_handler import (
    NUM_RANGE_BINS,
    Pulse,
    RadarFrame,
    RadarCSVHandler,
    ticks_to_radians,
    radians_to_ticks,
    angle_diff,
)
from .converter import RadarConverter, ConversionConfig
from .colormap import Colormap, apply_colormap, apply_colormap_to_image
from .mask import BinaryMask, PolarMask, create_water_mask, create_land_mask
from .annotations import AnnotationSet, Region, RegionType, Point, Polygon

__all__ = [
    # csv_handler
    'NUM_RANGE_BINS', 'Pulse', 'RadarFrame', 'RadarCSVHandler',
    'ticks_to_radians', 'radians_to_ticks', 'angle_diff',
    # converter
    'RadarConverter', 'ConversionConfig',
    # colormap
    'Colormap', 'apply_colormap', 'apply_colormap_to_image',
    # mask
    'BinaryMask', 'PolarMask', 'create_water_mask', 'create_land_mask',
    # annotations
    'AnnotationSet', 'Region', 'RegionType', 'Point', 'Polygon',
]
