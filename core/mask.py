"""Binary Mask Operations

Utilities for creating and manipulating binary masks
from polygon annotations for water/land region extraction.
"""

import math
import numpy as np
from typing import TYPE_CHECKING

from .annotations import AnnotationSet, Point, RegionType
from .csv_handler import NUM_RANGE_BINS, RadarFrame

if TYPE_CHECKING:
    from .converter import RadarConverter


class BinaryMask:
    def __init__(self, width: int, height: int, fill: bool = False):
        self.width = width
        self.height = height
        self.data = np.full((height, width), fill, dtype=bool)

    @classmethod
    def empty(cls, width: int, height: int) -> "BinaryMask":
        return cls(width, height, False)

    @classmethod
    def full(cls, width: int, height: int) -> "BinaryMask":
        return cls(width, height, True)

    def get(self, x: int, y: int) -> bool:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return bool(self.data[y, x])

    def set(self, x: int, y: int, value: bool):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.data[y, x] = value

    def get_normalized(self, x: float, y: float) -> bool:
        px = int(x * self.width)
        py = int(y * self.height)
        return self.get(px, py)

    def set_normalized(self, x: float, y: float, value: bool):
        px = int(x * self.width)
        py = int(y * self.height)
        self.set(px, py, value)

    def fill_polygon(self, polygon, value: bool = True):
        """Fill a polygon region. Uses point-in-polygon test matching the Rust version."""
        min_pt, max_pt = polygon.bounding_box()
        min_x = int(math.floor(min_pt.x * self.width))
        min_y = int(math.floor(min_pt.y * self.height))
        max_x = int(math.ceil(max_pt.x * self.width))
        max_y = int(math.ceil(max_pt.y * self.height))

        for y in range(max(min_y, 0), min(max_y, self.height)):
            for x in range(max(min_x, 0), min(max_x, self.width)):
                norm_x = x / self.width
                norm_y = y / self.height
                point = Point(norm_x, norm_y)
                if polygon.contains(point):
                    self.data[y, x] = value

    def fill_region(self, region, value: bool = True):
        self.fill_polygon(region.polygon, value)

    def union(self, other: "BinaryMask"):
        self.data = np.logical_or(self.data, other.data)

    def intersection(self, other: "BinaryMask"):
        self.data = np.logical_and(self.data, other.data)

    def subtract(self, other: "BinaryMask"):
        self.data = np.logical_and(self.data, np.logical_not(other.data))

    def invert(self):
        self.data = np.logical_not(self.data)

    def count_true(self) -> int:
        return int(np.sum(self.data))

    def count_false(self) -> int:
        return int(np.sum(~self.data))

    def coverage(self) -> float:
        return self.count_true() / self.data.size

    def to_rgba(self, true_color, false_color):
        rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        rgba[self.data] = true_color
        rgba[~self.data] = false_color
        return rgba

    @property
    def flat_data(self) -> np.ndarray:
        """Return flat bool array in row-major order (y*width+x) for converter compatibility."""
        return self.data.ravel()


def create_water_mask(annotations: AnnotationSet, width: int, height: int) -> BinaryMask:
    mask = BinaryMask.empty(width, height)
    water_regions = sorted(annotations.regions_by_type(RegionType.WATER), key=lambda r: r.z_order)
    for region in water_regions:
        mask.fill_region(region, True)

    land_regions = sorted(annotations.regions_by_type(RegionType.LAND), key=lambda r: r.z_order)
    for land_region in land_regions:
        land_mask = BinaryMask.empty(width, height)
        land_mask.fill_region(land_region, True)
        mask.subtract(land_mask)

    return mask


def create_land_mask(annotations: AnnotationSet, width: int, height: int) -> BinaryMask:
    mask = BinaryMask.empty(width, height)
    land_regions = sorted(annotations.regions_by_type(RegionType.LAND), key=lambda r: r.z_order)
    for region in land_regions:
        mask.fill_region(region, True)
    return mask


class PolarMask:
    def __init__(self, num_pulses: int, fill: bool = False):
        self.num_pulses = num_pulses
        self.num_bins = NUM_RANGE_BINS
        self.data = np.full((num_pulses, NUM_RANGE_BINS), fill, dtype=bool)

    @classmethod
    def from_cartesian(cls, cartesian: BinaryMask, converter: "RadarConverter") -> "PolarMask":
        polar_data = converter.cartesian_mask_to_polar(cartesian.flat_data)
        mask = cls(polar_data.shape[0], False)
        mask.data = polar_data
        return mask

    def to_cartesian(self, converter: "RadarConverter") -> BinaryMask:
        cart_flat = converter.polar_mask_to_cartesian(self.data)
        size = converter.image_size()
        mask = BinaryMask(size, size, False)
        mask.data = cart_flat.reshape((size, size))
        return mask

    def get(self, pulse_idx: int, bin_idx: int) -> bool:
        if 0 <= pulse_idx < self.num_pulses and 0 <= bin_idx < self.num_bins:
            return bool(self.data[pulse_idx, bin_idx])
        return False

    def set(self, pulse_idx: int, bin_idx: int, value: bool):
        if 0 <= pulse_idx < self.num_pulses and 0 <= bin_idx < self.num_bins:
            self.data[pulse_idx, bin_idx] = value

    def apply_to_frame(self, frame: RadarFrame, invert: bool = False) -> RadarFrame:
        import copy
        new_frame = RadarFrame(
            source_path=frame.source_path,
            timestamp=frame.timestamp,
            pulses=[],
            unique_angles=list(frame.unique_angles),
        )
        for pulse in frame.pulses:
            new_pulse = Pulse_copy(pulse)
            pulse_idx = int((pulse.angle_rad / (2.0 * math.pi)) * self.num_pulses) % self.num_pulses

            for bin_idx in range(len(new_pulse.echoes)):
                mask_val = self.get(pulse_idx, bin_idx)
                should_zero = mask_val if invert else not mask_val
                if should_zero:
                    new_pulse.echoes[bin_idx] = 0

            new_frame.pulses.append(new_pulse)
        return new_frame


def Pulse_copy(pulse):
    from .csv_handler import Pulse
    return Pulse(
        status=pulse.status,
        scale=pulse.scale,
        range_=pulse.range_,
        gain=pulse.gain,
        angle_ticks=pulse.angle_ticks,
        angle_rad=pulse.angle_rad,
        echoes=pulse.echoes.copy(),
    )
