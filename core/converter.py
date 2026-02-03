"""Polar <-> Cartesian Coordinate Conversion

Converts between radar polar coordinates (angle, range) and
Cartesian image coordinates (x, y) for visualization and annotation.
"""

import math
from dataclasses import dataclass

import numpy as np

from .csv_handler import NUM_RANGE_BINS, RadarFrame
from .colormap import apply_colormap, Colormap


@dataclass
class ConversionConfig:
    image_size: int = 1735
    num_pulses: int = 720
    max_gap_degrees: float = 1.0


class RadarConverter:
    def __init__(self, config: ConversionConfig = None):
        if config is None:
            config = ConversionConfig()
        self.config = config
        self._pulse_idx_lut: np.ndarray = None
        self._bin_idx_lut: np.ndarray = None
        self._valid_lut: np.ndarray = None
        self._build_lut()

    def _build_lut(self):
        size = self.config.image_size
        num_pulses = self.config.num_pulses
        center = size / 2.0
        max_radius = center

        ys = np.arange(size, dtype=np.float64)
        xs = np.arange(size, dtype=np.float64)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')

        dx = xx - center
        dy = center - yy

        r = np.sqrt(dx * dx + dy * dy)
        r_norm = r / max_radius

        valid = r_norm <= 1.0

        theta = np.arctan2(dx, dy)
        theta_norm = np.where(theta < 0, theta + 2.0 * math.pi, theta)

        pulse_idx = ((theta_norm / (2.0 * math.pi)) * num_pulses).astype(np.int32) % num_pulses
        bin_idx = (r_norm * NUM_RANGE_BINS).astype(np.int32)
        bin_idx = np.minimum(bin_idx, NUM_RANGE_BINS - 1)

        self._pulse_idx_lut = pulse_idx.ravel()
        self._bin_idx_lut = bin_idx.ravel()
        self._valid_lut = valid.ravel()

    def polar_to_cartesian(self, frame: RadarFrame, colormap: Colormap = Colormap.VIRIDIS) -> np.ndarray:
        size = self.config.image_size
        num_pulses = self.config.num_pulses
        regularized = frame.regularize(num_pulses, self.config.max_gap_degrees)
        echo_data = regularized.to_regularized_array(num_pulses)

        image = np.zeros((size * size, 4), dtype=np.uint8)
        valid = self._valid_lut
        p_idx = self._pulse_idx_lut[valid]
        b_idx = self._bin_idx_lut[valid]

        intensities = echo_data[p_idx, b_idx]
        colors = np.array([apply_colormap(i, colormap) for i in intensities], dtype=np.uint8)
        valid_indices = np.where(valid)[0]
        image[valid_indices, 0] = colors[:, 0]
        image[valid_indices, 1] = colors[:, 1]
        image[valid_indices, 2] = colors[:, 2]
        image[valid_indices, 3] = 255

        return image.reshape((size, size, 4))

    def polar_to_cartesian_gray(self, frame: RadarFrame) -> np.ndarray:
        size = self.config.image_size
        num_pulses = self.config.num_pulses
        regularized = frame.regularize(num_pulses, self.config.max_gap_degrees)
        echo_data = regularized.to_regularized_array(num_pulses)

        image = np.zeros(size * size, dtype=np.uint8)
        valid = self._valid_lut
        image[valid] = echo_data[self._pulse_idx_lut[valid], self._bin_idx_lut[valid]]

        return image.reshape((size, size))

    def cartesian_to_polar(self, x: float, y: float):
        size = self.config.image_size
        center = size / 2.0
        max_radius = center

        dx = x - center
        dy = center - y

        r = math.sqrt(dx * dx + dy * dy)
        r_norm = r / max_radius

        if r_norm > 1.0:
            return None

        theta = math.atan2(dx, dy)
        if theta < 0:
            theta += 2.0 * math.pi

        bin_idx = int(r_norm * NUM_RANGE_BINS)
        bin_idx = min(bin_idx, NUM_RANGE_BINS - 1)

        return (theta, bin_idx)

    def polar_to_cartesian_point(self, angle_rad: float, range_bin: int):
        size = self.config.image_size
        center = size / 2.0
        max_radius = center

        r_norm = range_bin / NUM_RANGE_BINS
        r = r_norm * max_radius

        x = center + r * math.sin(angle_rad)
        y = center - r * math.cos(angle_rad)
        return (x, y)

    def cartesian_mask_to_polar(self, mask_flat: np.ndarray) -> np.ndarray:
        size = self.config.image_size
        num_pulses = self.config.num_pulses

        assert mask_flat.size == size * size, "Mask size must match image size"

        polar_mask = np.zeros((num_pulses, NUM_RANGE_BINS), dtype=bool)
        valid = self._valid_lut
        valid_mask = mask_flat[valid] if mask_flat.dtype == bool else mask_flat.astype(bool)[valid]
        p_idx = self._pulse_idx_lut[valid]
        b_idx = self._bin_idx_lut[valid]

        # Set polar mask where both valid and mask are true
        true_indices = valid_mask
        polar_mask[p_idx[true_indices], b_idx[true_indices]] = True

        return polar_mask

    def polar_mask_to_cartesian(self, polar_mask: np.ndarray) -> np.ndarray:
        size = self.config.image_size
        mask = np.zeros(size * size, dtype=bool)
        valid = self._valid_lut
        mask[valid] = polar_mask[self._pulse_idx_lut[valid], self._bin_idx_lut[valid]]
        return mask

    def image_size(self) -> int:
        return self.config.image_size

    def num_pulses(self) -> int:
        return self.config.num_pulses

    def polar_bbox_to_cartesian_bbox(self, pulse_center, bin_center,
                                      pulse_width, bin_height):
        """Convert normalized polar YOLO bbox to normalized Cartesian YOLO bbox.

        Args:
            pulse_center: normalized pulse center (0-1)
            bin_center: normalized bin center (0-1)
            pulse_width: normalized pulse width (0-1)
            bin_height: normalized bin height (0-1)

        Returns:
            (cx, cy, w, h) normalized Cartesian YOLO bbox, or None if outside coverage.
        """
        num_pulses = self.config.num_pulses
        # Denormalize to pulse/bin indices
        p_center = pulse_center * num_pulses
        b_center = bin_center * NUM_RANGE_BINS
        p_half = (pulse_width * num_pulses) / 2.0
        b_half = (bin_height * NUM_RANGE_BINS) / 2.0

        # 4 corners in polar
        corners_polar = [
            (p_center - p_half, b_center - b_half),
            (p_center + p_half, b_center - b_half),
            (p_center - p_half, b_center + b_half),
            (p_center + p_half, b_center + b_half),
        ]

        # Convert each corner to Cartesian
        size = self.config.image_size
        xs, ys = [], []
        for p, b in corners_polar:
            angle_rad = (p / num_pulses) * 2.0 * math.pi
            bin_idx = max(0, min(int(b), NUM_RANGE_BINS - 1))
            cx, cy = self.polar_to_cartesian_point(angle_rad, bin_idx)
            xs.append(cx)
            ys.append(cy)

        # Axis-aligned bounding box
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Normalize by image size
        cart_cx = ((min_x + max_x) / 2.0) / size
        cart_cy = ((min_y + max_y) / 2.0) / size
        cart_w = (max_x - min_x) / size
        cart_h = (max_y - min_y) / size

        return (cart_cx, cart_cy, cart_w, cart_h)

    def is_within_coverage(self, x: float, y: float) -> bool:
        size = self.config.image_size
        center = size / 2.0
        dx = x - center
        dy = y - center
        return math.sqrt(dx * dx + dy * dy) <= center
