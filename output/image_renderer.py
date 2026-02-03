"""Stage 4: Convert CSVs to PNG images with Cartesian YOLO labels."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from ..core import ConversionConfig, RadarConverter, NUM_RANGE_BINS, RadarCSVHandler


def _render_single(csv_path, output_path, converter):
    """Render a single CSV to a grayscale PNG."""
    frame = RadarCSVHandler.read_csv_uncached(str(csv_path))
    gray = converter.polar_to_cartesian_gray(frame)
    img = Image.fromarray(gray, mode='L')
    img.save(str(output_path))


def _convert_labels(polar_label_path, cart_label_path, converter, num_pulses):
    """Convert a polar YOLO label file to Cartesian coordinates."""
    lines = Path(polar_label_path).read_text().strip().splitlines()
    if not lines:
        return

    with open(cart_label_path, "w") as f:
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = parts[0]
            px, py, pw, ph = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            result = converter.polar_bbox_to_cartesian_bbox(px, py, pw, ph)
            if result is not None:
                cx, cy, cw, ch = result
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {cw:.6f} {ch:.6f}\n")


def render_images(csv_dir, output_dir, polar_labels_dir=None,
                  converter=None, threads=8):
    """Render all CSVs in csv_dir to PNG images.

    Args:
        csv_dir: Directory containing augmented CSV files
        output_dir: Base output directory
        polar_labels_dir: Directory with polar YOLO labels (optional)
        converter: RadarConverter instance (created if None)
        threads: Number of worker threads

    Returns:
        (images_dir, cart_labels_dir) paths. cart_labels_dir is None if no labels.
    """
    if converter is None:
        converter = RadarConverter(ConversionConfig())

    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(Path(csv_dir).glob("augmented_*.csv"))
    if not csv_files:
        csv_files = sorted(Path(csv_dir).glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return str(images_dir), None

    print(f"Rendering {len(csv_files)} images...")

    # Render images (single-threaded due to shared converter LUT)
    for i, csv_path in enumerate(csv_files):
        out_path = images_dir / f"frame_{i:04d}.png"
        _render_single(csv_path, out_path, converter)

        if (i + 1) % 10 == 0 or i + 1 == len(csv_files):
            print(f"  Rendered {i + 1}/{len(csv_files)} images")

    # Convert labels if available
    cart_labels_dir = None
    if polar_labels_dir:
        polar_dir = Path(polar_labels_dir)
        label_files = sorted(polar_dir.glob("augmented_*.txt"))

        if label_files:
            cart_labels_dir_path = Path(output_dir) / "YOLO_labels"
            cart_labels_dir_path.mkdir(parents=True, exist_ok=True)
            cart_labels_dir = str(cart_labels_dir_path)

            num_pulses = converter.num_pulses()
            for i, label_path in enumerate(label_files):
                cart_path = cart_labels_dir_path / f"frame_{i:04d}.txt"
                _convert_labels(str(label_path), str(cart_path), converter, num_pulses)

            print(f"  Converted {len(label_files)} label files to Cartesian")

    print(f"Images saved to: {images_dir}")
    return str(images_dir), cart_labels_dir
