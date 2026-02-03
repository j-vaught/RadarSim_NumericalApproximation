"""Stage 3: Write augmented CSVs and polar YOLO labels."""

import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

from ..core import NUM_RANGE_BINS, RadarCSVHandler, angle_diff


def place_object(frame, echo_data, center_pulse, center_bin, obj_width, obj_height, num_pulses):
    """Blend object echo data into a radar frame using max-blending."""
    pulse_offset = obj_width // 2
    bin_offset = obj_height // 2

    for pulse_rel, row in enumerate(echo_data):
        target_pulse = (max(0, center_pulse + pulse_rel - pulse_offset)) % num_pulses
        target_angle = (target_pulse / num_pulses) * 2.0 * math.pi

        frame_pulse = _find_nearest_pulse_mut(frame, target_angle)
        if frame_pulse is None:
            continue

        for bin_rel, echo in enumerate(row):
            target_bin = center_bin + bin_rel - min(bin_offset, center_bin)
            if 0 <= target_bin < NUM_RANGE_BINS and echo > 0:
                current = int(frame_pulse.echoes[target_bin])
                frame_pulse.echoes[target_bin] = max(current, echo)


def _find_nearest_pulse_mut(frame, target_angle):
    if not frame.pulses:
        return None
    return min(frame.pulses, key=lambda p: angle_diff(p.angle_rad, target_angle))


def write_polar_labels(label_path, placements, num_pulses):
    """Write YOLO label file in polar coordinates."""
    with open(label_path, "w") as f:
        for p in placements:
            x_center = p.pulse_idx / num_pulses
            y_center = p.bin_idx / NUM_RANGE_BINS
            width = p.width / num_pulses
            height = p.height / NUM_RANGE_BINS
            f.write(f"{p.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def write_csv_and_labels(frame_plans, output_dir, num_pulses,
                         export_labels=True):
    """Write augmented CSVs and optional polar YOLO labels.

    Args:
        frame_plans: List[FramePlan] from scene_generator
        output_dir: Base output directory
        num_pulses: Number of radar pulses (720)
        export_labels: Whether to write YOLO label files

    Returns:
        (csv_dir, polar_labels_dir) paths. polar_labels_dir is None if export_labels=False.
    """
    csv_dir = Path(output_dir) / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = None
    if export_labels:
        labels_dir = Path(output_dir) / "RAW_labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

    total = len(frame_plans)
    for plan in frame_plans:
        frame = RadarCSVHandler.read_csv_uncached(plan.background_path)

        for p in plan.placements:
            place_object(frame, p.echo_data, p.pulse_idx, p.bin_idx,
                         p.width, p.height, num_pulses)

        output_path = csv_dir / f"augmented_{plan.frame_idx:04d}.csv"
        RadarCSVHandler.write_csv(frame, str(output_path))

        if labels_dir and plan.placements:
            label_path = labels_dir / f"augmented_{plan.frame_idx:04d}.txt"
            write_polar_labels(str(label_path), plan.placements, num_pulses)

        if (plan.frame_idx + 1) % 10 == 0 or plan.frame_idx + 1 == total:
            print(f"  Written {plan.frame_idx + 1}/{total} CSVs")

    return str(csv_dir), str(labels_dir) if labels_dir else None
