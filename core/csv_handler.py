"""FURUNO Radar CSV Handler

Parses and writes FURUNO radar CSV files with the format:
Status,Scale,Range,Gain,Angle,EchoValues...

Where EchoValues is 868 comma-separated intensity values (0-255).
"""

import csv
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

NUM_RANGE_BINS: int = 868
ANGLE_TICKS_PER_ROTATION: float = 8192.0


def ticks_to_radians(ticks: int) -> float:
    return (ticks / ANGLE_TICKS_PER_ROTATION) * 2.0 * math.pi


def radians_to_ticks(radians: float) -> int:
    normalized = radians % (2.0 * math.pi)
    return int((normalized / (2.0 * math.pi)) * ANGLE_TICKS_PER_ROTATION)


def angle_diff(a: float, b: float) -> float:
    diff = abs(a - b)
    return min(diff, 2.0 * math.pi - diff)


@dataclass
class Pulse:
    status: int = 0
    scale: int = 0
    range_: int = 0
    gain: int = 0
    angle_ticks: int = 0
    angle_rad: float = 0.0
    echoes: np.ndarray = field(default_factory=lambda: np.zeros(NUM_RANGE_BINS, dtype=np.uint8))


@dataclass
class RadarFrame:
    source_path: str = ""
    timestamp: str = ""
    pulses: List[Pulse] = field(default_factory=list)
    unique_angles: List[float] = field(default_factory=list)

    def num_pulses(self) -> int:
        return len(self.pulses)

    def find_nearest_pulse(self, angle_rad: float) -> Optional[Pulse]:
        if not self.pulses:
            return None
        normalized = angle_rad % (2.0 * math.pi)
        return min(self.pulses, key=lambda p: angle_diff(p.angle_rad, normalized))

    def to_regularized_array(self, num_angles: int = 720) -> np.ndarray:
        result = np.zeros((num_angles, NUM_RANGE_BINS), dtype=np.uint8)
        angle_step = 2.0 * math.pi / num_angles
        for i in range(num_angles):
            target_angle = i * angle_step
            pulse = self.find_nearest_pulse(target_angle)
            if pulse is not None:
                result[i, :] = pulse.echoes
        return result

    def regularize(self, num_pulses: int = 720, max_gap_degrees: float = 1.0) -> "RadarFrame":
        new_frame = RadarFrame(source_path=self.source_path, timestamp=self.timestamp)
        angle_step = 2.0 * math.pi / num_pulses
        max_gap_rad = math.radians(max_gap_degrees)

        for i in range(num_pulses):
            target_angle = i * angle_step
            nearest = self.find_nearest_pulse(target_angle)
            if nearest is not None:
                gap = angle_diff(nearest.angle_rad, target_angle)
                if gap <= max_gap_rad:
                    new_frame.pulses.append(Pulse(
                        status=nearest.status,
                        scale=nearest.scale,
                        range_=nearest.range_,
                        gain=nearest.gain,
                        angle_ticks=radians_to_ticks(target_angle),
                        angle_rad=target_angle,
                        echoes=nearest.echoes.copy(),
                    ))
                else:
                    new_frame.pulses.append(Pulse(
                        status=0,
                        scale=nearest.scale,
                        range_=nearest.range_,
                        gain=nearest.gain,
                        angle_ticks=radians_to_ticks(target_angle),
                        angle_rad=target_angle,
                        echoes=np.zeros(NUM_RANGE_BINS, dtype=np.uint8),
                    ))

        new_frame.update_unique_angles()
        return new_frame

    def update_unique_angles(self):
        angles = sorted(set(p.angle_rad for p in self.pulses))
        self.unique_angles = angles


def _extract_timestamp(path: str) -> str:
    return Path(path).stem


def _parse_csv_file(path: str) -> RadarFrame:
    frame = RadarFrame(source_path=path, timestamp=_extract_timestamp(path))

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header

        for row in reader:
            if len(row) < 6:
                continue

            try:
                status = int(row[0])
            except ValueError:
                status = 0
            try:
                scale = int(row[1])
            except ValueError:
                scale = 0
            try:
                range_ = int(row[2])
            except ValueError:
                range_ = 0
            try:
                gain = int(row[3])
            except ValueError:
                gain = 0
            try:
                angle_ticks = int(row[4])
            except ValueError:
                angle_ticks = 0

            echoes = np.zeros(NUM_RANGE_BINS, dtype=np.uint8)
            n = min(len(row) - 5, NUM_RANGE_BINS)
            for j in range(n):
                try:
                    echoes[j] = int(row[5 + j])
                except ValueError:
                    echoes[j] = 0

            pulse = Pulse(
                status=status,
                scale=scale,
                range_=range_,
                gain=gain,
                angle_ticks=angle_ticks,
                angle_rad=ticks_to_radians(angle_ticks),
                echoes=echoes,
            )
            frame.pulses.append(pulse)

    frame.pulses.sort(key=lambda p: p.angle_rad)
    frame.update_unique_angles()
    return frame


class RadarCSVHandler:
    def __init__(self):
        self.frame_cache: Dict[str, RadarFrame] = {}

    def read_csv(self, path: str) -> RadarFrame:
        path = str(path)
        if path in self.frame_cache:
            return self.frame_cache[path]
        frame = _parse_csv_file(path)
        self.frame_cache[path] = frame
        return frame

    @staticmethod
    def read_csv_uncached(path: str) -> RadarFrame:
        return _parse_csv_file(str(path))

    @staticmethod
    def write_csv(frame: RadarFrame, path: str) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["Status", "Scale", "Range", "Gain", "Angle", "EchoValues"]
            header.extend(str(i) for i in range(1, NUM_RANGE_BINS))
            writer.writerow(header)

            for pulse in frame.pulses:
                row = [
                    str(pulse.status),
                    str(pulse.scale),
                    str(pulse.range_),
                    str(pulse.gain),
                    str(pulse.angle_ticks),
                ]
                row.extend(str(int(v)) for v in pulse.echoes)
                writer.writerow(row)

    def read_directory(self, dir_path: str) -> List[RadarFrame]:
        files = self.list_csv_files(dir_path)
        return [self.read_csv(f) for f in files]

    @staticmethod
    def read_directory_parallel(dir_path: str) -> List[RadarFrame]:
        files = RadarCSVHandler.list_csv_files(dir_path)
        with ThreadPoolExecutor() as pool:
            frames = list(pool.map(RadarCSVHandler.read_csv_uncached, files))
        return frames

    @staticmethod
    def list_csv_files(dir_path: str) -> List[str]:
        dir_path = str(dir_path)
        files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if f.lower().endswith(".csv")
        ]
        files.sort()
        return files

    def clear_cache(self):
        self.frame_cache.clear()
