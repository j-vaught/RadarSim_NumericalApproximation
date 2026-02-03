"""Scene Configuration — YAML loader and dataclasses for scene profiles."""

import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class FlickerConfig:
    enabled: bool = False
    rate: float = 0.3
    primary_frames: List[int] = field(default_factory=lambda: [15, 50])
    flicker_frames: List[int] = field(default_factory=lambda: [2, 6])
    intensity_drop: float = 0.4


@dataclass
class IntensityConfig:
    base: float = 1.0
    variability: float = 0.1
    profile: str = "constant"
    period: int = 30
    amplitude: float = 0.3
    ramp_start: float = 0.2
    ramp_end: float = 1.5
    expression: str = ""


@dataclass
class PathConfig:
    type: str = "fixed"
    position: Any = "random"           # [p, b] or "random"
    start: Any = "edge"                # [p, b], "edge", or "random"
    end: Any = "edge"                  # [p, b], "edge", or "random"
    heading: float = 0.0               # degrees
    speed: Any = 1.0                   # number or expression string
    curvature: float = 0.4
    points: List[List[int]] = field(default_factory=list)
    smoothing: str = "cubic"           # cubic | linear | none
    loop: bool = False
    pulse_expr: str = ""               # equation type
    bin_expr: str = ""                 # equation type
    duration: List[float] = field(default_factory=lambda: [0.4, 0.9])
    allow_land: bool = False           # if True, snap to water; if False, truncate path


@dataclass
class PlacementConfig:
    margin: int = 50
    prefer_edge: bool = False


@dataclass
class ObjectGroupConfig:
    name: str = ""
    count: int = 1
    size: Any = "medium"               # "small"|"medium"|"large"|[min, max]
    intensity: IntensityConfig = field(default_factory=IntensityConfig)
    flicker: FlickerConfig = field(default_factory=FlickerConfig)
    path: PathConfig = field(default_factory=PathConfig)
    placement: PlacementConfig = field(default_factory=PlacementConfig)


@dataclass
class LabelConfig:
    export: bool = True
    class_map: Dict[str, int] = field(default_factory=dict)


@dataclass
class SceneConfig:
    seed: Optional[int] = None
    count: int = 200
    objects: List[ObjectGroupConfig] = field(default_factory=list)
    labels: LabelConfig = field(default_factory=LabelConfig)


def _parse_flicker(d: Optional[Dict]) -> FlickerConfig:
    if d is None:
        return FlickerConfig()
    return FlickerConfig(
        enabled=d.get("enabled", False),
        rate=float(d.get("rate", 0.3)),
        primary_frames=list(d.get("primary_frames", [15, 50])),
        flicker_frames=list(d.get("flicker_frames", [2, 6])),
        intensity_drop=float(d.get("intensity_drop", 0.4)),
    )


def _parse_intensity(d: Optional[Dict]) -> IntensityConfig:
    if d is None:
        return IntensityConfig()
    return IntensityConfig(
        base=float(d.get("base", 1.0)),
        variability=float(d.get("variability", 0.1)),
        profile=str(d.get("profile", "constant")),
        period=int(d.get("period", 30)),
        amplitude=float(d.get("amplitude", 0.3)),
        ramp_start=float(d.get("ramp_start", 0.2)),
        ramp_end=float(d.get("ramp_end", 1.5)),
        expression=str(d.get("expression", "")),
    )


def _parse_path(d: Optional[Dict]) -> PathConfig:
    if d is None:
        return PathConfig()
    cfg = PathConfig(
        type=str(d.get("type", "fixed")),
        position=d.get("position", "random"),
        start=d.get("start", "edge"),
        end=d.get("end", "edge"),
        heading=float(d.get("heading", 0.0)),
        speed=d.get("speed", 1.0),
        curvature=float(d.get("curvature", 0.4)),
        points=[list(p) for p in d.get("points", [])],
        smoothing=str(d.get("smoothing", "cubic")),
        loop=bool(d.get("loop", False)),
        pulse_expr=str(d.get("pulse", "")),
        bin_expr=str(d.get("bin", "")),
        duration=list(d.get("duration", [0.4, 0.9])),
        allow_land=bool(d.get("allow_land", False)),
    )
    return cfg


def _parse_placement(d: Optional[Dict]) -> PlacementConfig:
    if d is None:
        return PlacementConfig()
    return PlacementConfig(
        margin=int(d.get("margin", 50)),
        prefer_edge=bool(d.get("prefer_edge", False)),
    )


def _parse_object_group(d: Dict) -> ObjectGroupConfig:
    return ObjectGroupConfig(
        name=str(d.get("name", "")),
        count=int(d.get("count", 1)),
        size=d.get("size", "medium"),
        intensity=_parse_intensity(d.get("intensity")),
        flicker=_parse_flicker(d.get("flicker")),
        path=_parse_path(d.get("path")),
        placement=_parse_placement(d.get("placement")),
    )


def _parse_labels(d: Optional[Dict]) -> LabelConfig:
    if d is None:
        return LabelConfig()
    return LabelConfig(
        export=bool(d.get("export", True)),
        class_map=dict(d.get("class_map", {})),
    )


def load_scene(path: str) -> SceneConfig:
    """Load a scene profile from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}

    objects = [_parse_object_group(o) for o in raw.get("objects", [])]

    return SceneConfig(
        seed=raw.get("seed"),
        count=int(raw.get("count", 200)),
        objects=objects,
        labels=_parse_labels(raw.get("labels")),
    )
