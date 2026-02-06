"""Scene Configuration — YAML loader and dataclasses for scene profiles."""

import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class DuctingConfig:
    enabled: bool = False
    height_m: float = 50.0
    strength: float = 0.3


@dataclass
class LandSceneConfig:
    """Land configuration within the scene environment block."""
    type: str = "parametric"          # "parametric" | "annotation" | "mask" | "frames"
    # Parametric options (existing Tier2 approach)
    coastline_range: float = 0.6
    land_start_az: float = 30.0
    land_end_az: float = 150.0
    roughness: float = 0.3
    intensity: float = 0.9
    bay_enabled: bool = True
    bay_center_az: float = 90.0
    bay_width_az: float = 30.0
    bay_depth: float = 0.15
    # Annotation option
    annotation_path: str = ""
    # Mask option
    mask_path: str = ""
    # Frames option (real radar CSV backgrounds, like Tier3 land_frames/)
    land_frames_dir: str = ""


@dataclass
class EnvironmentConfig:
    """Environment configuration — Tier2 uses, Tier3 ignores."""
    sea_state: int = 3
    wind_speed_mps: float = 12.0
    rain_rate_mmhr: float = 0.0
    ducting: DuctingConfig = field(default_factory=DuctingConfig)
    land: Optional[LandSceneConfig] = None


@dataclass
class PhysicsConfig:
    """Per-object physics — Tier2 uses, Tier3 ignores."""
    rcs_m2: Any = None                # float or [min, max]; None = derive from size
    swerling_case: int = 1            # 0-4
    target_class: str = "vessel"
    target_height_m: float = 5.0


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
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)


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
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)


def _parse_ducting(d: Optional[Dict]) -> DuctingConfig:
    if d is None:
        return DuctingConfig()
    return DuctingConfig(
        enabled=bool(d.get("enabled", False)),
        height_m=float(d.get("height_m", 50.0)),
        strength=float(d.get("strength", 0.3)),
    )


def _parse_land_scene(d: Optional[Dict]) -> Optional[LandSceneConfig]:
    if d is None:
        return None
    return LandSceneConfig(
        type=str(d.get("type", "parametric")),
        coastline_range=float(d.get("coastline_range", 0.6)),
        land_start_az=float(d.get("land_start_az", 30.0)),
        land_end_az=float(d.get("land_end_az", 150.0)),
        roughness=float(d.get("roughness", 0.3)),
        intensity=float(d.get("intensity", 0.9)),
        bay_enabled=bool(d.get("bay_enabled", True)),
        bay_center_az=float(d.get("bay_center_az", 90.0)),
        bay_width_az=float(d.get("bay_width_az", 30.0)),
        bay_depth=float(d.get("bay_depth", 0.15)),
        annotation_path=str(d.get("annotation_path", "")),
        mask_path=str(d.get("mask_path", "")),
        land_frames_dir=str(d.get("land_frames_dir", "")),
    )


def _parse_environment(d: Optional[Dict]) -> EnvironmentConfig:
    if d is None:
        return EnvironmentConfig()
    return EnvironmentConfig(
        sea_state=int(d.get("sea_state", 3)),
        wind_speed_mps=float(d.get("wind_speed_mps", 12.0)),
        rain_rate_mmhr=float(d.get("rain_rate_mmhr", 0.0)),
        ducting=_parse_ducting(d.get("ducting")),
        land=_parse_land_scene(d.get("land")),
    )


def _parse_physics(d: Optional[Dict]) -> PhysicsConfig:
    if d is None:
        return PhysicsConfig()
    return PhysicsConfig(
        rcs_m2=d.get("rcs_m2"),
        swerling_case=int(d.get("swerling_case", 1)),
        target_class=str(d.get("target_class", "vessel")),
        target_height_m=float(d.get("target_height_m", 5.0)),
    )


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
        physics=_parse_physics(d.get("physics")),
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
        environment=_parse_environment(raw.get("environment")),
    )
