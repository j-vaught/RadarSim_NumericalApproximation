"""Annotation Data Structures

JSON-serializable annotation format for water/land mask definitions.
"""

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION = "1.0.0"


@dataclass
class Point:
    x: float
    y: float

    def to_pixels(self, width: int, height: int) -> Tuple[int, int]:
        return (int(self.x * width), int(self.y * height))

    @staticmethod
    def from_pixels(px: int, py: int, width: int, height: int) -> "Point":
        return Point(x=px / width, y=py / height)

    def distance_to(self, other: "Point") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)


@dataclass
class Polygon:
    exterior: List[Point]
    holes: List[List[Point]] = field(default_factory=list)

    def contains(self, point: Point) -> bool:
        if not _point_in_polygon(point, self.exterior):
            return False
        for hole in self.holes:
            if _point_in_polygon(point, hole):
                return False
        return True

    def bounding_box(self) -> Tuple[Point, Point]:
        if not self.exterior:
            return (Point(0.0, 0.0), Point(0.0, 0.0))
        min_x = min(p.x for p in self.exterior)
        min_y = min(p.y for p in self.exterior)
        max_x = max(p.x for p in self.exterior)
        max_y = max(p.y for p in self.exterior)
        return (Point(min_x, min_y), Point(max_x, max_y))

    def area(self) -> float:
        return abs(_polygon_area(self.exterior))


def _point_in_polygon(point: Point, polygon: List[Point]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        pi = polygon[i]
        pj = polygon[j]
        if ((pi.y > point.y) != (pj.y > point.y)) and \
           (point.x < (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y) + pi.x):
            inside = not inside
        j = i
    return inside


def _polygon_area(polygon: List[Point]) -> float:
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i].x * polygon[j].y
        area -= polygon[j].x * polygon[i].y
    return area / 2.0


class RegionType(str, Enum):
    WATER = "water"
    LAND = "land"
    EXCLUDE = "exclude"


@dataclass
class Region:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: Optional[str] = None
    region_type: RegionType = RegionType.WATER
    polygon: Polygon = field(default_factory=lambda: Polygon([]))
    z_order: int = 0
    color: Optional[str] = None
    visible: bool = True
    locked: bool = False
    notes: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def contains(self, point: Point) -> bool:
        return self.polygon.contains(point)


@dataclass
class ImageDimensions:
    width: int
    height: int


@dataclass
class SourceInfo:
    path: str
    checksum: Optional[str] = None
    image_dimensions: Optional[ImageDimensions] = None


@dataclass
class Metadata:
    created_at: Optional[str] = None
    modified_at: Optional[str] = None
    author: Optional[str] = None
    notes: Optional[str] = None
    tool_version: Optional[str] = None


class CoordinateSystem(str, Enum):
    NORMALIZED = "normalized"
    PIXEL = "pixel"


@dataclass
class AnnotationSet:
    version: str = SCHEMA_VERSION
    source: SourceInfo = field(default_factory=lambda: SourceInfo(path=""))
    metadata: Metadata = field(default_factory=Metadata)
    coordinate_system: CoordinateSystem = CoordinateSystem.NORMALIZED
    regions: List[Region] = field(default_factory=list)

    def add_region(self, region: Region):
        self.regions.append(region)

    def remove_region(self, region_id: str) -> Optional[Region]:
        for i, r in enumerate(self.regions):
            if r.id == region_id:
                return self.regions.pop(i)
        return None

    def regions_by_type(self, region_type: RegionType) -> List[Region]:
        return [r for r in self.regions if r.region_type == region_type]

    def regions_sorted(self) -> List[Region]:
        return sorted(self.regions, key=lambda r: r.z_order)

    def find_region(self, region_id: str) -> Optional[Region]:
        for r in self.regions:
            if r.id == region_id:
                return r
        return None

    def find_region_at(self, point: Point) -> Optional[Region]:
        for r in reversed(self.regions_sorted()):
            if r.visible and r.contains(point):
                return r
        return None

    def water_regions(self) -> List[Region]:
        return self.regions_by_type(RegionType.WATER)

    def land_regions(self) -> List[Region]:
        return self.regions_by_type(RegionType.LAND)

    def is_water(self, point: Point) -> bool:
        in_water = any(r.contains(point) for r in self.water_regions())
        in_land = any(r.contains(point) for r in self.land_regions())
        if in_water and in_land:
            water_z = max((r.z_order for r in self.water_regions() if r.contains(point)), default=0)
            land_z = max((r.z_order for r in self.land_regions() if r.contains(point)), default=0)
            return water_z > land_z
        return in_water

    def is_land(self, point: Point) -> bool:
        in_water = any(r.contains(point) for r in self.water_regions())
        in_land = any(r.contains(point) for r in self.land_regions())
        if in_water and in_land:
            water_z = max((r.z_order for r in self.water_regions() if r.contains(point)), default=0)
            land_z = max((r.z_order for r in self.land_regions() if r.contains(point)), default=0)
            return land_z > water_z
        return in_land

    @staticmethod
    def load(path: str) -> "AnnotationSet":
        with open(path, "r") as f:
            data = json.load(f)
        return _annotation_set_from_dict(data)

    def save(self, path: str) -> None:
        self.metadata.modified_at = datetime.now(timezone.utc).isoformat()
        data = _annotation_set_to_dict(self)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def _point_to_dict(p: Point) -> dict:
    return {"x": p.x, "y": p.y}


def _point_from_dict(d: dict) -> Point:
    return Point(x=d["x"], y=d["y"])


def _polygon_to_dict(p: Polygon) -> dict:
    d = {"exterior": [_point_to_dict(pt) for pt in p.exterior]}
    if p.holes:
        d["holes"] = [[_point_to_dict(pt) for pt in hole] for hole in p.holes]
    return d


def _polygon_from_dict(d: dict) -> Polygon:
    exterior = [_point_from_dict(pt) for pt in d.get("exterior", [])]
    holes = [[_point_from_dict(pt) for pt in hole] for hole in d.get("holes", [])]
    return Polygon(exterior=exterior, holes=holes)


def _region_to_dict(r: Region) -> dict:
    d = {
        "id": r.id,
        "region_type": r.region_type.value,
        "polygon": _polygon_to_dict(r.polygon),
        "z_order": r.z_order,
        "visible": r.visible,
        "locked": r.locked,
    }
    if r.name is not None:
        d["name"] = r.name
    if r.color is not None:
        d["color"] = r.color
    if r.notes is not None:
        d["notes"] = r.notes
    if r.tags:
        d["tags"] = r.tags
    return d


def _region_from_dict(d: dict) -> Region:
    return Region(
        id=d.get("id", str(uuid.uuid4())),
        name=d.get("name"),
        region_type=RegionType(d.get("region_type", "water")),
        polygon=_polygon_from_dict(d.get("polygon", {})),
        z_order=d.get("z_order", 0),
        color=d.get("color"),
        visible=d.get("visible", True),
        locked=d.get("locked", False),
        notes=d.get("notes"),
        tags=d.get("tags", []),
    )


def _annotation_set_to_dict(a: AnnotationSet) -> dict:
    d = {
        "version": a.version,
        "source": {"path": a.source.path},
        "coordinate_system": a.coordinate_system.value,
        "regions": [_region_to_dict(r) for r in a.regions],
    }
    if a.source.checksum:
        d["source"]["checksum"] = a.source.checksum
    if a.source.image_dimensions:
        d["source"]["image_dimensions"] = {
            "width": a.source.image_dimensions.width,
            "height": a.source.image_dimensions.height,
        }
    meta = {}
    for k in ("created_at", "modified_at", "author", "notes", "tool_version"):
        v = getattr(a.metadata, k)
        if v is not None:
            meta[k] = v
    if meta:
        d["metadata"] = meta
    return d


def _annotation_set_from_dict(d: dict) -> AnnotationSet:
    src = d.get("source", {})
    source = SourceInfo(path=src.get("path", ""))
    source.checksum = src.get("checksum")
    if "image_dimensions" in src:
        source.image_dimensions = ImageDimensions(
            width=src["image_dimensions"]["width"],
            height=src["image_dimensions"]["height"],
        )

    meta_d = d.get("metadata", {})
    metadata = Metadata(
        created_at=meta_d.get("created_at"),
        modified_at=meta_d.get("modified_at"),
        author=meta_d.get("author"),
        notes=meta_d.get("notes"),
        tool_version=meta_d.get("tool_version"),
    )

    cs_val = d.get("coordinate_system", "normalized")
    try:
        cs = CoordinateSystem(cs_val)
    except ValueError:
        cs = CoordinateSystem.NORMALIZED

    regions = [_region_from_dict(r) for r in d.get("regions", [])]

    return AnnotationSet(
        version=d.get("version", SCHEMA_VERSION),
        source=source,
        metadata=metadata,
        coordinate_system=cs,
        regions=regions,
    )
