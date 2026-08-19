"""
Zone definitions and geometry.

Zones are arbitrary polygons loaded from a JSON config. Each zone has a
`type` (intrusion | loitering) which controls what event logic is applied
to it in events.py. A zone can also be both, but for clarity we treat
"intrusion" as "any entry is newsworthy" and "loitering" as "dwell time
beyond a threshold is newsworthy" -- a zone can appear twice with the two
types if you want both behaviors on the same physical area.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from shapely.geometry import Point, Polygon

ZoneType = Literal["intrusion", "loitering"]


@dataclass
class Zone:
    name: str
    type: ZoneType
    polygon: Polygon
    loiter_threshold_seconds: float = 10.0
    color: tuple[int, int, int] = field(default_factory=lambda: (0, 0, 255))

    def contains(self, x: float, y: float) -> bool:
        """Point-in-polygon test. We use the bottom-center of a bbox
        (the person's approximate foot position) as the reference point,
        not the box centroid -- this is the standard convention for
        surveillance zone checks because it correlates with where a
        person is actually standing on the ground plane, not where
        their torso happens to be relative to the camera angle."""
        return self.polygon.contains(Point(x, y)) or self.polygon.touches(Point(x, y))

    @property
    def points(self) -> list[tuple[int, int]]:
        return [(int(x), int(y)) for x, y in self.polygon.exterior.coords]


def load_zones(config_path: str | Path) -> list[Zone]:
    with open(config_path, "r") as f:
        raw = json.load(f)

    zones = []
    for i, z in enumerate(raw.get("zones", [])):
        if len(z.get("polygon", [])) < 3:
            raise ValueError(f"Zone '{z.get('name', i)}' needs >= 3 points to form a polygon")

        poly = Polygon(z["polygon"])
        if not poly.is_valid:
            # Common cause: self-intersecting polygon from a typo'd point order.
            # buffer(0) is the standard shapely trick to fix minor self-intersections.
            poly = poly.buffer(0)

        default_color = (0, 0, 255) if z.get("type") == "intrusion" else (0, 165, 255)
        zones.append(
            Zone(
                name=z["name"],
                type=z.get("type", "intrusion"),
                polygon=poly,
                loiter_threshold_seconds=z.get("loiter_threshold_seconds", 10.0),
                color=tuple(z.get("color", default_color)),
            )
        )
    return zones
