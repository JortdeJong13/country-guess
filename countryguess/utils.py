import json
import logging
from datetime import datetime
from pathlib import Path

from shapely import LineString, MultiLineString, MultiPolygon, Polygon, to_geojson
from shapely.affinity import affine_transform, scale, translate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def normalize_geom(geom, shape=(64, 64), pad=2, eps=0.0001):
    """Scale and move the Shapely geom to within a box"""
    # Scale polygon
    minx, miny, maxx, maxy = geom.bounds
    scale_factor = min(
        (shape[0] - 2 * pad) / max((maxx - minx), eps),
        (shape[1] - 2 * pad) / max((maxy - miny), eps),
    )
    geom = scale(geom, xfact=scale_factor, yfact=scale_factor)

    # Translate polygon
    minx, miny, maxx, maxy = geom.bounds
    x_off = (shape[0] - minx - maxx) / 2
    y_off = (shape[1] - miny - maxy) / 2
    geom = translate(geom, xoff=x_off, yoff=y_off)

    return geom


def proces_lines(lines):
    lines = [LineString(line) for line in lines]
    lines = MultiLineString(lines)
    lines = affine_transform(lines, [1, 0, 0, -1, 0, 0])
    lines = normalize_geom(lines)

    return to_geojson(lines)


def decompose(polygon):
    if isinstance(polygon, Polygon):
        return [polygon]

    if isinstance(polygon, MultiPolygon):
        return list(polygon.geoms)


def save_drawing(country_name, drawing, output_dir="./data/drawings/"):
    output_dir = Path(output_dir)
    logger.info("Saving drawing of %s to %s", country_name, output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()

    # Create filename for new drawing
    filename = f"{country_name.lower().replace(' ', '_')}_{timestamp}.geojson"

    # Create GeoJSON feature, including CRS information
    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": [
            {
                "type": "Feature",
                "properties": {"country_name": country_name, "timestamp": timestamp},
                "geometry": json.loads(drawing),
            }
        ],
    }

    # Save to file
    with open(output_dir / filename, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)


class DrawingStore:
    def __init__(self, max_drawings: int = 10):
        self.drawings = {}
        self.max_drawings = max_drawings

    def store(self, drawing_id: str, drawing: str):
        if len(self.drawings) >= self.max_drawings:
            first_key = next(iter(self.drawings))
            self.drawings.pop(first_key)

        self.drawings[drawing_id] = drawing

    def get(self, drawing_id: str):
        return self.drawings.get(drawing_id)

    def remove(self, drawing_id: str):
        self.drawings.pop(drawing_id, None)

    def contains(self, drawing_id: str):
        return drawing_id in self.drawings
