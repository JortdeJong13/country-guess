import json
from datetime import datetime
from pathlib import Path

from shapely import LineString, MultiLineString, MultiPolygon, Polygon, to_geojson
from shapely.affinity import affine_transform, scale, translate


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
        return [poly for poly in polygon.geoms]


def save_drawing(country_name, drawing, output_dir="./data/drawings/"):
    # Create directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
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
    with open(Path(output_dir) / filename, "w") as f:
        json.dump(geojson, f, indent=2)
