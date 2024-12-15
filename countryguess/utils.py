import json
from datetime import datetime

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

    return lines


def decompose(polygon):
    if isinstance(polygon, Polygon):
        return [polygon]

    if isinstance(polygon, MultiPolygon):
        return [poly for poly in polygon.geoms]


def save_drawing(country_name, drawing, path="./data/drawings.geojson"):
    feature = {
        "type": "Feature",
        "properties": {
            "cntry_name": country_name,
            "timestamp": datetime.now().isoformat(),
        },
        "geometry": json.loads(to_geojson(drawing)),
    }

    # Open the GeoJSON file for appending
    with open(path, "r+") as f:
        existing_data = json.load(f)
        existing_data["features"].append(feature)
        f.seek(0)
        json.dump(existing_data, f)
        f.truncate()
