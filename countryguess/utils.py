"""Utility functions for processing and storing geographic drawings and shapes."""

import logging

from shapely import LineString, MultiLineString, MultiPolygon, Polygon, to_geojson
from shapely.affinity import affine_transform, scale, translate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def normalize_geom(geom, shape=(64, 64), pad=2):
    """Scales and moves the Shapely geom to fit within the shape."""
    # Scale polygon
    minx, miny, maxx, maxy = geom.bounds
    scale_factor = min(
        (shape[0] - 2 * pad) / max((maxx - minx), 0.0001),
        (shape[1] - 2 * pad) / max((maxy - miny), 0.0001),
    )
    geom = scale(geom, xfact=scale_factor, yfact=scale_factor)

    # Translate polygon
    minx, miny, maxx, maxy = geom.bounds
    x_off = (shape[0] - minx - maxx) / 2
    y_off = (shape[1] - miny - maxy) / 2
    geom = translate(geom, xoff=x_off, yoff=y_off)

    return geom


def proces_lines(lines):
    """Converts a list of lines into a normalized GeoJSON MultiLineString."""
    lines = [LineString(line) for line in lines]
    lines = MultiLineString(lines)
    lines = affine_transform(lines, [1, 0, 0, -1, 0, 0])
    lines = normalize_geom(lines)

    return to_geojson(lines)


def decompose(polygon):
    """Decomposes a Polygon or MultiPolygon into a list of polygons."""
    if isinstance(polygon, Polygon):
        return [polygon]

    if isinstance(polygon, MultiPolygon):
        return list(polygon.geoms)

    raise ValueError(f"Expected Polygon or MultiPolygon, received {type(polygon)}")
