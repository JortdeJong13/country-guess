import numpy as np
import random
import json
from datetime import datetime
from shapely.affinity import scale, translate, affine_transform
from shapely import Polygon, MultiPolygon, simplify, LineString, MultiLineString
from shapely import to_geojson
from skimage import draw
from imgaug.augmentables import polys
import imgaug.augmenters as iaa
from shapelysmooth import chaikin_smooth


def normalize_geom(geom, shape=(64, 64), pad=2, eps=0.0001):
    """Scale and move the Shapely geom to within a box"""
    #Scale polygon
    minx, miny, maxx, maxy = geom.bounds
    scale_factor = min((shape[0] - 2*pad) / max((maxx - minx), eps),
                       (shape[1] - 2*pad) / max((maxy - miny), eps))
    geom = scale(geom, xfact=scale_factor, yfact=scale_factor)
    
    #Translate polygon
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


def lines_to_img(lines, shape):
    img = np.zeros(shape, dtype=np.uint8)
    
    for line in lines.geoms:
        points = np.array(line.coords).astype(int)
        for r0, c0, r1, c1 in zip(points[:-1, 0], points[:-1, 1], points[1:, 0], points[1:, 1]):
            rr, cc = draw.line(r0, c0, r1, c1)
            img[rr, cc] = 1
    
    return np.swapaxes(img, 0, 1)


def poly_to_img(polygon, shape):
    if isinstance(polygon, Polygon):
        polygon = MultiPolygon([polygon])

    img = np.zeros(shape, dtype=np.uint8)

    for poly in polygon.geoms:
        points = np.array(poly.exterior.coords)
        rr, cc = draw.polygon_perimeter(points[:, 0], points[:, 1], shape=img.shape)
        img[rr, cc] = 1

    return np.swapaxes(img, 0, 1)


def rm_island(polygons, area):
    "Randomly removes the smaller polygons"
    max_area = max([poly.area for poly in polygons])
    polygons = [poly for poly in polygons if poly.area/max_area > area]
    
    return polygons


def generate_drawing(polygon, shape, temp=1.0):
    #Simplify Polygon
    polygon = simplify(polygon, tolerance=random.uniform(0, 3*temp))

    #Decompose MultiPolygon
    polygon = decompose(polygon)

    #Forget island
    polygon = rm_island(polygon, area=random.triangular(0, 0.08*temp, 0))
    
    #Smooth Polygon
    polygon = [chaikin_smooth(poly, 
                             iters=int(random.uniform(1, 4.5*temp))) 
               for poly in polygon]

    #Move Polygon to Imgaug Polygon
    polygon = polys.PolygonsOnImage(
        [polys.Polygon.from_shapely(poly) for poly in polygon],
        shape)
    
    #Augment Polygon
    aug = iaa.Sequential([
        iaa.ScaleX((1/(1+0.6*temp), 1+0.6*temp)),
        iaa.ShearX((-19*temp, 19*temp)),
        iaa.PerspectiveTransform(scale=(0, 0.13*temp)),
        iaa.WithPolarWarping(
            iaa.Affine(translate_percent={
                "x": (-0.06*temp, 0.06*temp), 
                "y": (-0.06*temp, 0.06*temp)})
            ),
        ])
    polygon = aug(polygons=polygon)

    #Transform Polygon into Shapley MultiPolygon
    polygon = MultiPolygon([poly.to_shapely_polygon() for poly in polygon])

    #Normalize Augmented Polygon
    polygon = normalize_geom(polygon, shape)

    #Transform into img
    img = poly_to_img(polygon, shape)

    return img


#tmp
import torch
def predict(model, drawing):
    drawing = lines_to_img(drawing, model.shape)[None, None, :, :]
    drawing = torch.tensor(drawing, dtype=torch.float32).to(torch.device("mps"))
    countries, distances = model.rank_countries(drawing)
    ranking = zip(countries, np.squeeze(distances))
    ranking = sorted(ranking, key=lambda t: t[1])
    countries = [country for country, rank in ranking]

    return countries


def save_drawing(country_name, drawing, path='./data/drawings.geojson'):
    feature = {
        "type": "Feature",
        "properties": {
            "cntry_name": country_name,
            "timestamp": datetime.now().isoformat()
        },
        "geometry": json.loads(to_geojson(drawing))
    }
    
    # Open the GeoJSON file for appending
    with open(path, 'r+') as f:
        existing_data = json.load(f)
        existing_data['features'].append(feature)
        f.seek(0)
        json.dump(existing_data, f)
        f.truncate()