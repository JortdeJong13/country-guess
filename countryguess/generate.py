import random
from shapely import MultiPolygon, simplify
from imgaug.augmentables import polys
import imgaug.augmenters as iaa
from shapelysmooth import chaikin_smooth

from .utils import decompose, normalize_geom, poly_to_img


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
