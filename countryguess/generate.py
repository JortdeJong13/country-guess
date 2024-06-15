import random
from shapely import MultiPolygon, simplify
from imgaug.augmentables import polys
import imgaug.augmenters as iaa
from shapelysmooth import chaikin_smooth

from .utils import decompose, normalize_geom
from .data import poly_to_img, Dataset


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


class ValDataset(Dataset):
    """Extends the base dataset for evaluating on generated drawings
    """
    def __init__(self, temp=1.0, path='./data/reference.geojson', shape=(64, 64)):
        Dataset.__init__(self, path=path, shape=shape)
        self.temp = temp

    
    def __getitem__(self, idx):
        geom = super().__getitem__(idx)
        drawing = generate_drawing(geom, self.shape, self.temp)
        
        return {"country_name": self.country_name[idx], 
                "drawing": drawing}


class TripletDataset(Dataset):
    """Extends the base dataset for fetching triplet samples
    """
    def __init__(self, temp=1.0, path='./data/reference.geojson', shape=(64, 64)):
        Dataset.__init__(self, path=path, shape=shape)
        self.temp = temp


    def __getitem__(self, idx):
        pos_poly = super().__getitem__(idx)
        neg_idx = self.random_neg(idx)
        neg_poly = super().__getitem__(neg_idx)

        drawing = generate_drawing(pos_poly, self.shape, self.temp)
        pos_img = poly_to_img(pos_poly, self.shape)
        neg_img = poly_to_img(neg_poly, self.shape)
        
        return {"drawing": drawing,
                "pos_img": pos_img,
                "pos_idx": idx,
                "neg_img": neg_img,
                "neg_idx": neg_idx}


    def random_neg(self, ref_idx):
        idx = random.randint(0, len(self.gdf) - 1)
        while idx == ref_idx:
            idx = random.randint(0, len(self.gdf) - 1)
            
        return idx