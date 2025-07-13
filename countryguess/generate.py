"""Country drawing generation for model training."""

import random

import imgaug.augmenters as iaa
from imgaug.augmentables import polys
from shapely import MultiPolygon, simplify
from shapelysmooth import chaikin_smooth

from .data import Dataset, geom_to_img
from .utils import decompose, normalize_geom


def rm_island(polygons, area):
    """Randomly removes the smaller polygons"""
    max_area = max(poly.area for poly in polygons)
    polygons = [poly for poly in polygons if poly.area / max_area > area]

    return polygons


def augment_polygon(polygon, temp):
    """Augment polygon with linear transformations."""
    # Augment Polygon
    aug = iaa.Sequential(
        [
            iaa.ScaleX((1 / (1 + 0.3 * temp), 1 + 0.3 * temp)),
            iaa.ShearX((-8 * temp, 8 * temp)),
            iaa.PerspectiveTransform(scale=(0, 0.12 * temp)),
            iaa.WithPolarWarping(
                iaa.Affine(
                    translate_percent={
                        "x": (-0.05 * temp, 0.05 * temp),
                        "y": (-0.05 * temp, 0.05 * temp),
                    }
                )
            ),
            iaa.WithPolarWarping(
                iaa.ShearX((-2 * temp, -2 * temp)),
            ),
            iaa.Rotate((-5 * temp, 5 * temp)),
        ]
    )
    return aug(polygons=polygon)


def generate_drawing(polygon, shape, temp=1.0):
    """Generate a drawing from a reference country polygon."""
    # Simplify Polygon
    polygon = simplify(polygon, tolerance=random.uniform(0, 4 * temp))

    # Decompose MultiPolygon
    polygon = decompose(polygon)

    # Forget island
    polygon = rm_island(polygon, area=random.triangular(0, 0.12 * temp, 0))

    # Smooth Polygon
    polygon = [
        chaikin_smooth(poly, iters=int(random.uniform(1, 4.5 * temp)))
        for poly in polygon
    ]

    # Move Polygon to Imgaug Polygon
    polygon = polys.PolygonsOnImage(
        [polys.Polygon.from_shapely(poly) for poly in polygon], shape
    )

    # Apply augmentations
    polygon = augment_polygon(polygon, temp)

    # Transform Polygon into Shapley MultiPolygon
    polygon = MultiPolygon([poly.to_shapely_polygon() for poly in polygon])  # type: ignore

    # Normalize Augmented Polygon
    polygon = normalize_geom(polygon, shape)

    # Transform into img
    img = geom_to_img(polygon, shape)

    return img


class ValDataset(Dataset):
    """Extends the base dataset for evaluating on generated drawings"""

    def __init__(self, shape=(64, 64), temp=1.0):
        Dataset.__init__(self, shape=shape)
        self.temp = temp

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        country_name, geom = item["country_name"], item["geometry"]
        drawing = generate_drawing(geom, self.shape, self.temp)

        return {"country_name": country_name, "drawing": drawing}


class TripletDataset(Dataset):
    """Extends the base dataset for fetching triplet samples"""

    def __init__(self, shape=(64, 64), temp=1.0):
        Dataset.__init__(self, shape=shape)
        self.temp = temp

    def __getitem__(self, idx):
        pos_poly = super().__getitem__(idx)["geometry"]
        neg_idx = self.get_random_neg(idx)
        neg_poly = super().__getitem__(neg_idx)["geometry"]

        drawing = generate_drawing(pos_poly, self.shape, self.temp)
        pos_img = geom_to_img(pos_poly, self.shape)
        neg_img = geom_to_img(neg_poly, self.shape)

        return {
            "drawing": drawing,
            "pos_img": pos_img,
            "pos_idx": idx,
            "neg_img": neg_img,
            "neg_idx": neg_idx,
        }

    def get_random_neg(self, ref_idx):
        """Get a random other index."""
        idx = random.randint(0, len(self.gdf) - 1)
        while idx == ref_idx:
            idx = random.randint(0, len(self.gdf) - 1)

        return idx
