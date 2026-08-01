"""Load and preprocess reference country geometries."""

import logging
import random
from itertools import pairwise
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import LineString, Polygon
from skimage import draw

from .utils import normalize_geom

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def geom_to_img(geom, shape, img=None):
    """Convert any geometry to binary image."""
    if img is None:
        # Initialize an empty image
        img = np.zeros(shape, dtype=np.uint8)

    if hasattr(geom, "geoms"):
        # Add subgeometries to the image
        for subgeom in geom.geoms:
            img = geom_to_img(subgeom, shape, img)
        return img

    elif isinstance(geom, LineString):
        points = np.array(geom.coords).astype(int)

        # Draw lines between consecutive points
        for p1, p2 in pairwise(points):
            rr, cc = draw.line(p1[1], p1[0], p2[1], p2[0])
            img[rr, cc] = 1
        return img

    elif isinstance(geom, Polygon):
        points = np.array(geom.exterior.coords)
        rr, cc = draw.polygon_perimeter(points[:, 1], points[:, 0], shape=img.shape)
        img[rr, cc] = 1

        # Draw interior borders
        for interior in geom.interiors:
            interior_coords = np.array(interior.coords).astype(int)
            rr, cc = draw.polygon_perimeter(
                interior_coords[:, 1], interior_coords[:, 0], shape=img.shape
            )
            img[rr, cc] = 1
        return img

    else:
        raise ValueError(f"Unsupported geometry type: {type(geom)}")


class ReferenceDataset:
    """Dataset of local reference country geometries."""

    # Class variable for sharing reference data
    _ref_gdf = None

    def __init__(self, shape=(64, 64)):
        self.shape = shape
        self._idx = 0

        # Keep the GeoDataFrame for the reference-data notebook and normalize
        # a small in-memory sample list for model access.
        self.ref_gdf = self.get_ref_gdf()
        self.reference_samples = self._normalize_samples(self.ref_gdf)
        self.samples = self.reference_samples

    @classmethod
    def get_ref_gdf(cls) -> gpd.GeoDataFrame:
        """Get and cache reference data"""
        if ReferenceDataset._ref_gdf is None:
            # Load reference data
            ReferenceDataset._ref_gdf = cls.load_gdf("./data/reference/")
        return ReferenceDataset._ref_gdf

    @staticmethod
    def load_gdf(path):
        """Load GeoDataFrame from path"""
        files = list(Path(path).glob("*.geojson"))
        gdfs = [gpd.read_file(file) for file in files]
        gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        logger.info("Loaded %d samples from %s", len(gdf), path)

        return gdf

    def _normalize_samples(self, gdf):
        """Convert a GeoDataFrame into the dataset's normalized sample shape."""
        return [
            {
                "country_name": row["country_name"],
                "geometry": normalize_geom(row.geometry, shape=self.shape),
            }
            for _, row in gdf.iterrows()
        ]

    def from_country_name(self, country_name):
        """Get the reference image for a country"""
        for sample in self.reference_samples:
            if sample["country_name"] == country_name:
                return geom_to_img(sample["geometry"], self.shape)

        logger.warning("Country %s not found", country_name)
        return np.zeros(self.shape, dtype=np.uint8)

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx < len(self):
            self._idx += 1
            return self[self._idx - 1]
        self._idx = 0
        raise StopIteration

    def __getitem__(self, idx):
        if idx is None:
            # Get a random sample
            idx = random.randrange(len(self))

        # Handle negative indices
        idx %= len(self)

        return self.samples[idx]
