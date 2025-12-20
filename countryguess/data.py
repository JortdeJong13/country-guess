"""Load and preprocess country geometries."""

import logging
import random
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
        for p1, p2 in zip(points[:-1], points[1:]):
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


class Dataset:
    """Base dataset for fetching country geometries"""

    # Class variable for sharing reference data
    _ref_gdf = None

    def __init__(self, shape=(64, 64)):
        self.shape = shape
        self.geom_col = f"geom_{shape[0]}_{shape[1]}"
        self._idx = 0

        # Load reference data and normalize geometries
        self.ref_gdf = self.get_ref_gdf()
        self.ref_gdf = self.add_normal_geom(self.ref_gdf)

        # Set working dataset
        self.gdf = self.set_working_gdf()

    def set_working_gdf(self) -> gpd.GeoDataFrame:
        """Set the reference countries as the working GeoDataFrame"""
        return self.ref_gdf

    @classmethod
    def get_ref_gdf(cls) -> gpd.GeoDataFrame:
        """Get and cache reference data"""
        if Dataset._ref_gdf is None:
            # Load reference data
            Dataset._ref_gdf = cls.load_gdf("./data/reference/")
        return Dataset._ref_gdf

    @staticmethod
    def load_gdf(path):
        """Load GeoDataFrame from path"""
        files = list(Path(path).glob("*.geojson"))
        gdfs = [gpd.read_file(file) for file in files]
        gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        logger.info("Loaded %d samples from %s", len(gdf), path)

        return gdf

    def add_normal_geom(self, gdf):
        """Add normalized geometry column to GeoDataFrame"""
        if self.geom_col not in gdf:
            gdf[self.geom_col] = gdf["geometry"].apply(normalize_geom, shape=self.shape)
        return gdf

    def from_country_name(self, country_name):
        """Get the reference image for a country"""
        idx = self.ref_gdf.index[self.ref_gdf["country_name"] == country_name]
        if idx.empty:
            logger.warning("Country %s not found", country_name)
            return np.zeros(self.shape, dtype=np.uint8)

        geom = self.ref_gdf.loc[idx.item(), self.geom_col]
        ref_img = geom_to_img(geom, self.shape)

        return ref_img

    def __len__(self):
        return len(self.gdf)

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

        geom = self.gdf.loc[idx, self.geom_col]
        country_name = self.gdf.loc[idx, "country_name"]

        return {"country_name": country_name, "geometry": geom}


class TestDataset(Dataset):
    """For accessing user drawn countries"""

    def __init__(self, shape=(64, 64)):
        Dataset.__init__(self, shape=shape)
        # Drop countries without a reference
        reference_countries = set(self.ref_gdf["country_name"])
        self.gdf = self.gdf[self.gdf["country_name"].isin(reference_countries)].copy()  # type: ignore
        self.gdf = self.gdf[self.gdf["validated"]].copy()

        # Normalize test data
        self.gdf = self.add_normal_geom(self.gdf)

        # Sort test data by timestamp
        self.gdf = self.gdf.sort_values(by="timestamp")  # type: ignore
        self.gdf = self.gdf.reset_index(drop=True)

    def set_working_gdf(self):
        """Set user drawings as the working GeoDataFrame"""
        return self.load_gdf("./data/drawings/")

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        country_name, geom = item["country_name"], item["geometry"]
        drawing = geom_to_img(geom, self.shape)

        return {"country_name": country_name, "drawing": drawing}
