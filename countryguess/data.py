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


def geom_to_img(geometry, shape):
    """Convert any geometry to binary image."""
    img = np.zeros(shape, dtype=np.uint8)

    # Handle single geometries and multi-geometries
    if isinstance(geometry, (Polygon, LineString)):
        geoms = [geometry]
    else:
        geoms = geometry.geoms

    for geom in geoms:
        if isinstance(geom, LineString):
            points = np.array(geom.coords).astype(int)
            # Draw lines between consecutive points
            for p1, p2 in zip(points[:-1], points[1:]):
                rr, cc = draw.line(p1[1], p1[0], p2[1], p2[0])
                img[rr, cc] = 1

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


class Dataset:
    """Dataset for fetching country geometries"""

    # Class variable for sharing reference data
    _ref_gdf = None

    def __init__(self, shape=(64, 64)):
        self.shape = shape
        self.geom_col = "geom_{}_{}".format(*shape)
        self._idx = 0

        # Load reference data and normalize geometries
        self.ref_gdf = self.get_ref_gdf()
        self.ref_gdf = self.add_normal_geom(self.ref_gdf)

        # Set working dataset
        self.gdf = self.set_gdf()

    @classmethod
    def get_ref_gdf(cls):
        """Get and cache reference data"""
        if Dataset._ref_gdf is None:
            # Load reference data
            Dataset._ref_gdf = cls.load_gdf("./data/reference/")
        return Dataset._ref_gdf

    def set_gdf(self):
        """Set the reference countries as the main GeoDataFrame"""
        return self.ref_gdf

    @classmethod
    def load_gdf(cls, path):
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
            idx = random.randint(0, len(self) - 1)

        while idx < 0:
            idx += len(self)

        geom = self.gdf.loc[idx, self.geom_col]
        country_name = self.gdf.loc[idx, "country_name"]

        return {"country_name": country_name, "geometry": geom}

    def from_country_name(self, country_name):
        """Get the reference image for a country"""
        idx = self.ref_gdf.index[self.ref_gdf["country_name"] == country_name]
        geom = self.ref_gdf.loc[idx.item(), self.geom_col]
        ref_img = geom_to_img(geom, self.shape)

        return ref_img


class TestDataset(Dataset):
    """For accessing user drawn countries"""

    def __init__(self, shape=(64, 64)):
        Dataset.__init__(self, shape=shape)
        # Normalize test data
        self.gdf = self.add_normal_geom(self.gdf)

        # Sort test data by timestamp
        self.gdf.sort_values(by="timestamp", inplace=True)
        self.gdf.reset_index(drop=True, inplace=True)

    def set_gdf(self):
        """Override method to set user drawings as the test data"""
        return self.load_gdf("./data/drawings/")

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        country_name, geom = item["country_name"], item["geometry"]
        drawing = geom_to_img(geom, self.shape)

        return {"country_name": country_name, "drawing": drawing}
