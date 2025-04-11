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
    """Base dataset for fetching country geometries"""

    def __init__(self, path="./data/reference/", shape=(64, 64)):
        self.path = Path(path)
        self.shape = shape
        self._idx = 0

        # Load all geojson files form directory
        self.files = list(self.path.glob("*.geojson"))

        # Create a GeoDataFrame
        gdfs = [gpd.read_file(file) for file in self.files]
        self.gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        logger.info("Loaded %d countries", len(self.gdf))

        # Normalize geometries
        self.gdf["geometry"] = self.gdf["geometry"].apply(normalize_geom, shape=shape)

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

        geom = self.gdf.loc[idx, "geometry"]
        country_name = self.gdf.loc[idx, "country_name"]

        return {"country_name": country_name, "geometry": geom}

    def from_country_name(self, country_name):
        idx = self.gdf.index[self.gdf["country_name"] == country_name]

        return self[idx.item()]["geometry"]


class TestDataset(Dataset):
    """For evaluating on user drawn countries"""

    def __init__(self, path="./data/drawings/", shape=(64, 64)):
        Dataset.__init__(self, path=path, shape=shape)
        # Sort test data by timestamp
        self.gdf.sort_values(by="timestamp", inplace=True)
        self.gdf.reset_index(drop=True, inplace=True)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        country_name, geom = item["country_name"], item["geometry"]
        drawing = geom_to_img(geom, self.shape)

        return {"country_name": country_name, "drawing": drawing}
