import random
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from skimage import draw

from .utils import normalize_geom


def lines_to_img(lines, shape):
    img = np.zeros(shape, dtype=np.uint8)

    for line in lines.geoms:
        points = np.array(line.coords).astype(int)
        for r0, c0, r1, c1 in zip(
            points[:-1, 0], points[:-1, 1], points[1:, 0], points[1:, 1]
        ):
            rr, cc = draw.line(r0, c0, r1, c1)
            img[rr, cc] = 1

    return np.swapaxes(img, 0, 1)


def poly_to_img(polygon, shape):
    img = np.zeros(shape, dtype=np.uint8)

    for poly in polygon.geoms:
        points = np.array(poly.exterior.coords)
        rr, cc = draw.polygon_perimeter(points[:, 0], points[:, 1], shape=img.shape)
        img[rr, cc] = 1

    return np.swapaxes(img, 0, 1)


class Dataset:
    """Base dataset for fetching reference country geometry"""

    def __init__(self, path="./data/reference/", shape=(64, 64)):
        self.path = Path(path)
        self.shape = shape
        self._idx = 0

        # Load all geojson files form directory
        self.files = list(self.path.glob("*.geojson"))

        # Create a GeoDataFrame
        gdfs = [gpd.read_file(file) for file in self.files]
        self.gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

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
        else:
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
        drawing = lines_to_img(geom, self.shape)

        return {"country_name": country_name, "drawing": drawing}
