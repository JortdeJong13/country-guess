import geopandas as gpd
import numpy as np
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

    def __init__(self, path="./data/reference.geojson", shape=(64, 64)):
        gdf = gpd.read_file(path)
        gdf["normal_geom"] = gdf["geometry"].apply(normalize_geom, shape=shape)
        self.gdf = gdf
        self.shape = shape
        self.country_name = gdf["cntry_name"].to_list()
        self._idx = 0

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
        while idx < 0:
            idx += len(self)

        geom = self.gdf.loc[idx, "normal_geom"]

        return geom

    def from_country_name(self, country_name):
        idx = self.gdf.index[self.gdf["cntry_name"] == country_name]

        return self[idx.item()]


class TestDataset(Dataset):
    """For evaluating on user drawn countries"""

    def __init__(self, path="./data/drawings.geojson", shape=(64, 64)):
        Dataset.__init__(self, path=path, shape=shape)

    def __getitem__(self, idx):
        geom = super().__getitem__(idx)
        drawing = lines_to_img(geom, self.shape)

        return {"country_name": self.country_name[idx], "drawing": drawing}
