import geopandas as gpd
import random

from .utils import normalize_geom, lines_to_img, poly_to_img
from .generate import generate_drawing


class Dataset():
    """Base dataset for fetching reference country geometry
    """
    def __init__(self, path='./data/reference.geojson', shape=(64, 64)):
        gdf = gpd.read_file(path)
        gdf['normal_geom'] = gdf['geometry'].apply(normalize_geom, shape=shape)
        self.gdf = gdf
        self.shape = shape
        self.country_name = gdf['cntry_name'].to_list()
        self._idx = 0

     
    def __len__(self):
        return len(self.gdf)


    def __iter__(self):
        return self

    
    def __next__(self):
        if self._idx < len(self):
            self._idx += 1
            return self[self._idx-1]
        else:
            self._idx=0
            raise StopIteration
        
    
    def __getitem__(self, idx):
        while idx < 0:
            idx += len(self)

        geom = self.gdf.loc[idx, 'normal_geom']
        
        return geom


    def from_country_name(self, country_name):
        idx = self.gdf.index[self.gdf['cntry_name'] == country_name]
        
        return self[idx.item()]



class TestDataset(Dataset):
    """For evaluating on user drawn countries
    """
    def __init__(self, path='./data/drawings.geojson', shape=(64, 64)):
        Dataset.__init__(self, path=path, shape=shape)
        
    
    def __getitem__(self, idx):
        geom = super().__getitem__(idx)
        drawing = lines_to_img(geom, self.shape)
        
        return {"country_name": self.country_name[idx], 
                "drawing": drawing}


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
