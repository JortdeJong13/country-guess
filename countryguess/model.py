import numpy as np
import mlx.core as mx
import mlx.nn as nn

from .utils import poly_to_img, lines_to_img


class Model(nn.Module):
    def __init__(self, dims=[1, 16, 32, 64, 80, 16*80, 96], shape=(64, 64)):
        super().__init__()
        self.shape = shape
        self._ref_countries = None
        self.conv_1 = self.conv_block(dims[0], dims[1])
        self.conv_2 = self.conv_block(dims[1], dims[2])
        self.conv_3 = self.conv_block(dims[2], dims[3])
        self.conv_4 = self.conv_block(dims[3], dims[4])
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(dims[5], dims[6])

    
    def conv_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            nn.BatchNorm(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.BatchNorm(out_dim),
            nn.ReLU(),
            nn.MaxPool2d(2))


    def __call__(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = x.flatten(start_axis=1)
        x = self.dropout(x)
        x = self.linear(x)
        
        return x

    
    def load_reference(self, ref_data):
        self._ref_countries = {}
        for idx in range(len(ref_data)):
            img = poly_to_img(ref_data[idx], ref_data.shape)
            img = mx.array(np.expand_dims(img, axis=0), dtype=mx.float32)
            embedding = self(img)
            self._ref_countries[ref_data.country_name[idx]] = embedding

    
    def rank_countries(self, drawings):
        embedding = self(drawings)
        countries = []
        distances = []

        if not self._ref_countries:
            raise Exception("First the reference dataset needs to be loaded!")            

        for country, ref_emb in self._ref_countries.items():
            countries.append(country)
            distance = mx.linalg.norm(embedding - ref_emb, axis=-1)
            distances.append(distance)

        return countries, np.array(distances)


def predict(model, drawing):
    drawing = np.expand_dims(lines_to_img(drawing, model.shape), 0)
    drawing = mx.array(drawing, dtype=mx.float32)
    countries, distances = model.rank_countries(drawing)
    ranking = zip(countries, np.squeeze(distances))
    ranking = sorted(ranking, key=lambda t: t[1])
    countries = [country for country, rank in ranking]

    return countries