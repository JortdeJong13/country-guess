import numpy as np
import torch
from mlflow import MlflowClient
from mlflow.pytorch import load_model
from torch import nn

from .data import Dataset, poly_to_img


class TripletModel(nn.Module):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model
        self._ref_countries = None

    @property
    def shape(self):
        return self.embedding_model.shape

    def forward(self, x):
        return self.embedding_model(x)

    @torch.no_grad
    def load_reference(self, ref_data):
        assert ref_data.shape == self.shape
        self._ref_countries = {}
        for idx in range(len(ref_data)):
            img = poly_to_img(ref_data[idx], ref_data.shape)
            embedding = self(
                torch.tensor(img[None, None, :, :], dtype=torch.float32).to(
                    next(self.parameters()).device
                )
            )
            self._ref_countries[ref_data.country_name[idx]] = embedding

    @torch.no_grad
    def rank_countries(self, drawings):
        embedding = self(drawings)
        countries = []
        distances = []

        if not self._ref_countries:
            raise Exception("First the reference dataset needs to be loaded!")

        for country, ref_emb in self._ref_countries.items():
            countries.append(country)
            distance = torch.linalg.norm(embedding - ref_emb, axis=-1)
            distances.append(distance.cpu())

        return countries, np.array(distances)


class CustomEmbeddingModel(nn.Module):
    def __init__(
        self,
        nr_conv_blocks=4,
        channels=16,
        embedding_size=128,
        dropout=0.2,
        shape=64,
        **kwargs,
    ):
        super().__init__()
        self.shape = (shape, shape)

        conv_blocks = [self.conv_block(1, channels)]
        for idx in range(nr_conv_blocks - 1):
            conv_blocks.append(
                self.conv_block(channels * 2**idx, channels * 2 ** (idx + 1))
            )
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(
            int(
                channels
                * 2 ** (nr_conv_blocks - 1)
                * (shape / (2**nr_conv_blocks)) ** 2
            ),
            embedding_size,
        )

    def conv_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = self.linear(x)
        return x


def fetch_model(model_name):
    # Get model version
    client = MlflowClient()
    try:
        model_version = client.get_model_version_by_alias(model_name, "champion")

    except Exception as e:
        # Fallback to default model
        model_version = client.get_model_version_by_alias("default", "champion")

    model_path = "/".join(model_version.source.split("/")[-5:])

    # Load the model
    try:
        model = load_model(model_path)

    except RuntimeError as e:
        # Fallback to CPU
        model = load_model(model_path, map_location=torch.device("cpu"))

    # Load reference data
    ref_data = Dataset(shape=model.shape)
    model.load_reference(ref_data)

    return model, next(model.parameters()).device
