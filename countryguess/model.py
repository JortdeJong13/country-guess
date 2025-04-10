import logging

import numpy as np
import torch
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.pytorch import load_model
from torch import nn

from .data import Dataset, geom_to_img

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        for idx, item in enumerate(ref_data):
            country_name, geom = item["country_name"], item["geometry"]
            img = geom_to_img(geom, ref_data.shape)
            embedding = self(
                torch.tensor(img[None, None, :, :], dtype=torch.float32).to(
                    next(self.parameters()).device
                )
            )
            self._ref_countries[country_name] = embedding

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
    logger.info("Fetching model: %s...", model_name)
    try:
        model_version = client.get_model_version_by_alias(model_name, "champion")

    except MlflowException:
        # Fallback to default model
        logger.warning("Unable to fetch %s, falling back to default model", model_name)
        model_version = client.get_model_version_by_alias("default", "champion")

    model_path = "/".join(model_version.source.split("/")[-5:])

    # Load the model
    logger.info("Loading model...")
    try:
        model = load_model(model_path)
        device = next(model.parameters()).device
        logger.info("Successfully loaded model")

    except RuntimeError:
        # Fallback to CPU
        logger.warning("Falling back to CPU")
        device = torch.device("cpu")
        model = load_model(model_path, map_location=device)
        logger.info("Successfully loaded model on CPU")

    # Load reference data
    ref_data = Dataset(shape=model.shape)
    model.load_reference(ref_data)

    return model, device
