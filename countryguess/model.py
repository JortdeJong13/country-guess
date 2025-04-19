"""Model functions for matching country shapes."""

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
    """Triplet model for embedding country drawings and reference shapes."""

    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model
        self._ref_countries = None

    @property
    def shape(self):
        """Return the shape of the embedding model."""
        return self.embedding_model.shape

    def forward(self, x):
        """Forward pass through the model."""
        return self.embedding_model(x)

    @torch.no_grad
    def load_reference(self, ref_data):
        """Embed reference countries."""
        assert ref_data.shape == self.shape
        self._ref_countries = {}
        for item in ref_data:
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
        """Rank countries based on their similarity to the given drawings."""
        embedding = self(drawings)
        countries, distances = [], []

        if not self._ref_countries:
            raise RuntimeError("First the reference dataset needs to be loaded!")

        for country, ref_emb in self._ref_countries.items():
            countries.append(country)
            distance = torch.linalg.norm(embedding - ref_emb, axis=-1)
            distances.append(distance.cpu())

        countries = np.array(countries)
        distances = np.array(distances).T

        # Calculate confidence scores
        similarities = -distances
        exp_similarities = np.exp(
            similarities - np.max(similarities, axis=1, keepdims=True)
        )
        confidences = exp_similarities / np.sum(exp_similarities, axis=1, keepdims=True)

        # Sort by distances (ascending order)
        idx = np.argsort(distances, axis=1)
        countries = countries[idx]
        confidences = np.take_along_axis(confidences, idx, axis=1)

        return countries, confidences


class CustomEmbeddingModel(nn.Module):
    """Custom embedding model for embedding country shapes."""

    def __init__(self, **kwargs):
        super().__init__()
        self.shape = (kwargs.get("shape", 64), kwargs.get("shape", 64))
        channels = kwargs.get("channels", 16)
        nr_conv_blocks = kwargs.get("nr_conv_blocks", 4)
        embedding_size = kwargs.get("embedding_size", 128)
        dropout = kwargs.get("dropout", 0.2)

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
                * (self.shape[0] / (2**nr_conv_blocks)) ** 2
            ),
            embedding_size,
        )

    def conv_block(self, in_dim, out_dim):
        """Convolutional block with batch normalization and ReLU activation."""
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
        """Forward pass through the model."""
        x = self.conv_blocks(x)
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = self.linear(x)
        return x


def fetch_model(model_name):
    """Fetch and load model and load reference countries"""
    client = MlflowClient()
    logger.info("Fetching model: %s...", model_name)
    try:
        model_version = client.get_model_version_by_alias(model_name, "champion")

    except MlflowException:
        # Fallback to default model
        logger.warning("Unable to fetch %s, falling back to default model", model_name)
        model_version = client.get_model_version_by_alias("default", "champion")

    # Load the model
    model_uri = f"runs:/{model_version.run_id}/model"
    logger.info("Loading model from: %s", model_uri)
    try:
        model = load_model(model_uri)
        device = next(model.parameters()).device

    except RuntimeError:
        # Fallback to CPU
        logger.warning("Failed to load model, falling back to CPU")
        device = torch.device("cpu")
        model = load_model(model_uri, map_location=device)

    # Load reference data
    ref_data = Dataset(shape=model.shape)
    model.load_reference(ref_data)

    logger.info("Successfully loaded model on: %s", device)

    return model, device
