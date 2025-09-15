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
        self._ref_country_names = None
        self._ref_country_embeddings = None

    @property
    def shape(self):
        """Return the shape of the embedding model."""
        return self.embedding_model.shape

    def forward(self, x):
        """Forward pass through the model."""
        return self.embedding_model(x)

    @torch.no_grad()
    def load_reference(self, ref_data):
        """Embed reference countries."""
        if ref_data.shape != self.shape:
            raise ValueError(
                f"ref_data shape {ref_data.shape} does not match expected {self.shape}"
            )

        country_names, imgs = [], []
        for item in ref_data:
            country_names.append(item["country_name"])
            geom = item["geometry"]
            img = geom_to_img(geom, ref_data.shape)
            imgs.append(img)
        imgs = np.stack(imgs)  # [num_countries, H, W]

        device = next(self.parameters()).device
        imgs_tensor = torch.tensor(imgs[:, None, :, :], dtype=torch.float32).to(device)
        embeddings = self(imgs_tensor)  # [num_countries, embedding_dim]

        self._ref_country_names = country_names
        self._ref_country_embeddings = embeddings  # [num_countries, embedding_dim]

    @torch.no_grad()
    def rank_countries(self, drawings):
        """Rank countries based on their similarity to the given drawings."""
        embedding = self(drawings)  # [batch, embedding_dim]
        if self._ref_country_embeddings is None:
            raise RuntimeError("First the reference dataset needs to be loaded!")

        # Compute pairwise distances: [batch, num_countries]
        distances = torch.cdist(embedding, self._ref_country_embeddings)

        # Calculate confidence scores (softmax over negative distances)
        similarities = -distances
        confidences = torch.softmax(similarities, dim=1)  # [batch, num_countries]

        # Sort by distances (ascending order)
        idx = torch.argsort(distances, dim=1)
        countries = np.array(self._ref_country_names)[idx.cpu().numpy()]
        confidences = torch.gather(confidences, 1, idx)

        return countries, confidences.cpu().numpy()


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

        # Dynamically determine the input size for the linear layer
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *self.shape)
            out = self.conv_blocks(dummy)
            flattened_size = out.flatten(start_dim=1).shape[1]

        self.linear = nn.Linear(flattened_size, embedding_size)

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


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


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

    if not model_version or not model_version.source:
        raise RuntimeError(f"Failed to fetch model version or source for {model_name}")

    # Load the model
    model_path = "/".join(model_version.source.split("/")[-5:])
    device = get_device()
    logger.info("Loading model from path: %s on device: %s...", model_path, device)
    model = load_model(model_path, map_location=device)

    # Load reference data
    ref_data = Dataset(shape=model.shape)
    model.load_reference(ref_data)

    logger.info("Successfully loaded model and reference data")

    return model, device
