import os

import numpy as np
import torch
from flask import Flask, jsonify, request
from shapely import from_geojson

from countryguess.data import geom_to_img
from countryguess.model import fetch_model
from countryguess.utils import normalize_geom

app = Flask(__name__)

# Load model
model, device = fetch_model(os.getenv("MODEL_NAME", "default"))


@app.route("/predict", methods=["POST"])
def predict():
    # Preprocess input
    drawing = from_geojson(request.json)
    drawing = normalize_geom(drawing, model.shape)
    drawing = geom_to_img(drawing, model.shape)[None, None, :, :]
    drawing = torch.tensor(drawing, dtype=torch.float32).to(device)

    # Get ranking of countries
    countries, distances = model.rank_countries(drawing)
    ranking = zip(countries, np.squeeze(distances))
    ranking = sorted(ranking, key=lambda t: t[1])
    countries = [country for country, rank in ranking]

    return jsonify(countries)


@app.route("/health")
def health():
    # Check if model is loaded
    if model is None:
        return jsonify({"status": "unhealthy", "error": "Model not loaded"}), 500

    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    app.run(port=5001)
