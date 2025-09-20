import os

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
    countries, scores = model.rank_countries(drawing)

    return jsonify({"countries": countries[0].tolist(), "scores": scores[0].tolist()})


@app.route("/countries")
def countries():
    """Return all loaded reference countries"""
    country_list = getattr(model, "ref_country_names", [])
    return jsonify({"countries": sorted(country_list)})


@app.route("/health")
def health():
    # Check if model is loaded
    if model is None:
        return jsonify({"status": "unhealthy", "error": "Model not loaded"}), 500

    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    debug = os.getenv("DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5001, debug=debug)
