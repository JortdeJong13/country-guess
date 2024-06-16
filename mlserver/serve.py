from flask import Flask, request, jsonify
import torch
import numpy as np
from shapely import from_geojson
import os
import sys

# Add the top-level directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from countryguess.data import lines_to_img
from countryguess.utils import normalize_geom
from countryguess.model import fetch_model


app = Flask(__name__)

# Load model
model = fetch_model(os.environ['MODEL_NAME'])


@app.route('/predict', methods=['POST'])
def predict():
    # Preproces input
    drawing = from_geojson(request.json)
    drawing = normalize_geom(drawing, model.shape)
    drawing = lines_to_img(drawing, model.shape)[None, None, :, :]
    drawing = torch.tensor(drawing, dtype=torch.float32)

    # Get ranking of countries
    countries, distances = model.rank_countries(drawing)
    ranking = zip(countries, np.squeeze(distances))
    ranking = sorted(ranking, key=lambda t: t[1])
    countries = [country for country, rank in ranking]

    return jsonify(countries)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)