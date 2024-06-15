from flask import Flask, request, jsonify
import mlflow.pytorch
import torch
import numpy as np
from shapely import from_geojson
import os
import sys

# Add the top-level directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from countryguess.data import Dataset, lines_to_img
from countryguess.utils import normalize_geom


app = Flask(__name__)

print(mlflow.get_tracking_uri())

# tmp Get model path
from mlflow import MlflowClient
client = MlflowClient()
model_version = client.get_model_version_by_alias(os.environ['MODEL_NAME'], "Champion")
model_path = '/'.join(model_version.source.split('/')[-5:])

# Load the model
try:
    model = mlflow.pytorch.load_model(model_path)
except RuntimeError as e:
    # Fallback to CPU
    model = mlflow.pytorch.load_model(model_path, map_location=torch.device("cpu"))

# Load reference data
ref_data = Dataset(shape=model.shape)
model.load_reference(ref_data)


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