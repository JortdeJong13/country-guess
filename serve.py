from flask import Flask, request, jsonify
import mlflow.pytorch
import torch
import numpy as np
from shapely import from_geojson

from countryguess.data import Dataset
from countryguess.utils import lines_to_img, normalize_geom


app = Flask(__name__)


# Load the model
model = mlflow.pytorch.load_model(f"models:/triplet_model@Champion")
#model = mlflow.pytorch.load_model("/server/mlruns/217199850258879704/99e3a0230ba84bababfb00c377a70b51/artifacts/model")

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