from flask import Flask, request, jsonify
import mlflow.pytorch
import torch
from shapely import from_geojson

from countryguess.data import Dataset
from countryguess.utils import poly_to_img


app = Flask(__name__)

# Load the model
#model = mlflow.pytorch.load_model(f"models:/triplet_model@Champion")
model = mlflow.pytorch.load_model("/server/mlruns/217199850258879704/99e3a0230ba84bababfb00c377a70b51/artifacts/model")

# Load reference data
reference_data = Dataset(shape=model.shape)
model.load_reference(reference_data)

@app.route('/predict', methods=['POST'])
def predict():
    drawing = from_geojson(request.json)
    drawing = poly_to_img(drawing, model.shape)[None, None, :, :]
    drawing = torch.tensor(drawing, dtype=torch.float32)
    countries, distances = model.rank_countries(drawing)
    print(countries)
    print(distances)


    result = output.detach().numpy().tolist()
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)