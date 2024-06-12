from flask import Flask, request, jsonify
import mlflow.pytorch
from countryguess.data import Dataset
import torch


app = Flask(__name__)


print("This:", mlflow.get_tracking_uri())

# Load the model
#model = mlflow.pytorch.load_model(f"models:/triplet_model@Champion")
model = mlflow.pytorch.load_model("/server/mlruns/217199850258879704/99e3a0230ba84bababfb00c377a70b51/artifacts/model")

# Load reference data
reference_data = Dataset(shape=model.shape)
model.load_reference(reference_data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_tensor = torch.tensor(data['input'])
    output = model(input_tensor)
    result = output.detach().numpy().tolist()
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)