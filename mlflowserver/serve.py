from mlflow import MlflowClient
from mlflow.pytorch import load_model
from countryguess.data import Dataset

# Load model
client = MlflowClient()
model_version = client.get_latest_versions("triplet_model")
model_path = '/'.join(model_version[0].source.split('/')[-5:])
model = load_model(model_path)

# Load reference data 
model.load_reference(Dataset(shape=model.shape))

# Start inference server
model.serve(port=5001)