from mlflow.pytorch import load_model
from countryguess.data import Dataset

# Load model
model = load_model("models:/triplet_model@champion")

# Load reference data 
model.load_reference(Dataset(shape=model.shape))

# Start inference server
model.serve(port=5001)