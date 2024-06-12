from flask import Flask, render_template, request, jsonify
from mlflow.pytorch import load_model

from countryguess.utils import proces_lines, save_drawing, predict
from countryguess.data import Dataset
from mlflow import MlflowClient

app = Flask(__name__)

# Load model
client = MlflowClient()
model_version = client.get_latest_versions("triplet_model")
model_path = '/'.join(model_version[0].source.split('/')[-5:])
model = load_model(model_path)
# Load reference data 
model.load_reference(Dataset(shape=model.shape))

# Global variable to store drawing
current_drawing = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/guess', methods=['POST'])
def guess():
    global current_drawing
    global model
    
    data = request.json
    lines = data['lines']
    drawing = proces_lines(lines)

    # Store the drawing in the global variable
    current_drawing = drawing

    # Get predicitions
    ranking = predict(model, drawing)

    return jsonify({'message': 'Success', 'ranking': ranking})


@app.route('/feedback', methods=['POST'])
def feedback():
    global current_drawing
    
    data = request.json
    country_name = data['country']
    save_drawing(country_name, current_drawing)

    # Clear the global variable after processing
    current_drawing = None
    
    return jsonify({'message': 'Feedback received'})


if __name__ == '__main__':
    app.run(debug=True)