from flask import Flask, render_template, request, jsonify, current_app
import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError
from shapely import to_geojson
import os
import sys

# Add the top-level directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from countryguess.utils import proces_lines, save_drawing


app = Flask(__name__)


mlserver_url = os.environ['MLSERVER_URL']

# Global variable to store drawing
current_drawing = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/guess', methods=['POST'])
def guess():
    global current_drawing
    
    data = request.json
    lines = data['lines']
    drawing = proces_lines(lines)

    # Store the drawing in the global variable
    current_drawing = drawing

    try:
        # Request prediction from ML server
        response = requests.post(mlserver_url, json=to_geojson(drawing))

        # Check if there is an error
        response.raise_for_status()

        return jsonify({'message': 'Success', 'ranking': response.json()})

    except (ConnectionError, Timeout) as conn_err:
        # Handle connection errors and timeouts
        return jsonify({'message': 'Server unreachable', 'error': str(conn_err)}), 502
    
    except (HTTPError, ValueError) as http_err:
        # Handle HTTP errors and JSON decoding errors
        return jsonify({'message': 'Server error', 'error': str(http_err)}), 500
    

@app.route('/feedback', methods=['POST'])
def feedback():
    global current_drawing
    
    data = request.json
    country_name = data['country']
    drawing_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/drawings.geojson'))
    save_drawing(country_name, current_drawing, path=drawing_path)

    # Clear the global variable after processing
    current_drawing = None
    
    return jsonify({'message': 'Feedback received'})


if __name__ == '__main__':
    app.run(debug=True)