from flask import Flask, render_template, request, jsonify
from shapely import to_geojson
import requests

# tmp
import sys
sys.path.insert(0, '/Users/jortdejong/GitHub/country-guess')

from countryguess.utils import proces_lines, save_drawing


app = Flask(__name__)


# This probably should be an env variable
mlserver_url = "http://127.0.0.1:5001/predict"

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

    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as conn_err:
        # Handle connection errors and timeouts
        return jsonify({'message': 'Server unreachable', 'error': str(conn_err)}), 502
    
    except (requests.exceptions.HTTPError, ValueError) as http_err:
        # Handle HTTP errors and JSON decoding errors
        return jsonify({'message': 'Server error', 'error': str(http_err)}), 500
    

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