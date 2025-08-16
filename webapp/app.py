import os
import uuid

import requests
from flask import Flask, jsonify, render_template, request
from requests.exceptions import ConnectionError, HTTPError, Timeout

from countryguess.utils import proces_lines, save_drawing, DrawingStore

DRAWING_DIR = os.environ.get("DRAWING_DIR", "data/drawings")
MLSERVER_URL = os.environ["MLSERVER_URL"]
drawing_store = DrawingStore()
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/guess", methods=["POST"])
def guess():
    data = request.json
    if not data or "lines" not in data:
        return jsonify({"message": "Invalid input"}), 400
    lines = data["lines"]
    drawing = proces_lines(lines)

    # Store the drawing in the session
    drawing_id = str(uuid.uuid4())
    drawing_store.store(drawing_id, drawing)

    try:
        # Request prediction from ML server
        response = requests.post(MLSERVER_URL, json=drawing, timeout=10)

        # Check if there is an error
        response.raise_for_status()

        return jsonify(
            {"message": "Success", "ranking": response.json(), "drawing_id": drawing_id}
        )

    except (ConnectionError, Timeout) as conn_err:
        # Handle connection errors and timeouts
        return jsonify({"message": "Server unreachable", "error": str(conn_err)}), 502

    except (HTTPError, ValueError) as http_err:
        # Handle HTTP errors and JSON decoding errors
        return jsonify({"message": "Server error", "error": str(http_err)}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    if not data:
        return jsonify({"message": "Invalid input"}), 400
    country_name = data.get("country")
    drawing_id = data.get("drawing_id")

    if not drawing_id or not drawing_store.contains(drawing_id):
        return jsonify({"message": "Drawing not found"}), 400

    if country_name:
        drawing = drawing_store.get(drawing_id)
        save_drawing(country_name, drawing, output_dir=DRAWING_DIR)

    drawing_store.remove(drawing_id)
    return jsonify({"message": "Feedback received"})


@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    debug = os.getenv("DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5002, debug=debug)
