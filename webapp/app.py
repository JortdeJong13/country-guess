import datetime
import hashlib
import os
import uuid
from typing import Dict, Optional

import requests
from flask import Flask, jsonify, render_template, request
from requests.exceptions import ConnectionError, HTTPError, Timeout

from countryguess.utils import proces_lines
from webapp.drawing_utils import (
    Drawing,
    DrawingStore,
    load_ranked_drawing,
    save_drawing,
)

DRAWING_DIR = os.environ.get("DRAWING_DIR", "data/drawings")
MLSERVER_URL = os.environ["MLSERVER_URL"]
DAILY_COUNTRY: Dict[str, Optional[str]] = {"date": None, "country": None}

drawing_store = DrawingStore()
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


def get_daily_country():
    today = datetime.date.today().isoformat()

    # Check cache for daily country
    if DAILY_COUNTRY["date"] == today:
        return DAILY_COUNTRY["country"]

    # Get all reference countries from the ML server
    response = requests.get(f"{MLSERVER_URL}/countries", timeout=5)
    response.raise_for_status()
    countries = response.json()["countries"]

    # Pick a random country as the daily country
    hash = hashlib.sha256(f"{today}-country-guess-salt".encode()).hexdigest()
    index = int(hash, 16) % len(countries)
    country = countries[index]

    DAILY_COUNTRY["date"] = today
    DAILY_COUNTRY["country"] = country

    return country


@app.route("/daily_country")
def daily_country():
    try:
        country = get_daily_country()
        return jsonify({"country": country})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/guess", methods=["POST"])
def guess():
    data = request.json
    if not data or "lines" not in data:
        return jsonify({"message": "Invalid input"}), 400
    lines = data["lines"]
    geometry = proces_lines(lines)

    try:
        # Request prediction from ML server
        response = requests.post(f"{MLSERVER_URL}/predict", json=geometry, timeout=10)
        response.raise_for_status()
        ranking = response.json()["ranking"]

        # Create the drawing
        drawing = Drawing(
            geometry=geometry,
            timestamp=datetime.datetime.now().isoformat(),
            ranking=ranking,
        )

        # Store the drawing and ranking in the session
        drawing_id = str(uuid.uuid4())
        drawing_store.store(drawing_id, drawing)

        return jsonify({"ranking": ranking, "drawing_id": drawing_id})

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

    if "country" not in data:
        return jsonify({"message": "Country not provided"}), 400

    drawing_id = data.get("drawing_id")
    drawing = drawing_store.get(drawing_id)
    if not drawing:
        return jsonify({"message": "Drawing not found"}), 400

    # Add metadata to drawing
    drawing.country_name = data["country"]
    drawing.author = data.get("author")
    ip_addr = request.remote_addr
    hashed_ip = hashlib.sha256(ip_addr.encode()).hexdigest() if ip_addr else None
    drawing.hashed_ip = hashed_ip

    save_drawing(drawing, output_dir=DRAWING_DIR)

    drawing_store.remove(drawing_id)
    return jsonify({"message": "Feedback received"})


@app.route("/drawing")
def drawing():
    try:
        rank = int(request.args.get("rank", 0))
        drawing = load_ranked_drawing(rank, drawing_dir=DRAWING_DIR)

        if drawing is None:
            return jsonify({"message": f"No drawing found for rank {rank}"}), 404

        return jsonify(drawing)
    except Exception as e:
        return jsonify({"message": "Failed to load drawing", "error": str(e)}), 500


@app.route("/drawing/<path:filename>", methods=["DELETE"])
def delete_drawing(filename):
    file_path = os.path.join(DRAWING_DIR, filename)

    if not os.path.exists(file_path):
        return jsonify({"message": f"Drawing '{filename}' not found."}), 404

    # Prevent directory traversal
    if not os.path.abspath(file_path).startswith(os.path.abspath(DRAWING_DIR)):
        return jsonify({"message": "File outside the drawing directory."}), 403

    try:
        os.remove(file_path)
        return jsonify({"message": f"Drawing '{filename}' deleted successfully."}), 200
    except Exception as e:
        return jsonify({"message": "Failed to delete drawing", "error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    debug = os.getenv("DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5002, debug=debug)
