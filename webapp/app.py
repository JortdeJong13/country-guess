import datetime
import hashlib
import os
import uuid
from typing import Dict, Optional

import requests
from flask import Flask, jsonify, render_template, request
from requests.exceptions import ConnectionError, HTTPError, Timeout

from countryguess.utils import proces_lines
from webapp.drawing_utils import DrawingStore, load_ranked_drawing, save_drawing

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
    drawing = proces_lines(lines)

    try:
        # Request prediction from ML server
        response = requests.post(f"{MLSERVER_URL}/predict", json=drawing, timeout=10)
        response.raise_for_status()
        ranking = response.json()

        # Store the drawing and ranking in the session
        drawing_id = str(uuid.uuid4())
        drawing_store.store(drawing_id, drawing, ranking)

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
    if not drawing_id or not drawing_store.contains(drawing_id):
        return jsonify({"message": "Drawing not found"}), 400

    drawing = drawing_store.get(drawing_id)
    ip_addr = request.remote_addr
    hashed_ip = hashlib.sha256(ip_addr.encode()).hexdigest() if ip_addr else None
    save_drawing(data["country"], drawing, hashed_ip, output_dir=DRAWING_DIR)

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


@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    debug = os.getenv("DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5002, debug=debug)
