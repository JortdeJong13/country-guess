import datetime
import hashlib
import json
import os
import uuid

import requests
from flask import Flask, jsonify, make_response, render_template, request
from requests.exceptions import ConnectionError, HTTPError, Timeout

from countryguess.utils import proces_lines

MLSERVER_URL = os.environ["MLSERVER_URL"]
DRAWING_STORE_URL = os.environ["DRAWING_STORE_URL"]
DAILY_COUNTRY: dict[str, str | None] = {"date": None, "country": None}

app = Flask(__name__)


@app.route("/")
def index():
    if request.cookies.get("author_id"):
        return render_template("index.html")

    author_id = str(uuid.uuid4())
    response = make_response(render_template("index.html"))
    response.set_cookie(
        "author_id",
        author_id,
        max_age=60 * 60 * 24 * 365,  # 1 year
        httponly=True,
    )
    return response


@app.route("/robots.txt")
def serve_robots_txt():
    return app.send_static_file("robots.txt")


def get_daily_country():
    today = datetime.datetime.now(datetime.UTC).date().isoformat()

    # Check cache for daily country
    if DAILY_COUNTRY["date"] == today:
        return DAILY_COUNTRY["country"]

    # Get all reference countries from the ML server
    response = requests.get(f"{MLSERVER_URL}/countries", timeout=5)
    response.raise_for_status()
    countries = response.json()["countries"]

    # Pick a random country as the daily country
    digest = hashlib.sha256(f"{today}-country-guess-salt".encode()).hexdigest()
    index = int(digest, 16) % len(countries)
    country = countries[index]

    DAILY_COUNTRY["date"] = today
    DAILY_COUNTRY["country"] = country

    return country


@app.route("/daily_country")
def daily_country():
    try:
        country = get_daily_country()
        return jsonify({"country": country})
    except requests.RequestException as error:
        return jsonify({"error": str(error)}), 502
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 502


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

        store_response = requests.post(
            f"{DRAWING_STORE_URL}/drawings",
            json={
                "geometry": json.loads(geometry),
                "ranking": [
                    {"country": country, "score": score} for country, score in ranking
                ],
                "author_id": request.cookies.get("author_id"),
            },
            timeout=5,
        )
        store_response.raise_for_status()
        drawing_id = store_response.json()["id"]

        return jsonify({"ranking": ranking, "drawing_id": drawing_id})

    except (ConnectionError, Timeout) as conn_err:
        return jsonify({"message": "Server unreachable", "error": str(conn_err)}), 502

    except (HTTPError, ValueError) as http_err:
        return jsonify({"message": "Server error", "error": str(http_err)}), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json
    if not data:
        return jsonify({"message": "Invalid input"}), 400

    if "country" not in data:
        return jsonify({"message": "Country not provided"}), 400

    drawing_id = data.get("drawing_id")
    if not drawing_id:
        return jsonify({"message": "Drawing ID not provided"}), 400

    try:
        response = requests.patch(
            f"{DRAWING_STORE_URL}/drawings/{drawing_id}",
            json={
                "country": data["country"],
                "author": data.get("author"),
            },
            timeout=5,
        )
        response.raise_for_status()
        return jsonify({"message": "Feedback received"})
    except (ConnectionError, Timeout) as conn_err:
        return jsonify(
            {"message": "Drawing store unreachable", "error": str(conn_err)}
        ), 502
    except HTTPError as http_err:
        if http_err.response is not None and http_err.response.status_code == 404:
            return jsonify({"message": "Drawing not found"}), 404
        return jsonify({"message": "Drawing store error", "error": str(http_err)}), 502


@app.route("/report", methods=["POST"])
def report():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"message": "Invalid input"}), 400

    drawing_id = data.get("drawing_id")
    if not isinstance(drawing_id, str) or not drawing_id:
        return jsonify({"message": "Drawing ID not provided"}), 400

    try:
        response = requests.patch(
            f"{DRAWING_STORE_URL}/drawings/{drawing_id}",
            json={"report": True},
            timeout=5,
        )
        response.raise_for_status()
        return jsonify({"message": "Drawing reported"})
    except (ConnectionError, Timeout) as conn_err:
        return jsonify(
            {"message": "Drawing store unreachable", "error": str(conn_err)}
        ), 502
    except HTTPError as http_err:
        if http_err.response is not None and http_err.response.status_code == 404:
            return jsonify({"message": "Drawing not found"}), 404
        return jsonify({"message": "Drawing store error", "error": str(http_err)}), 502


@app.route("/drawing")
def drawing():
    rank_value = request.args.get("rank", "0")
    try:
        rank = int(rank_value)
    except ValueError:
        return jsonify({"message": "Invalid rank"}), 400
    if rank < 0:
        return jsonify({"message": "Invalid rank"}), 400

    try:
        response = requests.get(
            f"{DRAWING_STORE_URL}/leaderboard",
            params={"rank": rank},
            timeout=5,
        )
        response.raise_for_status()
        result = response.json()
        stored = result["drawing"]
        return jsonify(
            {
                "id": stored["id"],
                "lines": stored["geometry"]["coordinates"],
                "rank": result["rank"],
                "total": result["total"],
                "country_name": stored.get("country"),
                "country_score": stored.get("country_score"),
                "author": stored.get("author"),
                "timestamp": stored["created_at"],
            }
        )
    except (ConnectionError, Timeout) as conn_err:
        return jsonify(
            {"message": "Drawing store unreachable", "error": str(conn_err)}
        ), 502
    except HTTPError as http_err:
        if http_err.response is not None and http_err.response.status_code == 404:
            return jsonify({"message": f"No drawing found for rank {rank}"}), 404
        return jsonify(
            {"message": "Failed to load drawing", "error": str(http_err)}
        ), 502


@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    debug = os.getenv("DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5002, debug=debug)
