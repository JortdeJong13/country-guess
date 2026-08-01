import os

import requests
from flask import Flask, jsonify, render_template, request
from requests.exceptions import ConnectionError, HTTPError, Timeout

DRAWING_STORE_URL = os.environ["DRAWING_STORE_URL"]

app = Flask(__name__)


def fetch_summary(author_id=None):
    params = {"author_id": author_id} if author_id else None
    response = requests.get(
        f"{DRAWING_STORE_URL}/drawings/summary",
        params=params,
        timeout=5,
    )
    response.raise_for_status()
    return response.json()


@app.route("/")
def index():
    return render_template("index.html", is_admin=True)


@app.route("/unvalidated_drawing")
def unvalidated_drawing():
    try:
        response = requests.get(
            f"{DRAWING_STORE_URL}/drawings",
            params={"queue": "validation", "limit": 1},
            timeout=5,
        )
        response.raise_for_status()
        drawings = response.json().get("drawings", [])
        drawing = drawings[0] if drawings else None
        author_id = drawing.get("author_id") if drawing else None
        summary = fetch_summary(author_id)

        if drawing is None:
            return jsonify(
                {
                    "message": "No unvalidated drawings found",
                    "unvalidated_count": summary["unvalidated"],
                }
            )

        return jsonify(
            {
                "message": "Drawing loaded successfully",
                "id": drawing["id"],
                "lines": drawing["geometry"]["coordinates"],
                "timestamp": drawing["created_at"],
                "country_name": drawing.get("country"),
                "country_score": drawing.get("country_score"),
                "country_guess": drawing.get("country_guess"),
                "guess_score": drawing.get("guess_score"),
                "author": drawing.get("author"),
                "author_id": author_id,
                "unvalidated_count": summary["unvalidated"],
                "validated_author_count": summary.get("validated_by_author"),
            }
        )
    except (ConnectionError, Timeout) as error:
        return jsonify(
            {"message": "Drawing store unreachable", "error": str(error)}
        ), 502
    except HTTPError as error:
        return jsonify(
            {"message": "Failed to load unvalidated drawing", "error": str(error)}
        ), 502


@app.route("/drawing/<drawing_id>", methods=["PATCH"])
def update_drawing(drawing_id):
    request_data = request.get_json(silent=True)
    if not isinstance(request_data, dict):
        return jsonify({"message": "Invalid JSON data provided."}), 400

    try:
        response = requests.patch(
            f"{DRAWING_STORE_URL}/drawings/{drawing_id}",
            json=request_data,
            timeout=5,
        )
        response.raise_for_status()
        return jsonify({"message": f"Drawing '{drawing_id}' updated successfully."})
    except (ConnectionError, Timeout) as error:
        return jsonify(
            {"message": "Drawing store unreachable", "error": str(error)}
        ), 502
    except HTTPError as error:
        status = error.response.status_code if error.response is not None else 502
        return jsonify(
            {"message": "Failed to update drawing", "error": str(error)}
        ), status


@app.route("/drawing/<drawing_id>", methods=["DELETE"])
def delete_drawing(drawing_id):
    try:
        response = requests.delete(
            f"{DRAWING_STORE_URL}/drawings/{drawing_id}",
            timeout=5,
        )
        response.raise_for_status()
        return jsonify({"message": f"Drawing '{drawing_id}' deleted successfully."})
    except (ConnectionError, Timeout) as error:
        return jsonify(
            {"message": "Drawing store unreachable", "error": str(error)}
        ), 502
    except HTTPError as error:
        status = error.response.status_code if error.response is not None else 502
        return jsonify(
            {"message": "Failed to delete drawing", "error": str(error)}
        ), status


if __name__ == "__main__":
    debug = os.getenv("DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=5003, debug=debug)
