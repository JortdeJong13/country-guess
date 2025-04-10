import os
import sys
import uuid

import requests
from flask import Flask, jsonify, render_template, request, session
from requests.exceptions import ConnectionError, HTTPError, Timeout

# Add the top-level directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from countryguess.utils import proces_lines, save_drawing

app = Flask(__name__)

# Set secret key for session
app.secret_key = os.urandom(24)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/guess", methods=["POST"])
def guess():
    data = request.json
    lines = data["lines"]
    drawing = proces_lines(lines)

    # Store the drawing in the session
    drawing_id = str(uuid.uuid4())
    session[drawing_id] = drawing

    try:
        # Request prediction from ML server
        response = requests.post(os.environ["MLSERVER_URL"], json=drawing)

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
    country_name = data.get("country")
    drawing_id = data.get("drawing_id")

    # Retrieve the drawing from the session
    if drawing_id and drawing_id in session:
        drawing = session[drawing_id]
        save_drawing(country_name, drawing)
        del session[drawing_id]

        return jsonify({"message": "Feedback received"})
    else:
        return jsonify({"message": "Drawing not found"}), 400


if __name__ == "__main__":
    app.run(debug=True)
