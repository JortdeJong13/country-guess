import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from webapp.drawing_utils import load_drawing, load_unvalidated_drawing, save_drawing

DRAWING_DIR = os.environ.get("DRAWING_DIR", "data/drawings")

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", is_admin=True)


@app.route("/unvalidated_drawing")
def unvalidated_drawing():
    try:
        drawing = load_unvalidated_drawing(drawing_dir=DRAWING_DIR)
        if drawing is None:
            return jsonify({"message": "No unvalidated drawing found"}), 404

        return jsonify(
            {
                "lines": json.loads(drawing.geometry)["coordinates"],
                "timestamp": drawing.timestamp,
                "country_name": drawing.country_name,
                "country_score": drawing.country_score,
                "country_guess": drawing.country_guess,
                "author": drawing.author,
                "filename": drawing.filename,
            }
        )
    except Exception as e:
        return jsonify(
            {"message": "Failed to load unvalidated drawing", "error": str(e)}
        ), 500


@app.route("/drawing/<path:filename>", methods=["PUT"])
def update_drawing(filename):
    file_path = os.path.join(DRAWING_DIR, filename)

    if not os.path.exists(file_path):
        return jsonify({"message": f"Drawing '{filename}' not found."}), 404

    # Prevent directory traversal
    if not os.path.abspath(file_path).startswith(os.path.abspath(DRAWING_DIR)):
        return jsonify({"message": "File outside the drawing directory."}), 403

    request_data = request.json
    if not request_data or not isinstance(request_data, dict):
        return jsonify({"message": "Invalid JSON data provided."}), 400

    try:
        # Load drawing
        drawing = load_drawing(Path(file_path))

        # Update drawing fields
        for field_name, value in request_data.items():
            setattr(drawing, field_name, value)

        # Save updated drawing
        save_drawing(drawing, filename, output_dir=DRAWING_DIR)

        return jsonify({"message": f"Drawing '{filename}' updated successfully."}), 200
    except Exception as e:
        return jsonify({"message": "Failed to update drawing", "error": str(e)}), 500


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
