"""Utility functions for processing drawings."""

import json
import logging
import random
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_drawing(drawing_dir="./data/drawings/"):
    """Loads a random user drawing."""
    drawing_files = list(Path(drawing_dir).glob("*.geojson"))
    if not drawing_files:
        logger.warning("No drawings found in the drawing directory.")
        return None

    drawing_file = random.choice(drawing_files)
    with open(drawing_file, "r", encoding="utf-8") as f:
        drawing = json.load(f)

    feature = drawing["features"][0]
    properties = feature.get("properties", {})

    return {
        "lines": feature["geometry"]["coordinates"],
        "country_name": properties.get("country_name", ""),
        "timestamp": properties.get("timestamp", ""),
        "country_guess": properties.get("country_guess", ""),
        "guess_score": properties.get("guess_score", None),
    }


def save_drawing(country_name, drawing, output_dir="./data/drawings/"):
    """Saves a country drawing as a GeoJSON file with metadata."""
    output_dir = Path(output_dir)
    logger.info("Saving drawing of %s to %s", country_name, output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()

    # Create filename for new drawing
    filename = f"{country_name.lower().replace(' ', '_')}_{timestamp}.geojson"

    # Create GeoJSON feature, including CRS information
    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "country_name": country_name,
                    "timestamp": timestamp,
                    "country_guess": drawing["guess"][0],
                    "guess_score": drawing["guess"][1],
                },
                "geometry": json.loads(drawing["geometry"]),
            }
        ],
    }

    # Save to file
    with open(output_dir / filename, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)


class DrawingStore:
    """Stores and manages user drawings."""

    def __init__(self, max_drawings: int = 10):
        self.drawings = {}
        self.max_drawings = max_drawings

    def store(self, drawing_id: str, drawing: str, guess: tuple[str, float]):
        """Stores a drawing with the given ID, removing oldest if at capacity."""
        if len(self.drawings) >= self.max_drawings:
            first_key = next(iter(self.drawings))
            self.drawings.pop(first_key)

        self.drawings[drawing_id] = {"geometry": drawing, "guess": guess}

    def get(self, drawing_id: str):
        """Retrieves a drawing by its ID."""
        return self.drawings.get(drawing_id)

    def remove(self, drawing_id: str):
        """Removes a drawing with the given ID if it exists."""
        self.drawings.pop(drawing_id, None)

    def contains(self, drawing_id: str):
        """Checks if a drawing with the given ID exists in the store."""
        return drawing_id in self.drawings
