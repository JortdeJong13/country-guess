"""Utility functions for processing drawings."""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Store scores for each drawing file
_SCORE_CACHE = {}


def load_drawing_file(drawing_file: Path):
    """Load and return a drawing file fully parsed."""
    logger.info(f"Loading {drawing_file}...")
    with open(drawing_file, "r", encoding="utf-8") as f:
        drawing = json.load(f)

    feature = drawing["features"][0]
    props = feature["properties"]

    return {"lines": feature["geometry"]["coordinates"], "properties": props}


def get_score(drawing_file: Path):
    """Return score from cache, loading from disk if missing."""
    if drawing_file in _SCORE_CACHE:
        return _SCORE_CACHE[drawing_file]

    # Add to cache
    drawing = load_drawing_file(drawing_file)
    props = drawing["properties"]
    score = props.get("country_score", None)
    if props["country_name"] != props["country_guess"]:
        score = None

    _SCORE_CACHE[drawing_file] = score

    return score


def load_ranked_drawing(rank, drawing_dir="./data/drawings/"):
    """Return the drawing with the given rank based on score."""
    drawing_files = list(Path(drawing_dir).glob("*.geojson"))
    if not drawing_files:
        logger.warning("No drawings found.")
        return None

    # Remove files with no score
    drawing_files = [f for f in drawing_files if get_score(f) is not None]

    # Sort files by score from highest to lowest
    drawing_files.sort(key=lambda f: get_score(f) or 0, reverse=True)

    total = len(drawing_files)
    if not (0 <= rank < total):
        logger.warning(f"Rank {rank} out of range (0 to {total - 1}).")
        return None

    # Load drawing with rank
    selected_file = drawing_files[rank]
    result = load_drawing_file(selected_file)
    result["rank"] = rank
    result["total"] = total

    return result


def save_drawing(country_name, drawing, hashed_ip, output_dir="./data/drawings/"):
    """Saves a country drawing as a GeoJSON file with metadata."""
    output_dir = Path(output_dir)
    logger.info("Saving drawing of %s to %s", country_name, output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat()

    # Create filename for new drawing
    filename = f"{country_name.lower().replace(' ', '_')}_{timestamp}.geojson"

    # Get the score of the correct country
    countries = drawing["ranking"]["countries"]
    if country_name in countries:
        idx = countries.index(country_name)
        score = drawing["ranking"]["scores"][idx]
    else:
        score = None

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
                    "country_score": score,
                    "timestamp": timestamp,
                    "country_guess": countries[0],
                    "hashed_ip": hashed_ip,
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

    def store(self, drawing_id: str, drawing: str, ranking: dict[str, list]):
        """Stores a drawing with the given ID, removing oldest if at capacity."""
        if len(self.drawings) >= self.max_drawings:
            first_key = next(iter(self.drawings))
            self.drawings.pop(first_key)

        self.drawings[drawing_id] = {"geometry": drawing, "ranking": ranking}

    def get(self, drawing_id: str):
        """Retrieves a drawing by its ID."""
        return self.drawings.get(drawing_id)

    def remove(self, drawing_id: str):
        """Removes a drawing with the given ID if it exists."""
        self.drawings.pop(drawing_id, None)

    def contains(self, drawing_id: str):
        """Checks if a drawing with the given ID exists in the store."""
        return drawing_id in self.drawings
