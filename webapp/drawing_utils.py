"""Utility functions for processing drawings."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Store scores for each drawing file
_SCORE_CACHE = {}


@dataclass
class Drawing:
    geometry: str
    timestamp: str
    ranking: list[tuple[str, float]]
    country_name: Optional[str] = None
    author: Optional[str] = None
    hashed_ip: Optional[str] = None
    validated: bool = False
    filename: Optional[str] = None

    @property
    def country_score(self) -> Optional[float]:
        """Return the score for the correct country."""
        if not self.country_name:
            return None

        for country, score in self.ranking:
            if country == self.country_name:
                return score
        return None

    @property
    def country_guess(self) -> str:
        """Return the guessed country."""
        return self.ranking[0][0]

    @property
    def guess_score(self) -> float:
        """Return the score for the guessed country."""
        return self.ranking[0][1]


def load_drawing(drawing_file: Path) -> Drawing:
    """Load and return a drawing file fully parsed."""
    logger.info(f"Loading {drawing_file}...")
    with open(drawing_file, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    # Extract the drawing properties
    feature = geojson["features"][0]
    props = feature["properties"]
    geometry = json.dumps(feature["geometry"])

    return Drawing(
        geometry=geometry,
        timestamp=props.get("timestamp"),
        ranking=props.get("ranking"),
        country_name=props.get("country_name"),
        author=props.get("author"),
        hashed_ip=props.get("hashed_ip"),
        validated=props.get("validated", False),
        filename=drawing_file.name,
    )


def get_score(drawing_file: Path):
    """Return score from cache, loading from disk if missing."""
    if drawing_file in _SCORE_CACHE:
        return _SCORE_CACHE[drawing_file]

    # Add to cache
    drawing = load_drawing(drawing_file)
    score = drawing.country_score
    if drawing.country_name != drawing.country_guess:
        # Return None for incorrect guesses
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
    drawing = load_drawing(selected_file)
    lines = json.loads(drawing.geometry)["coordinates"]
    result = {
        "lines": lines,
        "rank": rank,
        "total": total,
        "country_name": drawing.country_name,
        "country_score": drawing.country_score,
        "author": drawing.author,
        "timestamp": drawing.timestamp,
    }

    return result


def load_unvalidated_drawing(drawing_dir="./data/drawings/"):
    """Load a drawing that has not been validated."""
    drawing_files = list(Path(drawing_dir).glob("*.geojson"))
    if not drawing_files:
        logger.warning("No drawings found.")
        return None

    for drawing_file in drawing_files:
        drawing = load_drawing(drawing_file)
        if not drawing.validated:
            return drawing

    logger.info("No unvalidated drawings found.")
    return None


def save_drawing(drawing, filename=None, output_dir="./data/drawings/"):
    """Saves a country drawing as a GeoJSON file with metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving drawing of %s to %s", drawing.country_name, output_dir)

    # Create filename for new drawing
    if filename is None:
        filename = f"{drawing.country_name.lower().replace(' ', '_')}_{drawing.timestamp}.geojson"

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
                    "timestamp": drawing.timestamp,
                    "country_name": drawing.country_name,
                    "author": drawing.author,
                    "hashed_ip": drawing.hashed_ip,
                    "validated": drawing.validated,
                    "ranking": drawing.ranking,
                },
                "geometry": json.loads(drawing.geometry),
            }
        ],
    }

    # Save to file
    with open(output_dir / filename, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)


class DrawingStore:
    """Stores and manages user drawings."""

    def __init__(self, max_drawings: int = 10):
        self.drawings: dict[str, Drawing] = {}
        self.max_drawings = max_drawings

    def store(self, drawing_id: str, drawing: Drawing):
        """Stores a drawing with the given ID, removing oldest if at capacity."""
        if len(self.drawings) >= self.max_drawings:
            first_key = next(iter(self.drawings))
            self.drawings.pop(first_key)

        self.drawings[drawing_id] = drawing

    def get(self, drawing_id: str):
        """Retrieves a drawing by its ID."""
        return self.drawings.get(drawing_id)

    def remove(self, drawing_id: str):
        """Removes a drawing with the given ID if it exists."""
        self.drawings.pop(drawing_id, None)
