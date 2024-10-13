import os
import sys
from typing import List

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from requests.exceptions import ConnectionError, HTTPError, Timeout
from shapely import to_geojson

# Add the top-level directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from countryguess.utils import proces_lines, save_drawing

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="webapp/templates")


class DrawingInput(BaseModel):
    lines: List[List[List[float]]]


class FeedbackInput(BaseModel):
    country: str


# Global variable to store drawing
current_drawing = None


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/guess")
def guess(drawing_input: DrawingInput):
    global current_drawing

    lines = drawing_input.lines
    drawing = proces_lines(lines)

    # Store the drawing in the global variable
    current_drawing = drawing

    try:
        # Request prediction from ML server
        response = requests.post(
            f"http://{os.getenv("MLSERVER_URL", "localhost")}:5001/predict",
            json=to_geojson(drawing),
        )

        # Check if there is an error
        response.raise_for_status()

        return {"message": "Success", "ranking": response.json()}

    except (ConnectionError, Timeout) as conn_err:
        # Handle connection errors and timeouts
        raise HTTPException(
            status_code=502, detail=f"Server unreachable: {str(conn_err)}"
        )

    except (HTTPError, ValueError) as http_err:
        # Handle HTTP errors and JSON decoding errors
        raise HTTPException(status_code=500, detail=f"Server error: {str(http_err)}")


@app.post("/feedback")
def feedback(feedback_input: FeedbackInput):
    global current_drawing

    country_name = feedback_input.country
    drawing_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data/drawings.geojson")
    )
    save_drawing(country_name, current_drawing, path=drawing_path)

    # Clear the global variable after processing
    current_drawing = None

    return {"message": "Feedback received"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
