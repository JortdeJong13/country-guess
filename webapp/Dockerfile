FROM python:3.9-slim

# Install dependencies
RUN apt-get update
RUN apt-get install -y gdal-bin libgdal-dev g++ ffmpeg
COPY webapp/requirements.txt webapp/
RUN pip install -r webapp/requirements.txt

# Initialise an empty drawings file
RUN mkdir -p /data && echo '{"type": "FeatureCollection", "features": []}' > /data/drawings.geojson

# Copy code
COPY countryguess countryguess
COPY webapp/templates webapp/templates
COPY webapp/app.py webap/
COPY mlruns mlruns

CMD [ "python", "flask", "run", "--host=0.0.0.0"]