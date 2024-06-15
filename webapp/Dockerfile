FROM python:3.9-slim

# Install dependencies
COPY webapp/requirements.txt webapp/
RUN pip install -r webapp/requirements.txt

# Initialise an empty drawings file
RUN mkdir -p /data && echo '{"type": "FeatureCollection", "features": []}' > /data/drawings.geojson

# Copy code
COPY countryguess/utils.py countryguess/
COPY webapp/templates webapp/templates
COPY webapp/app.py webapp/

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]