FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y gdal-bin libgdal-dev g++ ffmpeg
COPY mlserver/requirements.txt mlserver/
RUN pip install -r mlserver/requirements.txt

# Copy reference data and code
COPY data/reference.geojson data/
COPY countryguess/utils.py countryguess/
COPY countryguess/data.py countryguess/
COPY countryguess/model.py countryguess/
COPY mlserver/serve.py mlserver/

CMD ["python", "mlserver/serve.py"]