FROM python:3.9-slim
WORKDIR /server
RUN apt-get update && apt-get install -y gdal-bin libgdal-dev g++ ffmpeg
COPY mlserver/requirements.txt .
RUN pip install -r requirements.txt
COPY data/reference.geojson data/
COPY countryguess countryguess
COPY mlserver/serve.py .
CMD ["python", "serve.py"]
