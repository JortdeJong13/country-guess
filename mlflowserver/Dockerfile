FROM python:3.9-slim
WORKDIR /server
RUN apt-get update
RUN apt-get install -y gdal-bin libgdal-dev g++ ffmpeg
COPY mlflowserver/requirements.txt .
RUN pip install -r requirements.txt
COPY data/reference.geojson data/
COPY countryguess countryguess
COPY mlflowserver/serve.py .
CMD ["python", "serve.py"]
