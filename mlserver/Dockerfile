FROM python:3.9-slim
RUN apt-get update && apt-get install -y gdal-bin libgdal-dev g++ ffmpeg
COPY mlserver/requirements.txt mlserver/
RUN pip install -r mlserver/requirements.txt
COPY data/reference.geojson data/
COPY countryguess countryguess
COPY mlserver/serve.py mlserver/
CMD ["python", "mlserver/serve.py"]
