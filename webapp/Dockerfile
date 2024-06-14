FROM python:3.9-slim
RUN apt-get update
RUN apt-get install -y gdal-bin libgdal-dev g++ ffmpeg
COPY webapp/requirements.txt webapp/
RUN pip install -r webapp/requirements.txt
COPY countryguess countryguess
COPY webapp/templates webapp/templates
COPY webapp/app.py webap/
COPY mlruns mlruns
COPY data/reference.geojson data/
CMD [ "python", "flask", "run", "--host=0.0.0.0"]
