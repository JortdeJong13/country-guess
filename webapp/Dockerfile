FROM python:3.9-slim
WORKDIR /webapp
RUN apt-get update
RUN apt-get install -y gdal-bin libgdal-dev g++ ffmpeg
COPY webapp/requirements.txt .
RUN pip install -r requirements.txt
COPY countryguess countryguess
COPY webapp/templates templates
COPY webapp/app.py .
COPY mlruns mlruns
COPY data/reference.geojson data/
CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]
