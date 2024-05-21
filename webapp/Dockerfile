FROM python:3.8-slim-buster
WORKDIR /webapp
RUN apt-get update
RUN apt-get install -y gdal-bin libgdal-dev g++
COPY webapp/requirements.txt .
RUN pip install -r requirements.txt
COPY countryguess/data.py countryguess/
COPY countryguess/utils.py countryguess/
COPY webapp/templates templates
COPY webapp/app.py .
CMD ["python", "app.py"]
