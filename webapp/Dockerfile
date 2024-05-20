FROM python:3.8-slim-buster
WORKDIR /webapp
COPY webapp/requirements.txt .
RUN pip install -r requirements.txt
COPY webapp/app.py .
COPY webapp/templates templates
COPY countryguess/data.py countryguess/
COPY countryguess/utils.py countryguess/
CMD ["python", "app.py"]
