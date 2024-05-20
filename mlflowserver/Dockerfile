FROM python:3.11-slim-buster
WORKDIR /server
RUN pip install mlflow
COPY mlflowserver/serve.py .
COPY countryguess/data.py ./countryguess/
COPY countryguess/utils.py ./countryguess/
CMD ["python", "serve.py"]
