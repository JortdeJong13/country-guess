FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y curl
COPY webapp/requirements.txt webapp/
RUN pip install -r webapp/requirements.txt

# Copy code
COPY countryguess/utils.py countryguess/
COPY webapp webapp

CMD ["python", "-m", "webapp.app", "--host=0.0.0.0"]
