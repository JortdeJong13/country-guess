FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y curl
COPY webapp/requirements.txt webapp/
RUN pip install -r webapp/requirements.txt

# Copy code
COPY countryguess/utils.py countryguess/
COPY webapp webapp

CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
