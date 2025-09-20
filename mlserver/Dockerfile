FROM python:3.9-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl gdal-bin libgdal-dev g++ ffmpeg \
    libgeos-dev libproj-dev \
    && rm -rf /var/lib/apt/lists/*
COPY mlserver/requirements.txt mlserver/
RUN pip install --no-cache-dir -r mlserver/requirements.txt

# Copy reference data and code
COPY data/reference data/reference
COPY mlruns/0 mlruns/
COPY mlruns/models/default mlruns/models/
COPY countryguess/utils.py countryguess/
COPY countryguess/data.py countryguess/
COPY countryguess/model.py countryguess/
COPY mlserver/serve.py mlserver/

CMD ["python", "-m", "mlserver.serve", "--host=0.0.0.0"]
