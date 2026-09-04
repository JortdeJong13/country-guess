FROM python:3.12-slim

WORKDIR /app
ENV PATH="/root/.local/bin:/app/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y \
    ca-certificates curl gdal-bin libgdal-dev g++ ffmpeg \
    libgeos-dev libproj-dev \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/0.12.1/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Install only the ML server dependency group.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --only-group ml --no-install-package torch
RUN uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# Copy reference data and code
COPY data/reference data/reference
COPY mlruns/0 mlruns/
COPY mlruns/models/default mlruns/models/
COPY countryguess/utils.py countryguess/
COPY countryguess/data.py countryguess/
COPY countryguess/model.py countryguess/
COPY mlserver/serve.py mlserver/

CMD ["python", "-m", "mlserver.serve", "--host=0.0.0.0"]
