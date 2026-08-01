FROM python:3.12-slim

WORKDIR /app
ENV PATH="/root/.local/bin:/app/.venv/bin:$PATH" \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/0.12.1/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Install only the web application dependency group.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --only-group web

# Copy code
COPY countryguess/utils.py countryguess/
COPY webapp webapp

CMD ["python", "-m", "webapp.app", "--host=0.0.0.0"]
