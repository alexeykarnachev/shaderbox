FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --no-dev

COPY shaderbox ./shaderbox/
COPY .env .

ENV PYTHONPATH=/app

CMD ["/app/.venv/bin/uvicorn", "shaderbox.app:app", "--host", "0.0.0.0", "--port", "8229"]

