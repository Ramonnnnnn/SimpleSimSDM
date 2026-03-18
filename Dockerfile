# =========================
# Base image
# =========================
FROM python:3.11-slim

# =========================
# Environment variables
# =========================
ENV POETRY_VERSION=1.7.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1 \
    CONFIG_PATH=configuration_files/config.yaml

# =========================
# System dependencies
# =========================
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Install Poetry
# =========================
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# =========================
# Set working directory
# =========================
WORKDIR /app

# =========================
# Copy dependency files (for caching)
# =========================
COPY pyproject.toml poetry.lock ./

# =========================
# Install Python dependencies
# =========================
RUN poetry install --no-root --only main

# =========================
# Copy project files
# =========================
COPY . .

# =========================
# Default command
# =========================
CMD ["python", "main.py"]