# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# Stage 1: build the API virtualenv (production deps only)
# ---------------------------------------------------------------------------
FROM python:3.12.15-slim-bookworm AS builder-api

# Copy uv binary into the build stage
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies first: this layer is cached until pyproject.toml / uv.lock change
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Then the project itself
COPY pyproject.toml uv.lock README.md* ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Stage 2 (optional): ingestion venv, adds the PDF parsing dependency (PyMuPDF)
#   docker build --target ingest -t ai-service-ingest .
# ---------------------------------------------------------------------------
FROM python:3.12.15-slim-bookworm AS builder-ingest

# Copy uv binary here as well
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app
COPY pyproject.toml uv.lock README.md* ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ingest

FROM python:3.12.15-slim-bookworm AS ingest

# PyMuPDF uses these shared libraries at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 1001 app && useradd --system --uid 1001 --gid app --create-home app
WORKDIR /app
COPY --from=builder-ingest --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
USER app
CMD ["python", "-m", "ai_service.ingestion"]

# ---------------------------------------------------------------------------
# Stage 3 (default): slim runtime image for the chat API
# ---------------------------------------------------------------------------
FROM python:3.12.15-slim-bookworm AS api

RUN groupadd --system --gid 1001 app && useradd --system --uid 1001 --gid app --create-home app

WORKDIR /app
COPY --from=builder-api --chown=app:app /app /app
# Migrations ship with the image and are run as a separate job (see notes), not at startup
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app migrations ./migrations

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    WEB_CONCURRENCY=2

USER app
EXPOSE 8000

# Adjust the path if your health route differs
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request as u; u.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=3)" || exit 1

# Shell form so $PORT and $WEB_CONCURRENCY expand; exec keeps uvicorn as PID 1 for clean shutdown
CMD exec uvicorn ai.main:app --host 0.0.0.0 --port "$PORT" --workers "$WEB_CONCURRENCY" --proxy-headers