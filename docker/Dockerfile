# Stage 1: Build & Dependency Installation
FROM python:3.13-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml README.md ./
COPY src/ src/
COPY data/ data/

# Increase timeout for large dependencies
ENV UV_HTTP_TIMEOUT=120
ENV UV_CONCURRENT_DOWNLOADS=4

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e .

# Stage 2: Runtime Image
FROM python:3.13-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

ENV GIT_PYTHON_REFRESH=quiet

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "arabic_legal_document_qa.main:app", "--host", "0.0.0.0", "--port", "8000"]
