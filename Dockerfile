# ---------- builder ----------
FROM python:3.12-slim AS builder

# gcc is needed here only; stripped in the runtime stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN uv sync --frozen --no-dev

# ---------- runtime ----------
FROM python:3.12-slim AS runtime

# Non-root user
RUN useradd --create-home --shell /bin/bash kormoran

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# PyBullet headless mode (no display required)
ENV PYBULLET_EGL=1
ENV PATH="/app/.venv/bin:$PATH"

USER kormoran

CMD ["kormoran", "--help"]
