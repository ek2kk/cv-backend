# === BUILDER ===
FROM python:3.13-bookworm AS builder

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.11.2

WORKDIR /app

COPY ./pyproject.toml ./uv.lock ./
COPY ./README.md ./

RUN uv sync --frozen --no-dev

# === PRODUCTION ===
FROM python:3.13-slim-bookworm AS production

WORKDIR /app

RUN apt-get update && apt-get install --no-install-recommends -y \
    curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY ./*.py ./
COPY ./data ./data
COPY ./resume.index ./resume.index
COPY ./resume_meta.json ./resume_meta.json

ENV PATH="/app/.venv/bin:$PATH"
ENV HF_HOME="/home/app/.cache/huggingface"

RUN groupadd --system app && \
    useradd --system --gid app --create-home --home-dir /home/app app && \
    mkdir -p /app/logs && \
    mkdir -p /home/app/.cache/huggingface && \
    chown -R app:app /app /home/app

USER app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
