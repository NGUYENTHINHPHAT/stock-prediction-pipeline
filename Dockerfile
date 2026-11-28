FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p data/raw data/processed models/saved logs mlruns results

# Non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# 7860 is the Hugging Face Spaces default; override with PORT
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/health || exit 1

CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-7860}
