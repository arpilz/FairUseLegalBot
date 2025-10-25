# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY RetrievalTesting/Data /app/RetrievalTesting/Data
COPY RetrievalTesting/Analysis /app/RetrievalTesting/Analysis
COPY RetrievalTesting/embedder.py /app/RetrievalTesting/embedder.py
COPY RetrievalTesting/evaluator.py /app/RetrievalTesting/evaluator.py
COPY RetrievalTesting/query.py /app/RetrievalTesting/query.py
COPY RetrievalTesting/util.py /app/RetrievalTesting/util.py
COPY RetrievalTesting/API /app/RetrievalTesting/API

ENV PYTHONPATH="/app/RetrievalTesting:/app/RetrievalTesting/API"

WORKDIR /app/RetrievalTesting/API

ENV PORT=8080
CMD exec gunicorn -k uvicorn.workers.UvicornWorker \
    --workers ${WORKERS:-2} --bind 0.0.0.0:${PORT} app:app
