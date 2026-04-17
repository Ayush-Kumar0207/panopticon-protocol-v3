# =============================================================================
# OpenEnv Starter Kit — Dockerfile
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py .
COPY environment.py .
COPY _server.py .
COPY client.py .
COPY inference.py .
COPY grader.py .
COPY smoke_test.py .

COPY tasks/ ./tasks/
COPY server/ ./server/
COPY static/ ./static/

COPY pyproject.toml .
COPY openenv.yaml .
COPY README.md .
COPY LICENSE .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
