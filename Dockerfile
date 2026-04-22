# =============================================================================
# The Panopticon Protocol v3 — Dockerfile
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install core dependencies only (no torch/TRL for server container)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    "uvicorn[standard]==0.27.1" \
    pydantic==2.6.1 \
    httpx==0.26.0 \
    openai==1.12.0 \
    numpy==1.26.4 \
    tqdm==4.66.1

# Copy application code
COPY models.py .
COPY environment.py .
COPY _server.py .
COPY client.py .
COPY inference.py .
COPY grader.py .
COPY smoke_test.py .
COPY stub_env.py .
COPY gym_wrapper.py .

COPY tasks/ ./tasks/
COPY server/ ./server/

COPY pyproject.toml .
COPY openenv.yaml .
COPY README.md .
COPY LICENSE .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONIOENCODING=utf-8

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
