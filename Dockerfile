# =============================================================================
# The Panopticon Protocol v3 — Dockerfile
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Install core dependencies only (no torch/TRL for server container)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi==0.135.3 \
    "uvicorn[standard]==0.44.0" \
    pydantic==2.12.5 \
    httpx==0.28.1 \
    openai==2.30.0 \
    numpy==1.26.4 \
    tqdm==4.66.5

# Copy application code
COPY models.py .
COPY environment.py .
COPY hydra_policy.py .
COPY _server.py .
COPY client.py .
COPY inference.py .
COPY inference_local.py .
COPY argus_llm.py .
COPY security_policy.py .
COPY grader.py .
COPY smoke_test.py .
COPY gym_wrapper.py .

COPY tasks/ ./tasks/
COPY static/ ./static/

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

CMD ["uvicorn", "_server:app", "--host", "0.0.0.0", "--port", "7860"]
