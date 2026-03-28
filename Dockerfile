FROM python:3.11-slim

# HF Spaces runs as a non-root user; set workdir and permissions early
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY baseline.py .
COPY openenv.yaml .

# HF Spaces expects the server on port 7860
EXPOSE 7860

# Healthcheck so HF knows the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Start the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
