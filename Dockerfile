
# Base image
FROM python:3.14-slim

# Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install production dependencies
COPY requirements-prod.txt .

RUN pip install --no-cache-dir \
    -r requirements-prod.txt \
    && pip check

# Copy application code and saved models
COPY . .

# Application port
EXPOSE 8000

# Start FastAPI + Gradio
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
