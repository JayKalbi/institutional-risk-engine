# Multi-Stage Production Dockerfile for HybridCredit-LLM
# Base Image: Official Lightweight Python 3.10 (Locks exact Python version globally)
FROM python:3.10-slim

# Prevent Python from writing .pyc files & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Set working directory inside container
WORKDIR /app

# Install system dependencies (build essential tools for C++ extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (enables Docker layer caching)
COPY requirements-web.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-web.txt

# Copy application source code into container
COPY . .

# Expose port 5000 for web traffic
EXPOSE 5000

# Production launch command using Gunicorn WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "120", "flask-app.app:app"]
