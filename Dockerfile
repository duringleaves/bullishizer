# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal - just curl for health check)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for database
RUN mkdir -p /app/data

# Expose port
EXPOSE 5555

# Environment variables
ENV FLASK_APP=stock_tracker_no_talib.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Run the application
CMD ["python", "main.py"]