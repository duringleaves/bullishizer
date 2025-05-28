FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Add cache busting for git clone
ARG CACHEBUST=1
RUN git clone https://github.com/duringleaves/bullishizer.git .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Rest of your Dockerfile stays the same...
RUN mkdir -p /app/data
EXPOSE 5555
ENV FLASK_APP=main.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

CMD ["python", "main.py"]