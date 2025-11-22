FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements_api.txt .

# Install Python dependencies
# NumPy 1.24.4 đã được pin trong requirements_api.txt để tương thích với implicit 0.7.0
RUN pip install --no-cache-dir -r requirements_api.txt && \
    pip check

# Copy application code
COPY . .

# Create model directory
RUN mkdir -p model

# Expose API port
EXPOSE 5000

# Default command (can be overridden in docker-compose)
CMD ["python", "api_service.py"]

