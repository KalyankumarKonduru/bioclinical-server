FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY bioclinical_server.py .

# Expose stdio (MCP over stdio)
# No port exposure needed for stdio mode

# Run server in stdio mode
CMD ["python", "bioclinical_server.py"]