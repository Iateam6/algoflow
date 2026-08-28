# Base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libmagic1 \
    redis-server \
    libreoffice \
    libreoffice-writer \
    libreoffice-calc \
    libreoffice-impress \
    fonts-liberation \
    fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

# Verify LibreOffice installation
RUN soffice --version

# Install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Fix entrypoint permissions
RUN sed -i 's/\r$//' /app/docker/entrypoint.sh \
    && chmod +x /app/docker/entrypoint.sh

# Expose port
EXPOSE 8098

# Start application
ENTRYPOINT ["/app/docker/entrypoint.sh"]

CMD ["supervisord", "-c", "/app/docker/supervisord.conf"]
