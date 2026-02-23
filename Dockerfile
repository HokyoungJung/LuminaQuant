# Use official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build-essential for TA-Lib if needed, but we use binary wheels usually)
# If TA-Lib fails to install, we might need to compile it.
# For simplicity, we assume numpy/polars wheels work.
# TA-Lib python wrapper often requires the C library installed.
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C Library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xvzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Upgrade pip
RUN pip install --upgrade pip

# Copy project files
COPY . .

# Install Dependencies
# We use . to install the current package and dependencies from pyproject.toml
RUN pip install ".[live,optimize,dashboard]"

# Set Environment Variables
ENV PYTHONUNBUFFERED=1

# Default Command (Override in docker-compose)
CMD ["python", "run_live_ws.py"]
