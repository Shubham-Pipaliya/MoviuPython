FROM python:3.11.8-slim-bookworm

# Set work directory
WORKDIR /app

# Install system dependencies required for building scientific packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    liblapack-dev \
    libatlas-base-dev \
    libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

# Expose API port
EXPOSE 8000

# Start FastAPI app using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
