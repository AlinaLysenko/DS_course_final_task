# Dockerfile for training
# Start from a base Python image
FROM python:3.10

ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

WORKDIR /app

# Install any necessary packages listed in requirements.txt
COPY requirements.txt ./
RUN pip install -r requirements.txt
# --no-cache-dir
# Copy data folder to Docker
COPY data/ ./data

# Copy models folder to Docker
COPY models/ ./models/

# Copy the code itself
COPY inference ./inference

# Copy configs
COPY ${CONF_PATH} /app

# Command to run the inference script
RUN python3 inference/run.py