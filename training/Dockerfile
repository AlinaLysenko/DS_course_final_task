# Start from a base Python image
FROM python:3.10

# Initialize settings_name variable with the default value
ARG settings_name=settings.json
ENV CONF_PATH=${settings_name}

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt
#--no-cache-dir
# Copy the training script and data into the container
COPY data/ ./data
COPY data_process/ ./data_process
COPY training/ ./training

COPY ${CONF_PATH} /app

RUN python3 data_process/data_loader.py

# Command to run the training script
RUN python3 training/train.py