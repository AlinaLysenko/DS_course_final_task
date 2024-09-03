#!/bin/bash

# Load settings from the JSON configuration file
PROJECT_DIR=$(pwd)
WORK_DIR="/app"
DATA_DIR="/data"
MODELS_DIR="/models"
RESULTS_DIR="/results"


# Function to safely remove directories
safe_remove() {
    dir=$1
    if [ -d "$dir" ]; then
        echo "Removing directory: $dir"
        rm -rf "$dir"
    else
        echo "Directory not found, no need to remove: $dir"
    fi
}

# Remove specific directories
safe_remove "$PROJECT_DIR$DATA_DIR"
safe_remove "$PROJECT_DIR$MODELS_DIR"
safe_remove "$PROJECT_DIR$RESULTS_DIR"

# Function to build and run Docker containers for training and inference
run_docker() {
    stage=$1
    image_name=$2
    docker_file_path=$3

    echo "Building Docker image for $stage..."
    docker build -f "$docker_file_path" --build-arg settings_name=settings.json -t "$image_name" .

    echo "Running Docker container for $stage..."
    container_id=$(winpty -Xallow-non-tty -Xplain docker run -itd "$image_name" /bin/bash)

    container_id=$(echo $container_id | cut -c1-12)
    echo "Docker Container ID: $container_id"
    sleep 2 # Wait for the container to fully start

    if [ "$stage" == "training" ]; then
        # Copy trained model from container to host
        docker cp $container_id:$WORK_DIR$MODELS_DIR ./
        docker cp $container_id:$WORK_DIR$DATA_DIR ./
        docker cp $container_id:$WORK_DIR$RESULTS_DIR ./
    elif [ "$stage" == "inference" ]; then
        # Copy inference results from container to host
        docker cp $container_id:$WORK_DIR$RESULTS_DIR ./
    fi

    # Stop and remove container
    docker stop "$container_id"
    docker rm "$container_id"
}

# Function to run scripts locally
run_local() {
    echo "Running data loader locally..."
    pip install -r requirements.txt

    python "$PROJECT_DIR"/data_process/data_loader.py

    echo "Running training locally..."
    # shellcheck disable=SC2086
    python "$PROJECT_DIR"/training/train.py

    echo "Running inference locally..."
    python "$PROJECT_DIR"/inference/run.py
}

# Main execution logic
echo "Starting project workflow..."
echo "Select mode of execution: [1] Docker, [2] Local"

read mode

if [ "$mode" == "1" ]; then
    run_docker "training" "training_sentiment" "$PROJECT_DIR/training/Dockerfile"
    run_docker "inference" "inference_sentiment" "$PROJECT_DIR/inference/Dockerfile"
elif [ "$mode" == "2" ]; then
    run_local
else
    echo "Invalid mode selected."
fi
sleep 100000
echo "Workflow completed."
