#!/usr/bin/env bash
# Script to run Wan2.1 Multi-GPU with Docker using pre-downloaded model files

# Check if model directory is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <model_directory> [num_gpus]"
  echo "Example: $0 /mnt/llm/models/Wan-AI_Wan2.1-T2V-14B 2"
  exit 1
fi

MODEL_DIR=$1
NUM_GPUS=${2:-2} # Default to 2 GPUs if not specified

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
  echo "Error: Model directory '$MODEL_DIR' does not exist."
  exit 1
fi

echo "Starting Wan2.1 Multi-GPU with Docker..."
echo "Using model directory: $MODEL_DIR"
echo "Number of GPUs: $NUM_GPUS"

# Run Docker container
MODEL_DIR="$MODEL_DIR" \
  NUM_GPUS="$NUM_GPUS" \
  docker-compose up

echo "Docker container stopped."
