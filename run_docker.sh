#!/usr/bin/env bash
# Script to run Wan2.1 Multi-GPU with Docker

# Default values
CHECKPOINT_DIR="./cache"
OUTPUT_DIR="./output"
MODEL_DIR=""
NUM_GPUS=2
PARALLEL_STRATEGY="fsdp"
ULYSSES_SIZE=2
RING_SIZE=1
SERVER_PORT=7860
BUILD=false

# Help function
show_help() {
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -c, --checkpoint-dir DIR   Path to checkpoint directory (default: ./cache)"
  echo "  -m, --model-dir DIR        Path to specific model directory (takes precedence over checkpoint-dir)"
  echo "  -o, --output-dir DIR       Path to output directory (default: ./output)"
  echo "  -g, --gpus NUM             Number of GPUs to use (default: 2)"
  echo "  -p, --parallel STRATEGY    Parallelism strategy: fsdp or sequence (default: fsdp)"
  echo "  -u, --ulysses-size NUM     Size of ulysses parallelism (default: 2)"
  echo "  -r, --ring-size NUM        Size of ring attention parallelism (default: 1)"
  echo "  -s, --server-port PORT     Port for Gradio server (default: 7860)"
  echo "  -b, --build                Build Docker image before running"
  echo "  -h, --help                 Show this help message"
  echo ""
  echo "Example:"
  echo "  $0 -c /path/to/checkpoints -g 4 -p sequence -u 2 -r 2"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
  -c | --checkpoint-dir)
    CHECKPOINT_DIR="$2"
    shift 2
    ;;
  -m | --model-dir)
    MODEL_DIR="$2"
    shift 2
    ;;
  -o | --output-dir)
    OUTPUT_DIR="$2"
    shift 2
    ;;
  -g | --gpus)
    NUM_GPUS="$2"
    shift 2
    ;;
  -p | --parallel)
    PARALLEL_STRATEGY="$2"
    shift 2
    ;;
  -u | --ulysses-size)
    ULYSSES_SIZE="$2"
    shift 2
    ;;
  -r | --ring-size)
    RING_SIZE="$2"
    shift 2
    ;;
  -s | --server-port)
    SERVER_PORT="$2"
    shift 2
    ;;
  -b | --build)
    BUILD=true
    shift
    ;;
  -h | --help)
    show_help
    exit 0
    ;;
  *)
    echo "Unknown option: $1"
    show_help
    exit 1
    ;;
  esac
done

# Validate arguments
if [ -n "$MODEL_DIR" ]; then
  if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Model directory '$MODEL_DIR' does not exist."
    exit 1
  fi
elif [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "Error: Checkpoint directory '$CHECKPOINT_DIR' does not exist."
  exit 1
fi

if [ "$PARALLEL_STRATEGY" != "fsdp" ] && [ "$PARALLEL_STRATEGY" != "sequence" ]; then
  echo "Error: Parallel strategy must be 'fsdp' or 'sequence'."
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Build Docker image if requested
if [ "$BUILD" = true ]; then
  echo "Building Docker image..."
  docker-compose build
fi

# Run Docker container
echo "Starting Wan2.1 Multi-GPU with Docker..."
if [ -n "$MODEL_DIR" ]; then
  echo "  Model directory: $MODEL_DIR"
else
  echo "  Checkpoint directory: $CHECKPOINT_DIR"
fi
echo "  Output directory: $OUTPUT_DIR"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Parallel strategy: $PARALLEL_STRATEGY"
if [ "$PARALLEL_STRATEGY" = "sequence" ]; then
  echo "  Ulysses size: $ULYSSES_SIZE"
  echo "  Ring size: $RING_SIZE"
fi
echo "  Server port: $SERVER_PORT"
echo ""

CHECKPOINT_DIR="$CHECKPOINT_DIR" \
  MODEL_DIR="$MODEL_DIR" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  NUM_GPUS="$NUM_GPUS" \
  PARALLEL_STRATEGY="$PARALLEL_STRATEGY" \
  ULYSSES_SIZE="$ULYSSES_SIZE" \
  RING_SIZE="$RING_SIZE" \
  SERVER_PORT="$SERVER_PORT" \
  docker-compose up

echo "Docker container stopped."
