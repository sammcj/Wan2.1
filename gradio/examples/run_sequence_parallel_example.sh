#!/usr/bin/env bash
# Example script to run Wan2.1 text-to-video generation with sequence parallelism

# Check if checkpoint directory is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <checkpoint_directory> [num_gpus]"
  echo "Example: $0 ./cache 4"
  exit 1
fi

CKPT_DIR=$1
NUM_GPUS=${2:-4} # Default to 4 GPUs if not specified

# Calculate ulysses_size and ring_size
# For this example, we'll use a simple approach:
# If NUM_GPUS is even, set ulysses_size to NUM_GPUS/2 and ring_size to 2
# If NUM_GPUS is odd, set ulysses_size to NUM_GPUS and ring_size to 1
if [ $((NUM_GPUS % 2)) -eq 0 ]; then
  ULYSSES_SIZE=$((NUM_GPUS / 2))
  RING_SIZE=2
else
  ULYSSES_SIZE=$NUM_GPUS
  RING_SIZE=1
fi

# Create directory if it doesn't exist
mkdir -p $(dirname "$0")

echo "Starting Wan2.1 text-to-video generation with $NUM_GPUS GPUs"
echo "Using checkpoint directory: $CKPT_DIR"
echo "Sequence parallelism configuration: ulysses_size=$ULYSSES_SIZE, ring_size=$RING_SIZE"

# First run the test script to verify the setup
echo "Testing multi-GPU setup with sequence parallelism..."
torchrun --nproc_per_node=$NUM_GPUS ../test_multiGPU.py --ckpt_dir $CKPT_DIR \
  --ulysses_size $ULYSSES_SIZE --ring_size $RING_SIZE

# If the test was successful, run the Gradio interface
if [ $? -eq 0 ]; then
  echo "Test successful! Starting Gradio interface..."
  torchrun --nproc_per_node=$NUM_GPUS ../t2v_14B_multiGPU.py --ckpt_dir $CKPT_DIR \
    --ulysses_size $ULYSSES_SIZE --ring_size $RING_SIZE
else
  echo "Test failed. Please check the error messages above."
  exit 1
fi
