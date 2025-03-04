#!/usr/bin/env bash
# Example script to run Wan2.1 text-to-video generation on multiple GPUs

# Check if checkpoint directory is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <checkpoint_directory> [num_gpus]"
  echo "Example: $0 ./cache 2"
  exit 1
fi

CKPT_DIR=$1
NUM_GPUS=${2:-2}  # Default to 2 GPUs if not specified

# Create directory if it doesn't exist
mkdir -p $(dirname "$0")

echo "Starting Wan2.1 text-to-video generation with $NUM_GPUS GPUs"
echo "Using checkpoint directory: $CKPT_DIR"

# First run the test script to verify the setup
echo "Testing multi-GPU setup..."
torchrun --nproc_per_node=$NUM_GPUS ../test_multiGPU.py --ckpt_dir $CKPT_DIR --t5_fsdp --dit_fsdp

# If the test was successful, run the Gradio interface
if [ $? -eq 0 ]; then
  echo "Test successful! Starting Gradio interface..."
  torchrun --nproc_per_node=$NUM_GPUS ../t2v_14B_multiGPU.py --ckpt_dir $CKPT_DIR --t5_fsdp --dit_fsdp
else
  echo "Test failed. Please check the error messages above."
  exit 1
fi
