#!/usr/bin/env bash
# Entrypoint script for Wan2.1 Multi-GPU Docker container

# Use MODEL_DIR if provided, otherwise use CHECKPOINT_DIR
CKPT_DIR="/app/cache"
if [ -n "${MODEL_DIR}" ] && [ -d "/app/models" ]; then
  CKPT_DIR="/app/models"
  echo "Using mounted model directory: ${CKPT_DIR}"
fi

# Set default values for environment variables
NUM_GPUS=${NUM_GPUS:-2}
PARALLEL_STRATEGY=${PARALLEL_STRATEGY:-fsdp}
ULYSSES_SIZE=${ULYSSES_SIZE:-2}
RING_SIZE=${RING_SIZE:-1}
SERVER_PORT=${SERVER_PORT:-7860}

echo "Starting Wan2.1 Multi-GPU with the following configuration:"
echo "  Checkpoint directory: ${CKPT_DIR}"
echo "  Number of GPUs: ${NUM_GPUS}"
echo "  Parallel strategy: ${PARALLEL_STRATEGY}"
if [ "${PARALLEL_STRATEGY}" = "sequence" ]; then
  echo "  Ulysses size: ${ULYSSES_SIZE}"
  echo "  Ring size: ${RING_SIZE}"
fi
echo "  Server port: ${SERVER_PORT}"
echo ""

# Run the appropriate command based on the parallel strategy
if [ "${PARALLEL_STRATEGY}" = "fsdp" ]; then
  exec torchrun --nproc_per_node=${NUM_GPUS} gradio/t2v_14B_multiGPU.py --ckpt_dir ${CKPT_DIR} --t5_fsdp --dit_fsdp --server_port ${SERVER_PORT}
elif [ "${PARALLEL_STRATEGY}" = "sequence" ]; then
  exec torchrun --nproc_per_node=${NUM_GPUS} gradio/t2v_14B_multiGPU.py --ckpt_dir ${CKPT_DIR} --ulysses_size ${ULYSSES_SIZE} --ring_size ${RING_SIZE} --server_port ${SERVER_PORT}
else
  echo "Invalid PARALLEL_STRATEGY: ${PARALLEL_STRATEGY}. Must be 'fsdp' or 'sequence'."
  exit 1
fi
