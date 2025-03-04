# Wan2.1 Multi-GPU Text-to-Video Generation

This document explains how to use the multi-GPU version of the Wan2.1 text-to-video generation model.

## Files

- `t2v_14B_multiGPU.py`: Main script for running the Gradio interface with multi-GPU support
- `test_multiGPU.py`: Test script to verify your multi-GPU setup is working correctly
- `examples/run_multiGPU_example.sh`: Example script for running with FSDP parallelism
- `examples/run_sequence_parallel_example.sh`: Example script for running with sequence parallelism

## Overview

The `t2v_14B_multiGPU.py` script enables distributed inference using multiple GPUs for the Wan2.1 text-to-video generation model. This allows for faster generation and the ability to handle larger models that may not fit on a single GPU.

## Requirements

- Multiple NVIDIA GPUs
- PyTorch with CUDA support
- torchrun (comes with PyTorch)

## Usage

To run the multi-GPU version, use `torchrun` to launch the script across multiple GPUs:

```bash
torchrun --nproc_per_node=NUM_GPUS gradio/t2v_14B_multiGPU.py --ckpt_dir PATH_TO_CHECKPOINTS [options]
```

Where:
- `NUM_GPUS` is the number of GPUs you want to use
- `PATH_TO_CHECKPOINTS` is the path to the directory containing the model checkpoints

### Distributed Training Options

The script supports two types of parallelism:

1. **Fully Sharded Data Parallelism (FSDP)**: Shards model parameters across GPUs
   - `--t5_fsdp`: Enable FSDP for the T5 text encoder
   - `--dit_fsdp`: Enable FSDP for the DiT model

2. **Sequence Parallelism with xfuser**:
   - `--ulysses_size`: The size of the ulysses parallelism in DiT
   - `--ring_size`: The size of the ring attention parallelism in DiT

Note: When using sequence parallelism, the product of `ulysses_size` and `ring_size` should equal the total number of GPUs.

### Other Options

- `--prompt_extend_method`: Method for prompt extension ("dashscope" or "local_qwen")
- `--prompt_extend_model`: Model to use for prompt extension
- `--t5_cpu`: Place T5 model on CPU to save GPU memory
- `--server_port`: Port for the Gradio server (default: 7860)

## Examples

### Using 2 GPUs with FSDP

```bash
torchrun --nproc_per_node=2 gradio/t2v_14B_multiGPU.py --ckpt_dir ./cache --t5_fsdp --dit_fsdp
```

### Using 4 GPUs with Sequence Parallelism

```bash
torchrun --nproc_per_node=4 gradio/t2v_14B_multiGPU.py --ckpt_dir ./cache --ulysses_size=2 --ring_size=2
```

### Using 8 GPUs with Sequence Parallelism

```bash
torchrun --nproc_per_node=8 gradio/t2v_14B_multiGPU.py --ckpt_dir ./cache --ulysses_size=4 --ring_size=2
```

### Using Example Scripts

For convenience, example scripts are provided in the `examples` directory:

```bash
# Run with FSDP parallelism (default: 2 GPUs)
./gradio/examples/run_multiGPU_example.sh ./cache [num_gpus]

# Run with sequence parallelism (default: 4 GPUs)
./gradio/examples/run_sequence_parallel_example.sh ./cache [num_gpus]
```

These scripts will first run the test script to verify your setup, and if successful, launch the Gradio interface.

## Notes

- Only rank 0 serves the Gradio interface, while other ranks wait for requests
- For best performance, set `ulysses_size` and `ring_size` based on your model architecture
- The `offload_model` parameter is automatically set to `False` in multi-GPU mode to prevent issues with model offloading

## Testing Your Setup

Before running the full Gradio interface, you can test your multi-GPU setup using the provided test script:

```bash
torchrun --nproc_per_node=NUM_GPUS gradio/test_multiGPU.py --ckpt_dir PATH_TO_CHECKPOINTS [options]
```

This script will:
1. Initialize the distributed environment
2. Test GPU memory availability
3. Load the model configuration
4. Initialize the WanT2V model
5. Test synchronization between processes
6. Report model parameters

If the test completes successfully, you can proceed to run the full Gradio interface with the same configuration.
