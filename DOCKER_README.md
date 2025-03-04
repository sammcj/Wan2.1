# Running Wan2.1 Multi-GPU with Docker

This guide explains how to run the Wan2.1 text-to-video generation model with multiple GPUs using Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Multiple NVIDIA GPUs with CUDA support

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/alibaba/Wan2.1.git
   cd Wan2.1
   ```

2. Download the model checkpoints to a local directory (e.g., `./cache`).

3. Run the Docker container using the provided script:
   ```bash
   ./run_docker.sh -c /path/to/checkpoints
   ```

   Or manually with environment variables:
   ```bash
   CHECKPOINT_DIR=/path/to/checkpoints docker-compose up
   ```

4. Access the Gradio web interface at http://localhost:7860

## Using the run_docker.sh Script

We provide a convenient script to run the Docker container with different configurations:

```bash
./run_docker.sh -c /path/to/checkpoints -g 4 -p sequence -u 2 -r 2
```

Options:
- `-c, --checkpoint-dir DIR`: Path to checkpoint directory (default: ./cache)
- `-o, --output-dir DIR`: Path to output directory (default: ./output)
- `-g, --gpus NUM`: Number of GPUs to use (default: 2)
- `-p, --parallel STRATEGY`: Parallelism strategy: fsdp or sequence (default: fsdp)
- `-u, --ulysses-size NUM`: Size of ulysses parallelism (default: 2)
- `-r, --ring-size NUM`: Size of ring attention parallelism (default: 1)
- `-s, --server-port PORT`: Port for Gradio server (default: 7860)
- `-b, --build`: Build Docker image before running
- `-h, --help`: Show help message

## Configuration Options

You can customize the Docker setup using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `CHECKPOINT_DIR` | Path to the model checkpoints directory | `./cache` |
| `OUTPUT_DIR` | Path to save generated videos | `./output` |
| `NUM_GPUS` | Number of GPUs to use | `2` |
| `PARALLEL_STRATEGY` | Parallelism strategy (`fsdp` or `sequence`) | `fsdp` |
| `ULYSSES_SIZE` | Size of ulysses parallelism (for sequence strategy) | `2` |
| `RING_SIZE` | Size of ring attention parallelism (for sequence strategy) | `1` |
| `SERVER_PORT` | Port for the Gradio server | `7860` |

## Examples

### Using 2 GPUs with FSDP Parallelism

```bash
CHECKPOINT_DIR=/path/to/checkpoints \
NUM_GPUS=2 \
PARALLEL_STRATEGY=fsdp \
docker-compose up
```

### Using 4 GPUs with Sequence Parallelism

```bash
CHECKPOINT_DIR=/path/to/checkpoints \
NUM_GPUS=4 \
PARALLEL_STRATEGY=sequence \
ULYSSES_SIZE=2 \
RING_SIZE=2 \
docker-compose up
```

### Using 8 GPUs with Sequence Parallelism

```bash
CHECKPOINT_DIR=/path/to/checkpoints \
NUM_GPUS=8 \
PARALLEL_STRATEGY=sequence \
ULYSSES_SIZE=4 \
RING_SIZE=2 \
docker-compose up
```

## Building the Docker Image

If you need to modify the Dockerfile, you can rebuild the image:

```bash
docker-compose build
```

## Troubleshooting

### GPU Access Issues

If you encounter issues with GPU access, ensure the NVIDIA Container Toolkit is properly installed and configured:

```bash
# Verify NVIDIA Container Toolkit installation
nvidia-container-cli info

# Check if Docker can access GPUs
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Memory Issues

If you encounter out-of-memory errors:

1. Try using more GPUs to distribute the model
2. Enable T5 CPU offloading by modifying the command in docker-compose.yml to include `--t5_cpu`
3. Adjust the batch size or resolution in the Gradio interface

## Notes

- The Docker setup uses the `pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime` base image, which includes PyTorch with CUDA support.
- The container mounts your local checkpoint directory to `/app/cache` inside the container.
- Generated videos are saved to the mounted output directory.
