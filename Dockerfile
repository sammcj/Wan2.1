FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  git \
  wget \
  ffmpeg \
  libsm6 \
  libxext6 \
  libgl1-mesa-glx \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for multi-GPU support
RUN pip install --no-cache-dir xfuser

# Copy project files
COPY . .

# Make example scripts executable
RUN chmod +x gradio/examples/run_multiGPU_example.sh gradio/examples/run_sequence_parallel_example.sh

# Set environment variables
ENV PYTHONPATH=/app

# Default command - can be overridden in docker-compose.yml
CMD ["bash"]
