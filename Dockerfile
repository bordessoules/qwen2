FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# System dependencies including CUDA tools
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set CUDA_HOME environment variable
ENV CUDA_HOME=/usr/local/cuda

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install flash-attn --no-build-isolation

# Copy application code
COPY . .

# Set default command
ENTRYPOINT ["python", "qwen2vl_cli.py"]
