# LoRACat — Z-Image LoRA Training on NVIDIA DGX Spark (ARM64 / GB10 Blackwell)
#
# Uses kohya-ss/musubi-tuner for Z-Image LoRA training.
# Base image provides PyTorch + CUDA. Musubi-tuner is installed from source.

FROM nvcr.io/nvidia/pytorch:26.01-py3

LABEL maintainer="burritocatai"
LABEL description="Z-Image LoRA training pipeline for NVIDIA DGX Spark"

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone and install musubi-tuner (brings its own diffusers/accelerate/transformers/peft)
RUN git clone https://github.com/kohya-ss/musubi-tuner.git /app/musubi-tuner && \
    cd /app/musubi-tuner && pip install --no-cache-dir -e .

# Copy and install our own dependencies (collect, caption scripts)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY scripts/ /app/scripts/
COPY config/ /app/config/
COPY prompts.json /app/prompts.json

# Make scripts executable
RUN chmod +x /app/scripts/run.sh

# Create default directories
RUN mkdir -p /dataset/images /dataset/cache /output/lora /models

# Default environment variables — generous defaults for 128GB unified memory
ENV COMFYUI_ENDPOINT="http://host.docker.internal:8188"
ENV COMFYUI_WORKFLOW="/app/workflow_api.json"
ENV WORKFLOW_CONFIG="/app/config/workflow_config.json"
ENV FACE_REFERENCE=""
ENV PROMPTS_FILE="/app/prompts.json"
ENV GLOBAL_SEED=""
ENV DATASET_DIR="/dataset/images"
ENV CACHE_DIR="/dataset/cache"
ENV OUTPUT_DIR="/output/lora"
ENV DIT_PATH="/models/z_image/dit"
ENV VAE_PATH="/models/z_image/vae"
ENV TEXT_ENCODER_PATH="/models/z_image/text_encoder"
ENV TRAINING_CONFIG="/app/config/training_config.toml"
ENV DATASET_CONFIG="/app/config/dataset_config.toml"
ENV TRIGGER_WORD="nyafyi_woman"

# Training defaults (configurable via env vars)
ENV TRAIN_BATCH_SIZE="4"
ENV GRADIENT_ACCUMULATION_STEPS="1"
ENV LEARNING_RATE="1e-4"
ENV NUM_TRAIN_EPOCHS="16"
ENV NETWORK_DIM="32"
ENV RESOLUTION="1024"
ENV CAPTION_BATCH_SIZE="8"
ENV COMFYUI_DELAY="2.0"
ENV MIXED_PRECISION="bf16"
ENV DATALOADER_NUM_WORKERS="4"
ENV DISCRETE_FLOW_SHIFT="2.0"
ENV OUTPUT_NAME="lora_output"

ENTRYPOINT ["/app/scripts/run.sh"]
