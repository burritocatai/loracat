# LoRACat - SDXL LoRA Training on NVIDIA DGX Spark (ARM64 / GB10 Blackwell)
#
# Uses HuggingFace diffusers instead of Kohya SS / OneTrainer because
# neither has ARM64 support (xformers dependency is x86_64-only).
# NVIDIA officially validates diffusers on DGX Spark.

FROM nvcr.io/nvidia/pytorch:26.01-py3

LABEL maintainer="burritocatai"
LABEL description="SDXL LoRA training pipeline for NVIDIA DGX Spark"

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

# Copy requirements first for layer caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
# Note: PyTorch is already installed in the NGC container
# Note: xformers is intentionally excluded — no ARM64 support.
# PyTorch native SDPA with cuDNN 9.13 is faster on GB10 anyway.
RUN pip install --no-cache-dir -r requirements.txt

# Download the diffusers SDXL LoRA training script
RUN mkdir -p /app/diffusers_scripts && \
    wget -q -O /app/diffusers_scripts/train_dreambooth_lora_sdxl.py \
    "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora_sdxl.py"

# Copy application code
COPY scripts/ /app/scripts/
COPY config/ /app/config/

# Make scripts executable
RUN chmod +x /app/scripts/run.sh

# Create default directories
RUN mkdir -p /dataset/images /output/lora /models

# Default environment variables — generous defaults for 128GB unified memory
ENV COMFYUI_ENDPOINT="http://host.docker.internal:8188"
ENV COMFYUI_WORKFLOW="/app/workflow_api.json"
ENV WORKFLOW_CONFIG="/app/config/workflow_config.json"
ENV FACE_REFERENCE=""
ENV PROMPTS_FILE=""
ENV GLOBAL_SEED=""
ENV DATASET_DIR="/dataset/images"
ENV OUTPUT_DIR="/output/lora"
ENV MODEL_PATH="/models/stable-diffusion-xl-base-1.0"
ENV TRAINING_CONFIG="/app/config/training_config.toml"
ENV TRIGGER_WORD="nyafyi_woman"

# Training defaults (configurable via env vars)
ENV TRAIN_BATCH_SIZE="4"
ENV GRADIENT_ACCUMULATION_STEPS="1"
ENV LEARNING_RATE="1e-4"
ENV NUM_TRAIN_EPOCHS="20"
ENV LORA_RANK="32"
ENV RESOLUTION="1024"
ENV CAPTION_BATCH_SIZE="8"
ENV COMFYUI_DELAY="2.0"
ENV MIXED_PRECISION="bf16"
ENV DATALOADER_NUM_WORKERS="4"

ENTRYPOINT ["/app/scripts/run.sh"]
