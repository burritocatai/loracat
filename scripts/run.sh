#!/usr/bin/env bash
#
# run.sh — LoRACat pipeline entrypoint
#
# Runs the full LoRA training pipeline in order:
#   1. Collect images from ComfyUI
#   2. Auto-caption images with WD14 tagger
#   3. Train SDXL LoRA
#
# Each step logs with timestamps and fails loudly on error.
# Steps can be selected individually via flags.

set -euo pipefail

# --- Logging helpers ---
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

log_step() {
    echo ""
    echo "================================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STEP: $*"
    echo "================================================================"
}

# --- Parse arguments ---
RUN_COLLECT=false
RUN_CAPTION=false
RUN_TRAIN=false
PARSED_ANY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --collect)
            RUN_COLLECT=true
            PARSED_ANY=true
            shift
            ;;
        --caption)
            RUN_CAPTION=true
            PARSED_ANY=true
            shift
            ;;
        --train)
            RUN_TRAIN=true
            PARSED_ANY=true
            shift
            ;;
        --all)
            RUN_COLLECT=true
            RUN_CAPTION=true
            RUN_TRAIN=true
            PARSED_ANY=true
            shift
            ;;
        *)
            log_error "Unknown argument: $1"
            echo "Usage: run.sh [--collect] [--caption] [--train] [--all]"
            echo "  --collect  Collect images from ComfyUI"
            echo "  --caption  Auto-caption images with WD14 tagger"
            echo "  --train    Train SDXL LoRA"
            echo "  --all      Run all steps (default if no flags given)"
            exit 1
            ;;
    esac
done

# Default: run all steps if no flags given
if [ "$PARSED_ANY" = false ]; then
    RUN_COLLECT=true
    RUN_CAPTION=true
    RUN_TRAIN=true
fi

# --- Environment summary ---
log "LoRACat Pipeline Starting"
log "Configuration:"
log "  COMFYUI_ENDPOINT:    ${COMFYUI_ENDPOINT:-http://localhost:8188}"
log "  COMFYUI_WORKFLOW:    ${COMFYUI_WORKFLOW:-/app/workflow_api.json}"
log "  WORKFLOW_CONFIG:     ${WORKFLOW_CONFIG:-/app/config/workflow_config.json}"
log "  FACE_REFERENCE:      ${FACE_REFERENCE:-}"
log "  PROMPTS_FILE:        ${PROMPTS_FILE:-/app/prompts.json}"
log "  DATASET_DIR:         ${DATASET_DIR:-/dataset/images}"
log "  OUTPUT_DIR:          ${OUTPUT_DIR:-/output/lora}"
log "  MODEL_PATH:          ${MODEL_PATH:-/models/stable-diffusion-xl-base-1.0}"
log "  TRIGGER_WORD:        ${TRIGGER_WORD:-nyafyi_woman}"
log "  TRAIN_BATCH_SIZE:    ${TRAIN_BATCH_SIZE:-4}"
log "  LEARNING_RATE:       ${LEARNING_RATE:-1e-4}"
log "  NUM_TRAIN_EPOCHS:    ${NUM_TRAIN_EPOCHS:-20}"
log "  LORA_RANK:           ${LORA_RANK:-32}"
log "  RESOLUTION:          ${RESOLUTION:-1024}"
log "  MIXED_PRECISION:     ${MIXED_PRECISION:-bf16}"
log "Steps: collect=${RUN_COLLECT} caption=${RUN_CAPTION} train=${RUN_TRAIN}"

# --- GPU check ---
log "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
    log_error "nvidia-smi not found — GPU may not be available"
fi

# --- Step 1: Collect images ---
if [ "$RUN_COLLECT" = true ]; then
    log_step "1/3 — Collecting images from ComfyUI"

    WORKFLOW_FILE="${COMFYUI_WORKFLOW:-/app/workflow_api.json}"
    if [ ! -f "$WORKFLOW_FILE" ]; then
        log_error "Workflow file not found: $WORKFLOW_FILE"
        log_error "Export your ComfyUI workflow as API JSON and mount it, or set COMFYUI_WORKFLOW"
        exit 1
    fi

    CONFIG_FILE="${WORKFLOW_CONFIG:-/app/config/workflow_config.json}"
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Workflow config not found: $CONFIG_FILE"
        log_error "Create a workflow_config.json with node IDs, or set WORKFLOW_CONFIG"
        exit 1
    fi

    COLLECT_ARGS=(
        --endpoint "${COMFYUI_ENDPOINT:-http://localhost:8188}"
        --output-dir "${DATASET_DIR:-/dataset/images}"
        --workflow "$WORKFLOW_FILE"
        --workflow-config "$CONFIG_FILE"
        --delay "${COMFYUI_DELAY:-2.0}"
    )

    # Prompts file is only required for per-batch mode
    if [ -f "${PROMPTS_FILE:-/app/prompts.json}" ]; then
        COLLECT_ARGS+=(--prompts "${PROMPTS_FILE:-/app/prompts.json}")
    fi

    if [ -n "${FACE_REFERENCE:-}" ] && [ -f "${FACE_REFERENCE}" ]; then
        log "Face reference image: ${FACE_REFERENCE}"
        COLLECT_ARGS+=(--face-image "${FACE_REFERENCE}")
    fi

    if [ -n "${GLOBAL_SEED:-}" ]; then
        COLLECT_ARGS+=(--seed "${GLOBAL_SEED}")
    fi

    python /app/scripts/collect_images.py "${COLLECT_ARGS[@]}"

    log "Image collection complete"
else
    log "Skipping image collection (--collect not set)"
fi

# --- Step 2: Auto-caption ---
if [ "$RUN_CAPTION" = true ]; then
    log_step "2/3 — Auto-captioning images with WD14 tagger"

    IMAGE_COUNT=$(find "${DATASET_DIR:-/dataset/images}" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.webp" \) 2>/dev/null | wc -l)
    if [ "$IMAGE_COUNT" -eq 0 ]; then
        log_error "No images found in ${DATASET_DIR:-/dataset/images}"
        log_error "Run the collection step first or place images in the dataset directory"
        exit 1
    fi
    log "Found ${IMAGE_COUNT} images to caption"

    python /app/scripts/caption_images.py \
        --image-dir "${DATASET_DIR:-/dataset/images}" \
        --trigger-word "${TRIGGER_WORD:-nyafyi_woman}" \
        --batch-size "${CAPTION_BATCH_SIZE:-8}"

    log "Auto-captioning complete"
else
    log "Skipping auto-captioning (--caption not set)"
fi

# --- Step 3: Train LoRA ---
if [ "$RUN_TRAIN" = true ]; then
    log_step "3/3 — Training SDXL LoRA"

    python /app/scripts/train_lora.py \
        --config "${TRAINING_CONFIG:-/app/config/training_config.toml}"

    log "LoRA training complete"
    log "Output saved to: ${OUTPUT_DIR:-/output/lora}"
else
    log "Skipping LoRA training (--train not set)"
fi

# --- Done ---
echo ""
log "================================================================"
log "LoRACat pipeline finished successfully!"
log "================================================================"
