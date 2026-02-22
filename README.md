# LoRACat

SDXL LoRA training pipeline for NVIDIA DGX Spark (ARM64 / GB10 Blackwell).

Collects training images from a ComfyUI endpoint, auto-captions them with WD14 tagger, and trains an SDXL LoRA using HuggingFace diffusers.

## Why diffusers instead of Kohya SS / OneTrainer?

Neither Kohya SS nor OneTrainer works on ARM64. Both depend on xformers, which [has no ARM64 support](https://github.com/facebookresearch/xformers/issues/1071). HuggingFace diffusers is the NVIDIA-validated path for LoRA training on DGX Spark, and PyTorch native SDPA with cuDNN 9.13 is faster than xformers on GB10 anyway.

## Prerequisites

- NVIDIA DGX Spark (or any ARM64 system with NVIDIA GPU + CUDA 13+)
- Docker with NVIDIA Container Toolkit (`nvidia-docker`)
- A running ComfyUI instance (for image collection step)
- SDXL Base 1.0 model weights

## Directory Structure

```
loracat/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── workflow_api.json              # Your ComfyUI workflow (API export)
├── face_reference.png             # Face reference image
├── prompts_example.json           # Example batch prompts file
├── config/
│   ├── training_config.toml       # Training hyperparameters
│   └── workflow_config.json       # Node ID mapping for workflow injection
├── scripts/
│   ├── run.sh                     # Pipeline entrypoint
│   ├── collect_images.py          # ComfyUI image collection
│   ├── caption_images.py          # WD14 auto-captioning
│   └── train_lora.py              # LoRA training wrapper
├── dataset/                       # Mounted: training images + captions
│   └── images/
├── output/                        # Mounted: trained LoRA output
│   └── lora/
└── models/                        # Mounted: base SDXL model
    └── stable-diffusion-xl-base-1.0/
```

## Quick Start

### 1. Provide the Base SDXL Model

Download or copy the SDXL Base 1.0 model into the `models/` directory:

```bash
mkdir -p models/stable-diffusion-xl-base-1.0

# Option A: Download from HuggingFace (requires huggingface-cli)
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 \
    --local-dir models/stable-diffusion-xl-base-1.0

# Option B: Copy from an existing install
cp -r /path/to/your/sdxl-base-1.0/* models/stable-diffusion-xl-base-1.0/
```

The directory should contain at minimum: `model_index.json`, `unet/`, `text_encoder/`, `text_encoder_2/`, `vae/`, `tokenizer/`, `tokenizer_2/`, `scheduler/`.

### 2. Export Your ComfyUI Workflow

The collection step uses your actual ComfyUI workflow — not a hardcoded template. This means any workflow (IPAdapter, ControlNet, InstantID, etc.) works out of the box.

1. Build your image generation workflow in ComfyUI
2. Click **Save (API Format)** to export `workflow_api.json`
3. Place it in the project root:

```bash
cp /path/to/workflow_api.json ./workflow_api.json
```

### 3. Configure Node ID Mapping

Edit `config/workflow_config.json` to tell the script which nodes to interact with. The script supports two modes:

**Full-workflow mode** (for workflows with all prompts baked in, like the Qwen→Z-Turbo pipeline):

```json
{
  "image_loader_node": "16",
  "output_prefix": "Qwen_Consistent_Char/Z_IMG_FINAL"
}
```

The script uploads the face image, injects it into the LoadImage node, submits the entire workflow once, and downloads only the outputs from SaveImage nodes whose `filename_prefix` matches `output_prefix`. This is the default configuration.

**Per-batch mode** (for simple single-prompt workflows, submits once per batch entry):

```json
{
  "image_loader_node": "12",
  "positive_prompt_node": "6",
  "seed_node": "3",
  "output_node": "9"
}
```

The mode is auto-detected: if `positive_prompt_node` and `seed_node` are present, per-batch mode is used (requires `prompts.json`). Otherwise full-workflow mode is used.

| Key | Mode | What it targets |
|---|---|---|
| `image_loader_node` | Both | LoadImage node for face reference |
| `output_prefix` | Full-workflow | SaveImage filename_prefix to filter (e.g. `Qwen_Consistent_Char/Z_IMG_FINAL`) |
| `positive_prompt_node` | Per-batch | Prompt text node (triggers per-batch mode) |
| `seed_node` | Per-batch | KSampler seed node |
| `output_node` | Per-batch | Single SaveImage node to download from |

### 4. Provide a Face Reference Image

Place your face reference image in the project root:

```bash
cp /path/to/your/face.png ./face_reference.png
```

The script uploads this to ComfyUI's `/upload/image` endpoint and injects the server-side filename into the workflow's LoadImage node.

### 5. Set Up ComfyUI

Ensure ComfyUI is running and accessible. By default the pipeline connects to `http://host.docker.internal:8188` (the host machine's port 8188 from inside Docker).

If ComfyUI is on a different host:

```bash
export COMFYUI_ENDPOINT=http://192.168.1.100:8188
```

### 6. Build and Run

```bash
# Create output directories
mkdir -p dataset/images output/lora

# Build the container
docker compose build

# Run the full pipeline (collect → caption → train)
docker compose run loracat

# Or run specific steps:
docker compose run loracat --caption --train    # Skip collection, use existing images
docker compose run loracat --train              # Train only (images + captions exist)
docker compose run loracat --collect            # Collect images only

# Override seed for all KSampler nodes:
GLOBAL_SEED=42 docker compose run loracat --collect
```

### 7. Get Your LoRA

After training completes, your LoRA will be in `output/lora/` along with a `training_config_resolved.toml` snapshot of the exact configuration used.

## Configuration

### Environment Variables

All training parameters can be overridden via environment variables, either in `docker-compose.yml` or on the command line:

| Variable | Default | Description |
|---|---|---|
| `COMFYUI_ENDPOINT` | `http://host.docker.internal:8188` | ComfyUI API URL |
| `COMFYUI_WORKFLOW` | `/app/workflow_api.json` | Path to ComfyUI API-format workflow JSON |
| `WORKFLOW_CONFIG` | `/app/config/workflow_config.json` | Path to node ID mapping JSON |
| `FACE_REFERENCE` | _(empty)_ | Path to face reference image |
| `GLOBAL_SEED` | _(empty)_ | Override seed on all KSampler nodes |
| `COMFYUI_DELAY` | `2.0` | Seconds between ComfyUI requests |
| `TRIGGER_WORD` | `nyafyi_woman` | Trigger word prepended to all captions |
| `TRAIN_BATCH_SIZE` | `4` | Training batch size (increase for 128GB memory) |
| `GRADIENT_ACCUMULATION_STEPS` | `1` | Gradient accumulation steps |
| `LEARNING_RATE` | `1e-4` | Learning rate |
| `NUM_TRAIN_EPOCHS` | `20` | Number of training epochs |
| `LORA_RANK` | `32` | LoRA network rank |
| `RESOLUTION` | `1024` | Training resolution |
| `MIXED_PRECISION` | `bf16` | Mixed precision mode (bf16 recommended for Blackwell) |
| `CAPTION_BATCH_SIZE` | `8` | Batch size for WD14 captioning |
| `DATALOADER_NUM_WORKERS` | `4` | Data loader worker threads |
| `WANDB_API_KEY` | _(empty)_ | Weights & Biases API key for logging |

Example with overrides:

```bash
TRAIN_BATCH_SIZE=8 LEARNING_RATE=5e-5 docker compose run loracat --train
```

### Training Config (TOML)

For full control, edit `config/training_config.toml`. This file is the source of truth for all training parameters and is saved alongside the trained LoRA for reproducibility. Environment variables override values from the TOML file.

### ComfyUI Workflow Integration

The collection step uses your real ComfyUI workflow (exported as API JSON). Two modes are supported:

**Full-workflow mode** (default) — for complex multi-prompt workflows like Qwen→Z-Turbo:
1. Uploads `face_reference.png` to ComfyUI via `/upload/image`
2. Injects the uploaded filename into the LoadImage node
3. Submits the entire workflow once (all prompts run in a single execution)
4. Scans the workflow for SaveImage nodes matching `output_prefix`
5. Downloads only matching outputs (e.g., only `Z_IMG_FINAL`, not intermediate `CC` images)

**Per-batch mode** — for simple single-prompt template workflows:
1. Uploads face image
2. For each entry in `prompts.json`: injects prompt text, seed, and face image
3. Submits once per entry, downloads outputs

The mode is auto-detected from `workflow_config.json`. If `positive_prompt_node` and `seed_node` are present, per-batch mode is used. Otherwise full-workflow mode is used.

Any ComfyUI workflow works — Qwen Image Edit, IPAdapter, ControlNet, InstantID, etc. Just export it and configure the node IDs.

## Providing Your Own Images

If you already have training images, skip the collection step:

1. Place your images in `dataset/images/`
2. Run captioning + training only:

```bash
docker compose run loracat --caption --train
```

Or if you also have your own captions (`.txt` files alongside each image):

```bash
docker compose run loracat --train
```

## DGX Spark Notes

- **Unified memory**: The DGX Spark shares 128GB between CPU and GPU. The default batch size of 4 is conservative — you can safely increase to 8 or higher.
- **No swap death spiral**: If memory is exhausted, the unified memory architecture can freeze the machine instead of OOM-killing. Monitor memory usage and start with conservative settings.
- **bf16 precision**: Blackwell GPUs have excellent bf16 throughput. The default `bf16` mixed precision is optimal.
- **No xformers**: xformers has no ARM64 support. PyTorch native SDPA with cuDNN 9.13 is used instead and is actually faster on GB10.
- **Container**: Built on `nvcr.io/nvidia/pytorch:26.01-py3` which includes CUDA 13.1 and Blackwell (sm_121) support.

## Troubleshooting

**ComfyUI connection refused**: Ensure ComfyUI is running and the `COMFYUI_ENDPOINT` is reachable from inside the container. The default uses `host.docker.internal` which resolves to the host machine.

**Model not found**: Ensure the SDXL model is in `models/stable-diffusion-xl-base-1.0/` with the full diffusers directory structure (not just a single `.safetensors` file).

**Out of memory**: Reduce `TRAIN_BATCH_SIZE` or enable `gradient_checkpointing` (enabled by default in the TOML config).

**WD14 tagger slow**: The tagger runs on CPU (onnxruntime-gpu lacks ARM64 wheels). This is a one-time preprocessing step and should complete in a few minutes even for large datasets.
