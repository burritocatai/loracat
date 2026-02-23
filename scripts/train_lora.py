#!/usr/bin/env python3
"""
train_lora.py — SDXL LoRA training wrapper using HuggingFace diffusers.

Reads a .toml configuration file (for reproducibility), merges in environment
variable overrides, and launches the diffusers dreambooth LoRA training script.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import toml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

TRAINING_SCRIPT = "/app/diffusers_scripts/train_dreambooth_lora_sdxl.py"


def load_config(config_path: str) -> dict:
    """Load training configuration from a .toml file."""
    log.info("Loading training config from %s", config_path)
    with open(config_path) as f:
        config = toml.load(f)
    return config


def resolve_config(config: dict) -> dict:
    """Merge .toml config with environment variable overrides.

    Environment variables take precedence over .toml values.
    """
    resolved = dict(config)

    env_overrides = {
        "pretrained_model_name_or_path": "MODEL_PATH",
        "output_dir": "OUTPUT_DIR",
        "instance_data_dir": "DATASET_DIR",
        "train_batch_size": "TRAIN_BATCH_SIZE",
        "gradient_accumulation_steps": "GRADIENT_ACCUMULATION_STEPS",
        "learning_rate": "LEARNING_RATE",
        "num_train_epochs": "NUM_TRAIN_EPOCHS",
        "rank": "LORA_RANK",
        "resolution": "RESOLUTION",
        "mixed_precision": "MIXED_PRECISION",
        "dataloader_num_workers": "DATALOADER_NUM_WORKERS",
    }

    for param, env_var in env_overrides.items():
        val = os.environ.get(env_var)
        if val is not None:
            # Convert numeric types
            if param in (
                "train_batch_size",
                "gradient_accumulation_steps",
                "num_train_epochs",
                "rank",
                "resolution",
                "dataloader_num_workers",
            ):
                resolved[param] = int(val)
            elif param in ("learning_rate",):
                resolved[param] = float(val)
            else:
                resolved[param] = val

    # TRIGGER_WORD drives instance_prompt and validation_prompt so the
    # whole pipeline stays consistent regardless of what's in the .toml.
    trigger = os.environ.get("TRIGGER_WORD")
    if trigger:
        resolved["instance_prompt"] = f"a photo of {trigger}"
        if resolved.get("validation_prompt"):
            resolved["validation_prompt"] = f"a photo of {trigger} in a garden"
        log.info("Trigger word '%s' → instance_prompt='%s'", trigger, resolved["instance_prompt"])

    return resolved


def _bnb_available() -> bool:
    """Check if bitsandbytes CUDA binary exists for the current CUDA version."""
    try:
        import bitsandbytes
        import torch
        cuda_ver = torch.version.cuda
        if not cuda_ver:
            return False
        tag = cuda_ver.replace(".", "")
        so_path = Path(bitsandbytes.__file__).parent / f"libbitsandbytes_cuda{tag}.so"
        return so_path.exists()
    except Exception:
        return False


def build_command(config: dict) -> list[str]:
    """Build the accelerate launch command from resolved config."""
    cmd = [
        "accelerate",
        "launch",
        "--mixed_precision", config.get("mixed_precision", "bf16"),
        TRAINING_SCRIPT,
    ]

    # Disable 8-bit adam if bitsandbytes native library isn't available (e.g. ARM64)
    if config.get("use_8bit_adam") and not _bnb_available():
        log.warning(
            "bitsandbytes CUDA backend not available — disabling 8-bit Adam. "
            "Training will use standard AdamW instead."
        )
        config["use_8bit_adam"] = False

    # Boolean flags
    bool_flags = {
        "train_text_encoder": "--train_text_encoder",
        "use_8bit_adam": "--use_8bit_adam",
        "enable_xformers_memory_efficient_attention": "--enable_xformers_memory_efficient_attention",
        "gradient_checkpointing": "--gradient_checkpointing",
        "center_crop": "--center_crop",
        "random_flip": "--random_flip",
    }

    # String/numeric parameters
    param_flags = {
        "pretrained_model_name_or_path": "--pretrained_model_name_or_path",
        "output_dir": "--output_dir",
        "instance_data_dir": "--instance_data_dir",
        "instance_prompt": "--instance_prompt",
        "train_batch_size": "--train_batch_size",
        "gradient_accumulation_steps": "--gradient_accumulation_steps",
        "learning_rate": "--learning_rate",
        "num_train_epochs": "--num_train_epochs",
        "rank": "--rank",
        "resolution": "--resolution",
        "lr_scheduler": "--lr_scheduler",
        "lr_warmup_steps": "--lr_warmup_steps",
        "seed": "--seed",
        "dataloader_num_workers": "--dataloader_num_workers",
        "max_train_steps": "--max_train_steps",
        "checkpointing_steps": "--checkpointing_steps",
        "validation_prompt": "--validation_prompt",
        "validation_epochs": "--validation_epochs",
        "snr_gamma": "--snr_gamma",
    }

    for key, flag in bool_flags.items():
        if config.get(key, False):
            cmd.append(flag)

    for key, flag in param_flags.items():
        val = config.get(key)
        if val is not None:
            cmd.extend([flag, str(val)])

    # Caption-based training: use dataset captions instead of a single instance prompt
    if config.get("use_captions", False):
        cmd.append("--dataset_name")
        cmd.append(config.get("instance_data_dir", "/dataset/images"))

    return cmd


def generate_config_snapshot(config: dict, output_dir: str) -> None:
    """Save resolved config as a .toml snapshot for reproducibility."""
    snapshot_path = Path(output_dir) / "training_config_resolved.toml"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "w") as f:
        toml.dump(config, f)
    log.info("Saved resolved config snapshot to %s", snapshot_path)


def train(config_path: str) -> None:
    """Run the full training pipeline."""
    # Verify training script exists
    if not Path(TRAINING_SCRIPT).exists():
        log.error("Training script not found: %s", TRAINING_SCRIPT)
        log.error("Ensure the Dockerfile downloaded it correctly.")
        sys.exit(1)

    # Load and resolve config
    config = load_config(config_path)
    config = resolve_config(config)

    # Verify dataset exists
    data_dir = config.get("instance_data_dir", "/dataset/images")
    if not Path(data_dir).exists() or not any(Path(data_dir).iterdir()):
        log.error("Dataset directory is empty or missing: %s", data_dir)
        sys.exit(1)

    # Verify model exists — supports both local paths and HuggingFace Hub IDs
    model_path = config.get("pretrained_model_name_or_path", "")
    if model_path and not Path(model_path).exists():
        if "/" in model_path and not model_path.startswith("/"):
            log.info(
                "Model path '%s' looks like a HuggingFace Hub ID. "
                "Will download at training time.",
                model_path,
            )
        else:
            log.warning(
                "Model path %s does not exist locally. "
                "Will attempt to download from HuggingFace Hub.",
                model_path,
            )

    # Save config snapshot
    output_dir = config.get("output_dir", "/output/lora")
    generate_config_snapshot(config, output_dir)

    # Build and run command
    cmd = build_command(config)
    log.info("Training command:")
    log.info("  %s", " ".join(cmd))
    log.info("Starting LoRA training...")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log.error("Training failed with exit code %d", result.returncode)
        sys.exit(result.returncode)

    log.info("Training complete! LoRA saved to %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="SDXL LoRA training wrapper")
    parser.add_argument(
        "--config",
        default=os.environ.get("TRAINING_CONFIG", "/app/config/training_config.toml"),
        help="Path to training configuration .toml file",
    )
    args = parser.parse_args()

    train(args.config)


if __name__ == "__main__":
    main()
