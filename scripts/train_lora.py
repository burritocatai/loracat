#!/usr/bin/env python3
"""
train_lora.py — Z-Image LoRA training wrapper using musubi-tuner.

Reads a .toml configuration file (for reproducibility), merges in environment
variable overrides, and runs the 3-step musubi-tuner pipeline:
  1. Cache latents (VAE encode)
  2. Cache text encoder outputs
  3. Train LoRA network
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

MUSUBI_DIR = "/app/musubi-tuner/src/musubi_tuner"
CACHE_LATENTS_SCRIPT = f"{MUSUBI_DIR}/zimage_cache_latents.py"
CACHE_TEXT_ENC_SCRIPT = f"{MUSUBI_DIR}/zimage_cache_text_encoder_outputs.py"
TRAIN_SCRIPT = f"{MUSUBI_DIR}/zimage_train_network.py"


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
        "dit": "DIT_PATH",
        "vae": "VAE_PATH",
        "text_encoder": "TEXT_ENCODER_PATH",
        "output_dir": "OUTPUT_DIR",
        "output_name": "OUTPUT_NAME",
        "train_batch_size": "TRAIN_BATCH_SIZE",
        "gradient_accumulation_steps": "GRADIENT_ACCUMULATION_STEPS",
        "learning_rate": "LEARNING_RATE",
        "max_train_epochs": "NUM_TRAIN_EPOCHS",
        "network_dim": "NETWORK_DIM",
        "mixed_precision": "MIXED_PRECISION",
        "dataloader_num_workers": "DATALOADER_NUM_WORKERS",
        "discrete_flow_shift": "DISCRETE_FLOW_SHIFT",
    }

    int_params = {
        "train_batch_size",
        "gradient_accumulation_steps",
        "max_train_epochs",
        "network_dim",
        "dataloader_num_workers",
        "save_every_n_epochs",
        "seed",
        "lr_warmup_steps",
        "cache_text_encoder_batch_size",
    }
    float_params = {"learning_rate", "discrete_flow_shift"}

    for param, env_var in env_overrides.items():
        val = os.environ.get(env_var)
        if val is not None:
            if param in int_params:
                resolved[param] = int(val)
            elif param in float_params:
                resolved[param] = float(val)
            else:
                resolved[param] = val

    return resolved


def generate_dataset_config(template_path: str, cache_dir: str) -> str:
    """Generate runtime dataset config from template, applying env overrides.

    Writes the config to cache_dir/dataset_config.toml and returns the path.
    """
    log.info("Loading dataset config template from %s", template_path)
    with open(template_path) as f:
        ds_config = toml.load(f)

    # Apply env overrides
    resolution = os.environ.get("RESOLUTION")
    if resolution:
        res = int(resolution)
        ds_config["general"]["resolution"] = [res, res]

    batch_size = os.environ.get("TRAIN_BATCH_SIZE")
    if batch_size:
        ds_config["general"]["batch_size"] = int(batch_size)

    dataset_dir = os.environ.get("DATASET_DIR", "/dataset/images")
    ds_config["datasets"][0]["image_directory"] = dataset_dir
    ds_config["datasets"][0]["cache_directory"] = cache_dir

    output_path = str(Path(cache_dir) / "dataset_config.toml")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        toml.dump(ds_config, f)
    log.info("Generated runtime dataset config: %s", output_path)

    return output_path


def generate_config_snapshot(config: dict, output_dir: str) -> None:
    """Save resolved config as a .toml snapshot for reproducibility."""
    snapshot_path = Path(output_dir) / "training_config_resolved.toml"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(snapshot_path, "w") as f:
        toml.dump(config, f)
    log.info("Saved resolved config snapshot to %s", snapshot_path)


def run_step(cmd: list[str], step_name: str) -> None:
    """Run a subprocess, failing loudly on error."""
    log.info("%s command:", step_name)
    log.info("  %s", " ".join(cmd))
    log.info("Starting %s...", step_name)

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log.error("%s failed with exit code %d", step_name, result.returncode)
        sys.exit(result.returncode)

    log.info("%s complete.", step_name)


def cache_latents(dataset_config_path: str, vae_path: str) -> None:
    """Step 1: Pre-cache VAE latents."""
    cmd = [
        "python", CACHE_LATENTS_SCRIPT,
        "--dataset_config", dataset_config_path,
        "--vae", vae_path,
    ]
    run_step(cmd, "Cache latents")


def cache_text_encoder(dataset_config_path: str, text_encoder_path: str,
                       batch_size: int = 16) -> None:
    """Step 2: Pre-cache text encoder outputs."""
    cmd = [
        "python", CACHE_TEXT_ENC_SCRIPT,
        "--dataset_config", dataset_config_path,
        "--text_encoder", text_encoder_path,
        "--batch_size", str(batch_size),
    ]
    run_step(cmd, "Cache text encoder outputs")


def build_train_command(config: dict, dataset_config_path: str) -> list[str]:
    """Build the accelerate launch command for LoRA training."""
    cmd = [
        "accelerate", "launch",
        "--mixed_precision", config.get("mixed_precision", "bf16"),
        TRAIN_SCRIPT,
    ]

    # Path arguments
    for key in ("dit", "vae", "text_encoder"):
        val = config.get(key)
        if val:
            cmd.extend([f"--{key}", str(val)])

    cmd.extend(["--dataset_config", dataset_config_path])

    # Network config
    cmd.extend(["--network_module", config.get("network_module", "networks.lora_zimage")])
    cmd.extend(["--network_dim", str(config.get("network_dim", 32))])

    # Boolean flags
    bool_flags = {
        "sdpa": "--sdpa",
        "gradient_checkpointing": "--gradient_checkpointing",
    }
    for key, flag in bool_flags.items():
        if config.get(key, False):
            cmd.append(flag)

    # String/numeric parameters
    param_flags = {
        "optimizer_type": "--optimizer_type",
        "learning_rate": "--learning_rate",
        "lr_scheduler": "--lr_scheduler",
        "lr_warmup_steps": "--lr_warmup_steps",
        "max_train_epochs": "--max_train_epochs",
        "save_every_n_epochs": "--save_every_n_epochs",
        "seed": "--seed",
        "output_dir": "--output_dir",
        "output_name": "--output_name",
        "timestep_sampling": "--timestep_sampling",
        "weighting_scheme": "--weighting_scheme",
        "discrete_flow_shift": "--discrete_flow_shift",
        "gradient_accumulation_steps": "--gradient_accumulation_steps",
        "dataloader_num_workers": "--max_data_loader_n_workers",
        "mixed_precision": "--mixed_precision",
    }

    for key, flag in param_flags.items():
        val = config.get(key)
        if val is not None:
            cmd.extend([flag, str(val)])

    return cmd


def train(config_path: str) -> None:
    """Run the full 3-step training pipeline."""
    # Load and resolve config
    config = load_config(config_path)
    config = resolve_config(config)

    # Verify dataset exists
    dataset_dir = os.environ.get("DATASET_DIR", "/dataset/images")
    if not Path(dataset_dir).exists() or not any(Path(dataset_dir).iterdir()):
        log.error("Dataset directory is empty or missing: %s", dataset_dir)
        sys.exit(1)

    # Verify model paths exist
    for key, label in [("dit", "DiT"), ("vae", "VAE"), ("text_encoder", "Text Encoder")]:
        path = config.get(key, "")
        if not Path(path).exists():
            log.error("%s model not found: %s", label, path)
            sys.exit(1)

    # Save config snapshot
    output_dir = config.get("output_dir", "/output/lora")
    generate_config_snapshot(config, output_dir)

    # Generate runtime dataset config
    cache_dir = os.environ.get("CACHE_DIR", "/dataset/cache")
    dataset_template = os.environ.get("DATASET_CONFIG", "/app/config/dataset_config.toml")
    dataset_config_path = generate_dataset_config(dataset_template, cache_dir)

    # Step 1: Cache latents
    log.info("=" * 60)
    log.info("STEP 1/3 — Caching VAE latents")
    log.info("=" * 60)
    cache_latents(dataset_config_path, config["vae"])

    # Step 2: Cache text encoder outputs
    log.info("=" * 60)
    log.info("STEP 2/3 — Caching text encoder outputs")
    log.info("=" * 60)
    te_batch_size = config.get("cache_text_encoder_batch_size", 16)
    cache_text_encoder(dataset_config_path, config["text_encoder"], te_batch_size)

    # Step 3: Train LoRA
    log.info("=" * 60)
    log.info("STEP 3/3 — Training Z-Image LoRA")
    log.info("=" * 60)
    cmd = build_train_command(config, dataset_config_path)
    run_step(cmd, "LoRA training")

    log.info("Training complete! LoRA saved to %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Z-Image LoRA training wrapper (musubi-tuner)")
    parser.add_argument(
        "--config",
        default=os.environ.get("TRAINING_CONFIG", "/app/config/training_config.toml"),
        help="Path to training configuration .toml file",
    )
    args = parser.parse_args()

    train(args.config)


if __name__ == "__main__":
    main()
