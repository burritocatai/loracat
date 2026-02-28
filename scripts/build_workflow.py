"""
build_workflow.py — Dynamically build a ComfyUI API workflow from prompts.

Generates a two-pass Qwen→Z-Image workflow for each prompt:
  Pass 1: Qwen Image Edit (generate character-consistent image from reference)
  Pass 2: Z-Image Turbo (refine/upscale with detail enhancement)

The workflow structure mirrors the manual ComfyUI graph but is generated
programmatically so the number of images scales with the prompt list.
"""

import json


# ── Default values ───────────────────────────────────────────────────────────
# Override any of these via the corresponding keys in workflow_config.json.

DEFAULTS = {
    "seed": 10000,
    "resolution": 1024,
    "refine_prompt": "detailed, visible skin pores, photorealistic",
    "model_shift": 3,
    "models": {
        "z_image_unet": "z_image_turbo_bf16.safetensors",
        "z_image_vae": "ae.safetensors",
        "qwen_vae": "qwen_image_vae.safetensors",
        "qwen_unet_gguf": "qwen-image-edit-2511-Q4_K_M.gguf",
        "lumina_clip": "qwen_3_4b.safetensors",
        "qwen_clip": "qwen_2.5_vl_7b.safetensors",
        "lightning_lora": "Qwen-Image-Lightning-4steps-V1.0.safetensors",
    },
    "pass1": {
        "steps": 4,
        "cfg": 1,
        "sampler": "euler",
        "scheduler": "simple",
        "denoise": 1,
    },
    "pass2": {
        "steps": 9,
        "cfg": 1,
        "sampler": "res_multistep",
        "scheduler": "simple",
        "denoise": 0.3,
    },
}


def _cfg(cfg: dict, key: str):
    """Get a config value with fallback to DEFAULTS."""
    return cfg.get(key, DEFAULTS.get(key))


def _model(cfg: dict, key: str) -> str:
    """Get a model filename from cfg['models'] with fallback to DEFAULTS."""
    models = cfg.get("models", {})
    return models.get(key, DEFAULTS["models"][key])


# ── Base nodes (shared across all prompts) ───────────────────────────────────


def _build_base_nodes(cfg: dict) -> dict:
    """Build the shared nodes that are loaded once for all prompts."""
    resolution = _cfg(cfg, "resolution")
    shift = _cfg(cfg, "model_shift")
    refine_prompt = _cfg(cfg, "refine_prompt")

    return {
        # Z-Image Turbo UNet
        "1": {
            "inputs": {
                "unet_name": _model(cfg, "z_image_unet"),
                "weight_dtype": "default",
            },
            "class_type": "UNETLoader",
            "_meta": {"title": "Load Diffusion Model"},
        },
        # Z-Image VAE
        "3": {
            "inputs": {"vae_name": _model(cfg, "z_image_vae")},
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"},
        },
        # Qwen VAE
        "4": {
            "inputs": {"vae_name": _model(cfg, "qwen_vae")},
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"},
        },
        # Qwen UNet (GGUF)
        "5": {
            "inputs": {"unet_name": _model(cfg, "qwen_unet_gguf")},
            "class_type": "UnetLoaderGGUF",
            "_meta": {"title": "Unet Loader (GGUF)"},
        },
        # Lumina CLIP (for refinement prompt)
        "6": {
            "inputs": {
                "clip_name": _model(cfg, "lumina_clip"),
                "type": "lumina2",
                "device": "default",
            },
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load CLIP"},
        },
        # Qwen CLIP (for edit prompts)
        "7": {
            "inputs": {
                "clip_name": _model(cfg, "qwen_clip"),
                "type": "qwen_image",
                "device": "default",
            },
            "class_type": "CLIPLoader",
            "_meta": {"title": "Load CLIP"},
        },
        # Lightning LoRA on Qwen UNet
        "8": {
            "inputs": {
                "lora_name": _model(cfg, "lightning_lora"),
                "strength_model": 1,
                "model": ["5", 0],
            },
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": "Load LoRA"},
        },
        # Scale face reference to ~1 megapixel
        "15": {
            "inputs": {
                "upscale_method": "lanczos",
                "megapixels": 1,
                "resolution_steps": 1,
                "image": ["16", 0],
            },
            "class_type": "ImageScaleToTotalPixels",
            "_meta": {"title": "ImageScaleToTotalPixels"},
        },
        # Face reference image (placeholder — injected at runtime)
        "16": {
            "inputs": {"image": "face_reference.png"},
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"},
        },
        # Empty latent for pass-1 generation
        "21": {
            "inputs": {
                "width": resolution,
                "height": resolution,
                "batch_size": 1,
            },
            "class_type": "EmptyLatentImage",
            "_meta": {"title": "Empty Latent Image"},
        },
        # Shared refinement prompt (used by all pass-2 KSamplers)
        "40": {
            "inputs": {
                "text": refine_prompt,
                "clip": ["6", 0],
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Prompt)"},
        },
        # ModelSamplingAuraFlow — pass 1 (Qwen edit + Lightning LoRA)
        "370": {
            "inputs": {"shift": shift, "model": ["8", 0]},
            "class_type": "ModelSamplingAuraFlow",
            "_meta": {"title": "ModelSamplingAuraFlow"},
        },
        # ModelSamplingAuraFlow — pass 2 (Z-Image Turbo)
        "371": {
            "inputs": {"shift": shift, "model": ["1", 0]},
            "class_type": "ModelSamplingAuraFlow",
            "_meta": {"title": "ModelSamplingAuraFlow"},
        },
    }


# ── Per-prompt node block ────────────────────────────────────────────────────


def _build_prompt_block(
    prompt_text: str,
    block_index: int,
    cfg: dict,
) -> dict:
    """Build the 10 nodes for a single prompt (pass 1 + pass 2).

    Node IDs are allocated as 1000 + block_index * 10 + offset so they
    never collide with the base nodes (IDs < 1000).
    """
    seed = _cfg(cfg, "seed")
    p1 = {**DEFAULTS["pass1"], **cfg.get("pass1", {})}
    p2 = {**DEFAULTS["pass2"], **cfg.get("pass2", {})}
    output_prefix = cfg.get("output_prefix", "Qwen_Consistent_Char/Z_IMG_FINAL")

    base = 1000 + block_index * 10

    # Pass 1 node IDs
    text_encode = str(base)
    zero_out_1 = str(base + 1)
    ksampler_1 = str(base + 2)
    vae_decode_1 = str(base + 3)
    save_cc = str(base + 4)

    # Pass 2 node IDs
    vae_encode = str(base + 5)
    zero_out_2 = str(base + 6)
    ksampler_2 = str(base + 7)
    vae_decode_2 = str(base + 8)
    save_final = str(base + 9)

    return {
        # ── Pass 1: Qwen Image Edit ──────────────────────────────────────
        text_encode: {
            "inputs": {
                "prompt": prompt_text,
                "clip": ["7", 0],
                "vae": ["4", 0],
                "image1": ["15", 0],
            },
            "class_type": "TextEncodeQwenImageEditPlus",
            "_meta": {"title": "TextEncodeQwenImageEditPlus"},
        },
        zero_out_1: {
            "inputs": {"conditioning": [text_encode, 0]},
            "class_type": "ConditioningZeroOut",
            "_meta": {"title": "ConditioningZeroOut"},
        },
        ksampler_1: {
            "inputs": {
                "seed": seed,
                "steps": p1["steps"],
                "cfg": p1["cfg"],
                "sampler_name": p1["sampler"],
                "scheduler": p1["scheduler"],
                "denoise": p1["denoise"],
                "model": ["370", 0],
                "positive": [text_encode, 0],
                "negative": [zero_out_1, 0],
                "latent_image": ["21", 0],
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
        },
        vae_decode_1: {
            "inputs": {"samples": [ksampler_1, 0], "vae": ["4", 0]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        },
        save_cc: {
            "inputs": {
                "filename_prefix": "Qwen_Consistent_Char/CC",
                "images": [vae_decode_1, 0],
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
        },
        # ── Pass 2: Z-Image Refinement ───────────────────────────────────
        vae_encode: {
            "inputs": {"pixels": [vae_decode_1, 0], "vae": ["3", 0]},
            "class_type": "VAEEncode",
            "_meta": {"title": "VAE Encode"},
        },
        zero_out_2: {
            "inputs": {"conditioning": ["40", 0]},
            "class_type": "ConditioningZeroOut",
            "_meta": {"title": "ConditioningZeroOut"},
        },
        ksampler_2: {
            "inputs": {
                "seed": seed,
                "steps": p2["steps"],
                "cfg": p2["cfg"],
                "sampler_name": p2["sampler"],
                "scheduler": p2["scheduler"],
                "denoise": p2["denoise"],
                "model": ["371", 0],
                "positive": ["40", 0],
                "negative": [zero_out_2, 0],
                "latent_image": [vae_encode, 0],
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"},
        },
        vae_decode_2: {
            "inputs": {"samples": [ksampler_2, 0], "vae": ["3", 0]},
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"},
        },
        save_final: {
            "inputs": {
                "filename_prefix": output_prefix,
                "images": [vae_decode_2, 0],
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"},
        },
    }


# ── Public API ───────────────────────────────────────────────────────────────


def build_workflow(prompts: list[str], cfg: dict | None = None) -> dict:
    """Build a complete ComfyUI API workflow from a list of prompts.

    Args:
        prompts: List of prompt strings for image generation.
        cfg: Optional config dict (from workflow_config.json) with model
             names, seeds, sampler settings, etc.  Missing keys fall back
             to built-in DEFAULTS.

    Returns:
        Complete ComfyUI API-format workflow dict ready for queue_prompt().
    """
    if cfg is None:
        cfg = {}

    workflow = _build_base_nodes(cfg)

    for i, prompt_text in enumerate(prompts):
        block = _build_prompt_block(prompt_text, i, cfg)
        workflow.update(block)

    return workflow


def load_prompts(prompts_path: str) -> list[str]:
    """Load prompts from a JSON file.

    Supports three formats:
      - Simple list:  ["prompt1", "prompt2", ...]
      - Object:       {"prompts": ["prompt1", "prompt2", ...]}
      - Batches:      {"batches": [{"prompt": "...", "clothes": "...", ...}, ...]}

    For batches, the full prompt is assembled from the base prompt + keys.
    """
    with open(prompts_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "prompts" in data:
        return data["prompts"]

    if isinstance(data, dict) and "batches" in data:
        prompts = []
        for entry in data["batches"]:
            parts = [entry["prompt"]]
            if entry.get("clothes"):
                parts.append(f"wearing {entry['clothes']}")
            if entry.get("expression"):
                parts.append(f"with {entry['expression']} expression")
            if entry.get("angle"):
                parts.append(f"from {entry['angle']} angle")
            if entry.get("lighting"):
                parts.append(f"with {entry['lighting']} lighting")
            prompts.append(", ".join(parts))
        return prompts

    raise ValueError(
        f"Expected a JSON array, or an object with a 'prompts' or 'batches' key, "
        f"got {type(data).__name__}"
    )
