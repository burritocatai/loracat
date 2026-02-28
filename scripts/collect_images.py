#!/usr/bin/env python3
"""
collect_images.py — Batch image collection from a ComfyUI API endpoint.

Supports three modes, auto-detected from workflow_config.json:

  Generated-workflow mode (recommended for Qwen→Z-Turbo pipelines):
    - Builds the workflow dynamically from prompts.json
    - Number of images = number of prompts
    - workflow_config.json needs: generate_workflow: true, output_prefix
    - All model names, sampler settings, etc. configurable in workflow_config

  Full-workflow mode (legacy — static workflow file):
    - Workflow has all prompts baked in
    - Uploads face reference, injects into image loader node
    - Submits once, downloads only outputs matching output_prefix
    - workflow_config.json needs: image_loader_node, output_prefix

  Per-batch mode (for simple single-prompt workflows):
    - Injects prompt/seed per batch entry from prompts.json
    - workflow_config.json needs: positive_prompt_node, seed_node
"""

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urljoin, quote

import requests
import websocket

from build_workflow import build_workflow, load_prompts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Workflow config ──────────────────────────────────────────────────────────


def load_workflow_config(config_path: str) -> dict:
    """Load node-ID mapping from workflow_config.json."""
    log.info("Loading workflow config from %s", config_path)
    with open(config_path) as f:
        cfg = json.load(f)

    # Determine mode
    if cfg.get("generate_workflow"):
        cfg["_mode"] = "generated"
        log.info("  Mode: generated (workflow built from prompts)")
        return cfg

    has_batch_fields = "positive_prompt_node" in cfg and "seed_node" in cfg
    has_prefix = "output_prefix" in cfg

    if not has_batch_fields and not has_prefix:
        raise ValueError(
            "workflow_config.json must have either "
            "(positive_prompt_node + seed_node) for per-batch mode, "
            "or output_prefix for full-workflow mode"
        )

    cfg["_mode"] = "batch" if has_batch_fields else "full"
    log.info("  Mode: %s", cfg["_mode"])
    return cfg


def load_workflow(workflow_path: str) -> dict:
    """Load a ComfyUI API-format workflow JSON."""
    log.info("Loading workflow from %s", workflow_path)
    with open(workflow_path) as f:
        return json.load(f)


# ── Workflow scanning ────────────────────────────────────────────────────────


def find_output_nodes(workflow: dict, prefix: str) -> list[str]:
    """Find all SaveImage node IDs whose filename_prefix matches."""
    matches = []
    for node_id, node in workflow.items():
        if node.get("class_type") != "SaveImage":
            continue
        node_prefix = node.get("inputs", {}).get("filename_prefix", "")
        if node_prefix == prefix or node_prefix.endswith("/" + prefix):
            matches.append(node_id)
    matches.sort(key=lambda x: int(x) if x.isdigit() else x)
    return matches


def find_ksampler_nodes(workflow: dict) -> list[str]:
    """Find all KSampler node IDs in the workflow."""
    return sorted(
        (nid for nid, n in workflow.items() if n.get("class_type") == "KSampler"),
        key=lambda x: int(x) if x.isdigit() else x,
    )


# ── ComfyUI API helpers ─────────────────────────────────────────────────────


def upload_image(endpoint: str, image_path: str) -> str:
    """Upload an image to ComfyUI via /upload/image.

    Returns the server-side filename (may differ from the local name
    if ComfyUI deduplicates).
    """
    url = urljoin(endpoint, "/upload/image")
    path = Path(image_path)
    log.info("Uploading %s to ComfyUI ...", path.name)

    with open(path, "rb") as f:
        resp = requests.post(
            url,
            files={"image": (path.name, f, "image/png")},
            timeout=60,
        )
    resp.raise_for_status()
    data = resp.json()
    remote_name = data.get("name", path.name)
    subfolder = data.get("subfolder", "")
    log.info("  Uploaded as: %s (subfolder=%r)", remote_name, subfolder)
    return remote_name


def inject_face_image(workflow: dict, node_id: str, filename: str) -> dict:
    """Inject uploaded face image filename into a LoadImage node."""
    wf = json.loads(json.dumps(workflow))
    if node_id not in wf:
        raise KeyError(
            f"image_loader_node '{node_id}' not found in workflow. "
            f"Available: {sorted(wf.keys())}"
        )
    wf[node_id]["inputs"]["image"] = filename
    return wf


def inject_batch_values(
    workflow: dict,
    node_cfg: dict,
    *,
    prompt_text: str,
    seed: int,
    face_image: str | None = None,
) -> dict:
    """Deep-copy workflow and inject prompt/seed/image for per-batch mode."""
    wf = json.loads(json.dumps(workflow))

    # Positive prompt
    prompt_node = node_cfg["positive_prompt_node"]
    if prompt_node not in wf:
        raise KeyError(
            f"positive_prompt_node '{prompt_node}' not found in workflow. "
            f"Available: {sorted(wf.keys())}"
        )
    wf[prompt_node]["inputs"]["text"] = prompt_text

    # Seed
    seed_node = node_cfg["seed_node"]
    if seed_node not in wf:
        raise KeyError(
            f"seed_node '{seed_node}' not found in workflow. "
            f"Available: {sorted(wf.keys())}"
        )
    inputs = wf[seed_node]["inputs"]
    if "seed" in inputs:
        inputs["seed"] = seed
    elif "noise_seed" in inputs:
        inputs["noise_seed"] = seed
    else:
        inputs["seed"] = seed

    # Face image
    image_node = node_cfg.get("image_loader_node")
    if image_node and face_image:
        if image_node not in wf:
            raise KeyError(
                f"image_loader_node '{image_node}' not found in workflow. "
                f"Available: {sorted(wf.keys())}"
            )
        wf[image_node]["inputs"]["image"] = face_image

    return wf


def inject_global_seed(workflow: dict, seed: int, seed_nodes: list[str]) -> dict:
    """Set seed on specific KSampler nodes."""
    wf = json.loads(json.dumps(workflow))
    for nid in seed_nodes:
        if nid in wf:
            inputs = wf[nid]["inputs"]
            if "seed" in inputs:
                inputs["seed"] = seed
            elif "noise_seed" in inputs:
                inputs["noise_seed"] = seed
    return wf


def queue_prompt(endpoint: str, workflow: dict, client_id: str) -> str:
    """Submit a prompt to ComfyUI and return the prompt_id."""
    url = urljoin(endpoint, "/prompt")
    payload = {"prompt": workflow, "client_id": client_id}
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["prompt_id"]


def wait_for_completion(ws_url: str, prompt_id: str, timeout: int = 3600) -> None:
    """Wait for a prompt to finish executing via WebSocket.

    Default timeout is 1 hour — large multi-prompt workflows can take a while.
    """
    ws = websocket.create_connection(ws_url, timeout=timeout)
    try:
        while True:
            msg = ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                msg_type = data.get("type")

                if msg_type == "executing":
                    exec_data = data.get("data", {})
                    if (
                        exec_data.get("prompt_id") == prompt_id
                        and exec_data.get("node") is None
                    ):
                        return

                # Log progress for long-running workflows
                if msg_type == "progress":
                    prog = data.get("data", {})
                    value = prog.get("value", 0)
                    maximum = prog.get("max", 0)
                    if maximum:
                        log.info("  Progress: %d/%d", value, maximum)
    finally:
        ws.close()


def get_images(
    endpoint: str,
    prompt_id: str,
    output_nodes: list[str] | None = None,
) -> list[tuple[str, bytes]]:
    """Retrieve output images for a completed prompt.

    If output_nodes is given, only images from those nodes are returned.
    """
    url = urljoin(endpoint, f"/history/{prompt_id}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    history = resp.json()

    if prompt_id not in history:
        raise RuntimeError(f"Prompt {prompt_id} not found in history")

    images = []
    outputs = history[prompt_id].get("outputs", {})

    if output_nodes:
        nodes_to_check = {
            nid: outputs[nid] for nid in output_nodes if nid in outputs
        }
    else:
        nodes_to_check = outputs

    for node_id, node_output in nodes_to_check.items():
        for img_info in node_output.get("images", []):
            filename = img_info["filename"]
            subfolder = img_info.get("subfolder", "")
            img_type = img_info.get("type", "output")

            params = (
                f"filename={quote(filename)}"
                f"&subfolder={quote(subfolder)}"
                f"&type={quote(img_type)}"
            )
            view_url = urljoin(endpoint, f"/view?{params}")
            img_resp = requests.get(view_url, timeout=60)
            img_resp.raise_for_status()
            images.append((filename, img_resp.content))

    return images


# ── Memory cleanup ────────────────────────────────────────────────────────────

UNLOAD_MODELS_WORKFLOW = {
    "1": {
        "inputs": {"value": ["4", 0]},
        "class_type": "UnloadAllModels",
        "_meta": {"title": "UnloadAllModels"},
    },
    "3": {
        "inputs": {"any_input": ["1", 0]},
        "class_type": "DummyOut",
        "_meta": {"title": "Dummy Out"},
    },
    "4": {
        "inputs": {},
        "class_type": "ImpactDummyInput",
        "_meta": {"title": "ImpactDummyInput"},
    },
}


def unload_models(endpoint: str) -> None:
    """Run a dummy workflow that unloads all models from ComfyUI GPU memory.

    This frees VRAM after image generation so that subsequent training
    is not starved for memory.
    """
    endpoint = endpoint.rstrip("/")
    ws_endpoint = endpoint.replace("http://", "ws://").replace("https://", "wss://")
    client_id = str(uuid.uuid4())
    ws_url = f"{ws_endpoint}/ws?clientId={client_id}"

    log.info("Unloading ComfyUI models to free GPU memory ...")
    prompt_id = queue_prompt(endpoint, UNLOAD_MODELS_WORKFLOW, client_id)
    wait_for_completion(ws_url, prompt_id, timeout=120)
    log.info("Models unloaded successfully")


# ── Full-workflow mode ───────────────────────────────────────────────────────


def collect_full_workflow(
    endpoint: str,
    output_dir: str,
    workflow: dict,
    node_cfg: dict,
    face_image_path: str | None = None,
    seed: int | None = None,
) -> int:
    """Submit the full workflow once and download filtered outputs."""
    endpoint = endpoint.rstrip("/")
    ws_endpoint = endpoint.replace("http://", "ws://").replace("https://", "wss://")
    client_id = str(uuid.uuid4())
    ws_url = f"{ws_endpoint}/ws?clientId={client_id}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find output nodes by prefix
    prefix = node_cfg["output_prefix"]
    output_nodes = find_output_nodes(workflow, prefix)
    if not output_nodes:
        log.error(
            "No SaveImage nodes found with filename_prefix matching '%s'",
            prefix,
        )
        return 0
    log.info(
        "Found %d output nodes matching prefix '%s': %s",
        len(output_nodes), prefix, ", ".join(output_nodes),
    )

    # Deep copy for injection
    wf = json.loads(json.dumps(workflow))

    # Upload and inject face image
    if face_image_path:
        if not Path(face_image_path).exists():
            log.error("Face reference image not found: %s", face_image_path)
            sys.exit(1)
        remote_name = upload_image(endpoint, face_image_path)
        image_node = node_cfg.get("image_loader_node")
        if image_node:
            if image_node not in wf:
                raise KeyError(
                    f"image_loader_node '{image_node}' not found in workflow"
                )
            wf[image_node]["inputs"]["image"] = remote_name
            log.info("Injected face image into node %s", image_node)
        else:
            log.warning("Face image uploaded but no image_loader_node configured")

    # Optionally override all seeds
    if seed is not None:
        seed_nodes = node_cfg.get("seed_nodes")
        if seed_nodes:
            wf = inject_global_seed(wf, seed, seed_nodes)
            log.info("Set seed=%d on %d nodes", seed, len(seed_nodes))
        else:
            # Auto-discover all KSampler nodes
            all_ksamplers = find_ksampler_nodes(wf)
            wf = inject_global_seed(wf, seed, all_ksamplers)
            log.info(
                "Set seed=%d on all %d KSampler nodes", seed, len(all_ksamplers)
            )

    # Submit
    log.info("Submitting full workflow (%d nodes) ...", len(wf))
    prompt_id = queue_prompt(endpoint, wf, client_id)
    log.info("Queued prompt_id=%s — waiting for completion...", prompt_id)
    log.info("(This may take a while for large multi-prompt workflows)")

    wait_for_completion(ws_url, prompt_id)
    log.info("Workflow execution complete")

    # Download only the Z_IMG_FINAL outputs
    images = get_images(endpoint, prompt_id, output_nodes)
    log.info("Retrieved %d output images (filtered to '%s' nodes)", len(images), prefix)

    total_saved = 0
    for i, (orig_name, img_data) in enumerate(images):
        ext = Path(orig_name).suffix or ".png"
        filename = f"z_final_{i:04d}{ext}"
        save_path = output_path / filename
        save_path.write_bytes(img_data)
        log.info("  Saved: %s (%d bytes)", save_path.name, len(img_data))
        total_saved += 1

    log.info("Collection complete: %d images saved to %s", total_saved, output_dir)
    return total_saved


# ── Per-batch mode ───────────────────────────────────────────────────────────


def collect_per_batch(
    endpoint: str,
    prompts_file: str,
    output_dir: str,
    workflow: dict,
    node_cfg: dict,
    face_image_path: str | None = None,
    delay: float = 2.0,
) -> int:
    """Submit workflow once per batch entry, injecting prompt/seed each time."""
    endpoint = endpoint.rstrip("/")
    ws_endpoint = endpoint.replace("http://", "ws://").replace("https://", "wss://")
    client_id = str(uuid.uuid4())
    ws_url = f"{ws_endpoint}/ws?clientId={client_id}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_node = node_cfg.get("output_node")

    # Upload face reference image
    face_remote_name = None
    if face_image_path:
        if not Path(face_image_path).exists():
            log.error("Face reference image not found: %s", face_image_path)
            sys.exit(1)
        face_remote_name = upload_image(endpoint, face_image_path)
    elif node_cfg.get("image_loader_node"):
        log.warning(
            "image_loader_node '%s' configured but no --face-image provided",
            node_cfg["image_loader_node"],
        )

    # Load prompts
    with open(prompts_file) as f:
        batch_data = json.load(f)

    batches = batch_data.get("batches", [])
    if not batches:
        log.error("No batches found in %s", prompts_file)
        return 0

    log.info("Loaded %d batch entries from %s", len(batches), prompts_file)

    # Resolve output nodes for filtering
    output_nodes = None
    if node_cfg.get("output_prefix"):
        output_nodes = find_output_nodes(workflow, node_cfg["output_prefix"])
    elif output_node:
        output_nodes = [output_node]

    total_saved = 0
    for i, entry in enumerate(batches):
        base_prompt = entry["prompt"]
        base_seed = entry.get("seed", 0)
        count = entry.get("count", 1)
        angle = entry.get("angle")
        expression = entry.get("expression")
        clothes = entry.get("clothes")
        lighting = entry.get("lighting")

        # Build full prompt from base prompt + keys
        parts = [base_prompt]
        if clothes:
            parts.append(f"wearing {clothes}")
        if expression:
            parts.append(f"with {expression} expression")
        if angle:
            parts.append(f"from {angle} angle")
        if lighting:
            parts.append(f"with {lighting} lighting")
        prompt_text = ", ".join(parts)

        log.info(
            "[%d/%d] angle=%s, expression=%s, clothes=%s, lighting=%s, base_seed=%d, count=%d",
            i + 1, len(batches), angle, expression, clothes, lighting, base_seed, count,
        )

        for c in range(count):
            seed = base_seed + c
            wf = inject_batch_values(
                workflow, node_cfg,
                prompt_text=prompt_text, seed=seed,
                face_image=face_remote_name,
            )

            try:
                prompt_id = queue_prompt(endpoint, wf, client_id)
                log.info("  [%d/%d] seed=%d  prompt_id=%s", c + 1, count, seed, prompt_id)

                wait_for_completion(ws_url, prompt_id)

                images = get_images(endpoint, prompt_id, output_nodes)
                for j, (orig_name, img_data) in enumerate(images):
                    ext = Path(orig_name).suffix or ".png"
                    filename = f"{angle or 'unknown'}_{expression or 'unknown'}_{seed}_{j:03d}{ext}"
                    save_path = output_path / filename
                    save_path.write_bytes(img_data)
                    log.info("    Saved: %s (%d bytes)", save_path.name, len(img_data))
                    total_saved += 1

            except Exception:
                log.exception("  Failed on batch %d, seed %d", i + 1, seed)
                raise

            if not (i == len(batches) - 1 and c == count - 1):
                log.info("    Waiting %.1fs ...", delay)
                time.sleep(delay)

    log.info("Collection complete: %d images saved to %s", total_saved, output_dir)
    return total_saved


# ── Entrypoint ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Collect images from ComfyUI using an exported workflow"
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("COMFYUI_ENDPOINT", "http://localhost:8188"),
        help="ComfyUI API endpoint URL",
    )
    parser.add_argument(
        "--prompts",
        default=os.environ.get("PROMPTS_FILE", ""),
        help="Path to JSON batch prompts file (required for per-batch mode)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("DATASET_DIR", "/dataset/images"),
        help="Directory to save collected images",
    )
    parser.add_argument(
        "--workflow",
        default=os.environ.get("COMFYUI_WORKFLOW", "/app/workflow_api.json"),
        help="Path to ComfyUI API-format workflow JSON",
    )
    parser.add_argument(
        "--workflow-config",
        default=os.environ.get("WORKFLOW_CONFIG", "/app/config/workflow_config.json"),
        help="Path to workflow_config.json (node ID mapping)",
    )
    parser.add_argument(
        "--face-image",
        default=os.environ.get("FACE_REFERENCE", ""),
        help="Path to face reference image to upload to ComfyUI",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global seed override for all KSampler nodes (full-workflow mode)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=float(os.environ.get("COMFYUI_DELAY", "2.0")),
        help="Delay in seconds between API requests (per-batch mode)",
    )
    args = parser.parse_args()

    face = args.face_image if args.face_image else None

    node_cfg = load_workflow_config(args.workflow_config)

    if node_cfg["_mode"] == "generated":
        if not args.prompts:
            log.error("Generated-workflow mode requires --prompts file")
            sys.exit(1)
        prompts = load_prompts(args.prompts)
        log.info("Building workflow from %d prompts", len(prompts))
        workflow = build_workflow(prompts, node_cfg)
        log.info("Built workflow with %d nodes", len(workflow))
    else:
        workflow = load_workflow(args.workflow)

    if node_cfg["_mode"] in ("generated", "full"):
        log.info("Running in full-workflow mode (%d nodes)", len(workflow))
        count = collect_full_workflow(
            endpoint=args.endpoint,
            output_dir=args.output_dir,
            workflow=workflow,
            node_cfg=node_cfg,
            face_image_path=face,
            seed=args.seed,
        )
    else:
        log.info("Running in per-batch mode (injecting prompt/seed per entry)")
        if not args.prompts:
            log.error("Per-batch mode requires --prompts file")
            sys.exit(1)
        count = collect_per_batch(
            endpoint=args.endpoint,
            prompts_file=args.prompts,
            output_dir=args.output_dir,
            workflow=workflow,
            node_cfg=node_cfg,
            face_image_path=face,
            delay=args.delay,
        )

    if count == 0:
        log.error("No images were collected!")
        sys.exit(1)

    # Free GPU memory so subsequent training isn't starved for VRAM
    unload_models(args.endpoint)


if __name__ == "__main__":
    main()
