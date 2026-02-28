#!/usr/bin/env python3
"""
caption_images.py — Auto-caption images using WD14 tagger (onnxruntime CPU).

Generates booru-style tag captions for all images in the dataset directory,
injects a consistent trigger word, and saves .txt sidecar files.
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# WD14 tagger model — SmilingWolf's v3 ViT model
MODEL_REPO = "SmilingWolf/wd-vit-tagger-v3"
MODEL_FILENAME = "model.onnx"
LABELS_FILENAME = "selected_tags.csv"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def download_model(repo: str = MODEL_REPO) -> tuple[str, str]:
    """Download the WD14 ONNX model and labels from HuggingFace."""
    log.info("Downloading WD14 model from %s ...", repo)
    model_path = hf_hub_download(repo_id=repo, filename=MODEL_FILENAME)
    labels_path = hf_hub_download(repo_id=repo, filename=LABELS_FILENAME)
    log.info("Model cached at: %s", model_path)
    return model_path, labels_path


def load_labels(labels_path: str) -> tuple[list[str], list[int], list[int]]:
    """Load tag labels and category indices from the CSV file."""
    tags = []
    general_indices = []
    character_indices = []

    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            tags.append(row["name"].strip())
            category = int(row["category"])
            if category == 0:  # general tags
                general_indices.append(i)
            elif category == 4:  # character tags
                character_indices.append(i)

    return tags, general_indices, character_indices


def preprocess_image(image: Image.Image, size: int = 448) -> np.ndarray:
    """Preprocess an image for WD14 tagger inference."""
    # Flatten alpha onto white background
    img = image.convert("RGBA")
    canvas = Image.new("RGBA", img.size, (255, 255, 255, 255))
    canvas.alpha_composite(img)
    img = canvas.convert("RGB")

    # Pad to square (preserving aspect ratio) with white background
    w, h = img.size
    max_dim = max(w, h)
    pad_left = (max_dim - w) // 2
    pad_top = (max_dim - h) // 2
    padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    padded.paste(img, (pad_left, pad_top))

    # Resize to model input size
    if max_dim != size:
        padded = padded.resize((size, size), Image.BICUBIC)

    # Convert to numpy float32 — WD14 expects [0, 255] range, NOT [0, 1]
    arr = np.asarray(padded, dtype=np.float32)

    # WD14 expects BGR channel order
    arr = arr[:, :, ::-1]

    # Add batch dimension
    return np.expand_dims(arr, axis=0)


def predict_tags(
    session: ort.InferenceSession,
    image: Image.Image,
    tags: list[str],
    general_indices: list[int],
    general_threshold: float = 0.35,
) -> list[str]:
    """Run WD14 tagger inference and return tags above threshold."""
    input_data = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    results = session.run([output_name], {input_name: input_data})
    probs = results[0][0]

    # Extract general tags above threshold
    predicted = []
    for idx in general_indices:
        if probs[idx] >= general_threshold:
            predicted.append((tags[idx], float(probs[idx])))

    # Sort by confidence descending
    predicted.sort(key=lambda x: x[1], reverse=True)
    return [tag for tag, _ in predicted]


def build_caption(predicted_tags: list[str], trigger_word: str) -> str:
    """Build a caption string with the trigger word prepended."""
    # Replace underscores with spaces for readability in captions
    clean_tags = [tag.replace("_", " ") for tag in predicted_tags]
    # Trigger word goes first
    return ", ".join([trigger_word] + clean_tags)


def caption_directory(
    image_dir: str,
    trigger_word: str = "nyafyi_woman",
    threshold: float = 0.35,
    batch_size: int = 8,
    model_repo: str = MODEL_REPO,
) -> int:
    """Caption all images in a directory. Returns count of images captioned."""
    image_path = Path(image_dir)
    if not image_path.exists():
        log.error("Image directory does not exist: %s", image_dir)
        return 0

    # Find all images
    image_files = sorted(
        f for f in image_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        log.error("No images found in %s", image_dir)
        return 0

    log.info("Found %d images in %s", len(image_files), image_dir)

    # Download and load model
    model_path, labels_path = download_model(model_repo)
    tags, general_indices, character_indices = load_labels(labels_path)

    # Create ONNX session (CPU only — onnxruntime-gpu lacks ARM64 wheels)
    log.info("Loading ONNX model (CPU inference)...")
    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    count = 0
    for i, img_file in enumerate(image_files):
        log.info("[%d/%d] Captioning: %s", i + 1, len(image_files), img_file.name)

        try:
            img = Image.open(img_file)
            predicted = predict_tags(session, img, tags, general_indices, threshold)
            caption = build_caption(predicted, trigger_word)

            # Save caption as .txt sidecar
            caption_file = img_file.with_suffix(".txt")
            caption_file.write_text(caption, encoding="utf-8")
            log.info("  Tags: %s", caption[:120] + "..." if len(caption) > 120 else caption)
            count += 1

        except Exception:
            log.exception("  Failed to caption %s", img_file.name)
            raise

    log.info("Captioning complete: %d/%d images captioned", count, len(image_files))
    return count


def main():
    parser = argparse.ArgumentParser(description="Auto-caption images with WD14 tagger")
    parser.add_argument(
        "--image-dir",
        default=os.environ.get("DATASET_DIR", "/dataset/images"),
        help="Directory containing images to caption",
    )
    parser.add_argument(
        "--trigger-word",
        default=os.environ.get("TRIGGER_WORD", "nyafyi_woman"),
        help="Trigger word to prepend to all captions",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Confidence threshold for tag inclusion",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("CAPTION_BATCH_SIZE", "8")),
        help="Batch size for inference",
    )
    args = parser.parse_args()

    count = caption_directory(
        image_dir=args.image_dir,
        trigger_word=args.trigger_word,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    if count == 0:
        log.error("No images were captioned!")
        sys.exit(1)


if __name__ == "__main__":
    main()
