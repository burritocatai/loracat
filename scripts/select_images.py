#!/usr/bin/env python3
"""
select_images.py — Web UI for selecting which generated images to keep
for LoRA training.

After image collection, this script starts a local web server that shows
all generated images in a gallery.  The user selects images to keep;
unselected images are deleted when the user clicks "Continue".  The
server then shuts down and the pipeline resumes.
"""

import argparse
import logging
import os
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.serving import make_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# ── HTML template ────────────────────────────────────────────────────────────

GALLERY_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LoRACat &mdash; Image Selection</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0d1117;
    color: #c9d1d9;
    min-height: 100vh;
  }

  header {
    position: sticky; top: 0; z-index: 100;
    background: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 16px 24px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 16px;
  }
  header h1 { font-size: 20px; color: #f0f6fc; white-space: nowrap; }
  header p  { font-size: 14px; color: #8b949e; flex: 1; min-width: 200px; }

  .controls {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  .controls span {
    font-size: 14px;
    font-weight: 600;
    color: #58a6ff;
    min-width: 120px;
  }

  button {
    padding: 8px 16px;
    border: 1px solid #30363d;
    border-radius: 6px;
    background: #21262d;
    color: #c9d1d9;
    font-size: 13px;
    cursor: pointer;
    transition: background 0.15s;
  }
  button:hover { background: #30363d; }

  button.primary {
    background: #238636;
    border-color: #2ea043;
    color: #fff;
    font-weight: 600;
  }
  button.primary:hover { background: #2ea043; }
  button.primary:disabled {
    background: #1a4023;
    border-color: #1a4023;
    color: #3d6b4e;
    cursor: not-allowed;
  }

  .gallery {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
    gap: 12px;
    padding: 24px;
  }

  .card {
    position: relative;
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    border: 3px solid transparent;
    transition: border-color 0.15s, opacity 0.15s, filter 0.15s;
    background: #161b22;
  }
  .card.selected { border-color: #238636; }
  .card:not(.selected) { opacity: 0.4; filter: grayscale(0.7); }
  .card:hover { opacity: 1; filter: none; }

  .card img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    display: block;
  }

  .card .badge {
    position: absolute;
    top: 8px;
    right: 8px;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    font-weight: bold;
    pointer-events: none;
  }
  .card.selected .badge {
    background: #238636;
    color: #fff;
  }
  .card:not(.selected) .badge {
    background: rgba(200,0,0,0.7);
    color: #fff;
  }

  .card .label {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 6px 10px;
    background: rgba(0,0,0,0.7);
    font-size: 11px;
    color: #8b949e;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* Done overlay */
  #done-overlay {
    display: none;
    position: fixed; inset: 0; z-index: 200;
    background: rgba(13,17,23,0.92);
    align-items: center;
    justify-content: center;
    flex-direction: column;
    gap: 16px;
  }
  #done-overlay.visible { display: flex; }
  #done-overlay h2 { font-size: 24px; color: #58a6ff; }
  #done-overlay p  { font-size: 16px; color: #8b949e; }

  /* Confirm dialog */
  #confirm-overlay {
    display: none;
    position: fixed; inset: 0; z-index: 200;
    background: rgba(13,17,23,0.85);
    align-items: center;
    justify-content: center;
  }
  #confirm-overlay.visible { display: flex; }
  .confirm-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 32px;
    max-width: 420px;
    text-align: center;
  }
  .confirm-box h3 { font-size: 18px; color: #f0f6fc; margin-bottom: 12px; }
  .confirm-box p  { font-size: 14px; color: #8b949e; margin-bottom: 24px; }
  .confirm-box .btn-row { display: flex; gap: 12px; justify-content: center; }
</style>
</head>
<body>

<header>
  <h1>LoRACat</h1>
  <p>Select images to keep for LoRA training. Deselected images will be removed.</p>
  <div class="controls">
    <span id="count"></span>
    <button onclick="selectAll()">Select All</button>
    <button onclick="deselectAll()">Deselect All</button>
    <button class="primary" id="submit-btn" onclick="confirmSubmit()">
      Continue with Training &rarr;
    </button>
  </div>
</header>

<div class="gallery" id="gallery">
  <!-- cards injected by JS -->
</div>

<div id="confirm-overlay">
  <div class="confirm-box">
    <h3>Confirm selection</h3>
    <p id="confirm-msg"></p>
    <div class="btn-row">
      <button onclick="hideConfirm()">Cancel</button>
      <button class="primary" onclick="submitSelection()">Confirm</button>
    </div>
  </div>
</div>

<div id="done-overlay">
  <h2>Selection saved</h2>
  <p id="done-msg"></p>
  <p>Continuing with training&hellip;</p>
</div>

<script>
const IMAGES = __IMAGES_JSON__;
const selected = new Set(IMAGES);

function render() {
  const gallery = document.getElementById("gallery");
  gallery.innerHTML = "";
  IMAGES.forEach(name => {
    const card = document.createElement("div");
    card.className = "card" + (selected.has(name) ? " selected" : "");
    card.onclick = () => { toggle(name, card); };
    card.innerHTML =
      '<img src="/images/' + encodeURIComponent(name) + '" loading="lazy">' +
      '<div class="badge">' + (selected.has(name) ? "&#10003;" : "&#10005;") + '</div>' +
      '<div class="label">' + name + '</div>';
    gallery.appendChild(card);
  });
  updateCount();
}

function toggle(name, card) {
  if (selected.has(name)) selected.delete(name);
  else selected.add(name);
  card.className = "card" + (selected.has(name) ? " selected" : "");
  card.querySelector(".badge").innerHTML = selected.has(name) ? "&#10003;" : "&#10005;";
  updateCount();
}

function updateCount() {
  document.getElementById("count").textContent =
    selected.size + " / " + IMAGES.length + " selected";
  document.getElementById("submit-btn").disabled = selected.size === 0;
}

function selectAll()   { IMAGES.forEach(n => selected.add(n));    render(); }
function deselectAll() { selected.clear(); render(); }

function confirmSubmit() {
  const kept = selected.size;
  const removed = IMAGES.length - kept;
  document.getElementById("confirm-msg").textContent =
    "Keep " + kept + " image" + (kept !== 1 ? "s" : "") +
    " and remove " + removed + " image" + (removed !== 1 ? "s" : "") + "?";
  document.getElementById("confirm-overlay").classList.add("visible");
}

function hideConfirm() {
  document.getElementById("confirm-overlay").classList.remove("visible");
}

function submitSelection() {
  hideConfirm();
  document.getElementById("submit-btn").disabled = true;
  document.getElementById("submit-btn").textContent = "Saving\u2026";

  fetch("/submit", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({selected: Array.from(selected)})
  })
  .then(r => r.json())
  .then(data => {
    document.getElementById("done-msg").textContent =
      "Kept " + data.kept + " images, removed " + data.removed + ".";
    document.getElementById("done-overlay").classList.add("visible");
  })
  .catch(err => {
    alert("Error: " + err);
    document.getElementById("submit-btn").disabled = false;
    document.getElementById("submit-btn").textContent = "Continue with Training \u2192";
  });
}

render();
</script>
</body>
</html>"""

# ── Flask app ────────────────────────────────────────────────────────────────

app = Flask(__name__)

# Set at startup by main()
_image_dir: Path = Path(".")
_server = None


def _list_images() -> list[str]:
    """Return sorted list of image filenames in the image directory."""
    return sorted(
        f.name
        for f in _image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


@app.route("/")
def gallery():
    import json as _json

    images = _list_images()
    images_json = _json.dumps(images)
    html = GALLERY_HTML.replace("__IMAGES_JSON__", images_json)
    return html


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(_image_dir, filename)


@app.route("/submit", methods=["POST"])
def submit():
    data = request.get_json()
    keep = set(data.get("selected", []))
    all_images = set(_list_images())

    to_remove = all_images - keep
    for name in to_remove:
        path = _image_dir / name
        if path.is_file():
            path.unlink()
            log.info("  Removed: %s", name)

    kept = len(keep & all_images)
    removed = len(to_remove)
    log.info("Selection complete: keeping %d, removed %d", kept, removed)

    # Shut down the server after sending the response
    threading.Thread(target=_shutdown, daemon=True).start()

    return jsonify({"status": "ok", "kept": kept, "removed": removed})


def _shutdown():
    """Shut down the werkzeug server after a brief delay (lets the response flush)."""
    import time

    time.sleep(0.5)
    if _server is not None:
        _server.shutdown()


# ── Entrypoint ───────────────────────────────────────────────────────────────


def main():
    global _image_dir, _server

    parser = argparse.ArgumentParser(
        description="Web UI for selecting which generated images to keep"
    )
    parser.add_argument(
        "--image-dir",
        default=os.environ.get("DATASET_DIR", "/dataset/images"),
        help="Directory containing generated images",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("IMAGE_SELECTOR_PORT", "8080")),
        help="Port for the selection web UI (default: 8080)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    _image_dir = Path(args.image_dir)
    if not _image_dir.is_dir():
        log.error("Image directory not found: %s", _image_dir)
        sys.exit(1)

    images = _list_images()
    if not images:
        log.error("No images found in %s", _image_dir)
        sys.exit(1)

    log.info("")
    log.info("=" * 64)
    log.info("  IMAGE SELECTION")
    log.info("=" * 64)
    log.info("Found %d images in %s", len(images), _image_dir)
    log.info("")
    log.info("  Open in your browser:  http://<host>:%d", args.port)
    log.info("")
    log.info("Select images to keep, then click 'Continue with Training'.")
    log.info("Waiting for selection...")
    log.info("")

    _server = make_server(args.host, args.port, app)
    _server.serve_forever()

    log.info("Image selection server stopped")


if __name__ == "__main__":
    main()
