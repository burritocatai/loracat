#!/usr/bin/env bash

set -euo pipefail

# --- Configuration ---
LITELLM_API_KEY="${LITELLM_API_KEY:-your_api_key_here}"
LITELLM_ENDPOINT="${LITELLM_ENDPOINT:-your_endpoint}"
LITELLM_MODEL="${LITELLM_MODEL:-your_model}"
OUTPUT_DIR="${OUTPUT_DIR:-./image_batches}"

# --- Usage ---
usage() {
  echo "Usage: LITELLM_API_KEY=your_key $0 <normal_count> <face_count> [output_file]"
  echo "  normal_count  Number of scene prompts to generate"
  echo "  face_count    Number of close-up face prompts to generate"
  echo "  output_file   Optional output filename (default: batches_TIMESTAMP.json)"
  exit 1
}

# --- Helpers ---
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

log_error() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
}

check_deps() {
  local missing=()
  for cmd in curl jq python3; do
    command -v "$cmd" &>/dev/null || missing+=("$cmd")
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    log_error "Missing required tools: ${missing[*]}"
    exit 1
  fi
}

random_seed() {
  # Generate a random integer between 10000 and 99999
  echo $(( RANDOM * RANDOM % 90000 + 10000 ))
}

# --- LiteLLM Call ---
call_litellm() {
  local batch_num="$1"
  local total="$2"

  local prompt="You are a JSON-only output machine. Output a single valid JSON object and nothing else. No code fences, no backticks, no explanations.

Generate creative and varied values for a single photorealistic image generation prompt. This is batch ${batch_num} of ${total} -- make it meaningfully different from typical outputs, vary the settings, environments, and styles.

Output ONLY this exact JSON structure with no other text:
{\"angle\": \"...\", \"expression\": \"...\", \"clothes\": \"...\", \"lighting\": \"...\", \"prompt\": \"...\"}

Rules:
- \"angle\": camera angle such as: low side, dutch, slightly high, overhead, low front, side profile, eye level, three-quarter, bird's eye, worm's eye. Pick varied ones.
- \"expression\": a short natural human expression such as: calm and focused, thoughtful, smiling faintly, serious and composed, soft waking smile, distant and pensive, amused, etc.
- \"clothes\": a specific realistic clothing description with colors and garment types. Be specific and varied.
- \"lighting\": a specific lighting condition such as: neon reflections on wet pavement, natural window, natural daylight, soft indoor, sunrise, even studio, golden hour, overcast, harsh midday sun, candlelight, blue hour, etc.
- \"prompt\": A single sentence describing a realistic photographic scene. Must instruct the subject to be placed in a specific location or pose. Must reference a specific lens (e.g. 35mm, 50mm, 24mm, 85mm) and aperture (e.g. f/1.8, f/2.0, f/2.8, f/4.0). Must mention realistic skin detail (pores, imperfections, wrinkles, blemishes). Vary the locations, poses, and lens choices each time.

Example of correct output:
{\"angle\": \"low side\", \"expression\": \"calm and focused\", \"clothes\": \"beige raincoat\", \"lighting\": \"neon reflections on wet pavement\", \"prompt\": \"Make this person stand at a crosswalk, captured with a 35mm lens at f/2.8, skin showing pores, wrinkles, and subtle blemishes.\"}"

  local payload
  payload=$(jq -n \
    --arg model "$LITELLM_MODEL" \
    --arg content "$prompt" \
    '{model: $model, messages: [{role: "user", content: $content}], temperature: 0.95, max_tokens: 512}')

local response
  local curl_err
  curl_err=$(mktemp)
  if ! response=$(curl -sf \
    --max-time 60 \
    -X POST "$LITELLM_ENDPOINT" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${LITELLM_API_KEY}" \
    -d "$payload" \
    2>"$curl_err"); then
    log_error "curl failed: $(cat "$curl_err")"
    log_error "Endpoint: $LITELLM_ENDPOINT"
    log_error "Payload: $payload"
    rm -f "$curl_err"
    return 1
  fi
  rm -f "$curl_err"
  echo "$response" | jq -r '.choices[0].message.content'
}

# --- LiteLLM Call (Face Close-ups) ---
call_litellm_face() {
  local batch_num="$1"
  local total="$2"

  local prompt="You are a JSON-only output machine. Output a single valid JSON object and nothing else. No code fences, no backticks, no explanations.

Generate creative and varied values for a single photorealistic close-up face/portrait prompt. This is face batch ${batch_num} of ${total} -- make each one meaningfully different in face angle, expression, and lighting.

Output ONLY this exact JSON structure with no other text:
{\"angle\": \"...\", \"expression\": \"...\", \"clothes\": \"...\", \"lighting\": \"...\", \"prompt\": \"...\"}

Rules:
- \"angle\": a face-specific angle such as: straight-on, three-quarter left, three-quarter right, slight left profile, slight right profile, looking up, looking down, tilted head left, tilted head right, over-the-shoulder glance, chin slightly raised, chin tucked down. Pick varied ones each time.
- \"expression\": a short natural human expression such as: calm and focused, thoughtful, smiling faintly, serious and composed, soft smile, distant and pensive, amused, slight smirk, contemplative, warm gaze, subtle surprise, relaxed, etc.
- \"clothes\": a specific realistic clothing description visible in a close-up/headshot (collar, neckline, shoulders). Be specific and varied.
- \"lighting\": a specific lighting condition such as: soft window light, Rembrandt lighting, butterfly lighting, split lighting, golden hour glow, overcast diffused, studio softbox, backlit rim light, candlelight, blue hour, natural shade, harsh directional sun, etc.
- \"prompt\": A single sentence describing a close-up face/headshot portrait. Must use a portrait lens (85mm, 105mm, or 135mm) with a wide aperture (f/1.4, f/1.8, or f/2.0). Must emphasize facial detail: skin texture, pores, fine lines, eye detail, natural imperfections. The background should be blurred/bokeh. Vary the face angles and settings each time.

Example of correct output:
{\"angle\": \"three-quarter left\", \"expression\": \"soft smile\", \"clothes\": \"navy crew-neck sweater\", \"lighting\": \"Rembrandt lighting\", \"prompt\": \"Close-up headshot portrait with a 105mm lens at f/1.8, sharp focus on the eyes, visible skin pores and fine lines, soft bokeh background, natural and intimate feel.\"}"

  local payload
  payload=$(jq -n \
    --arg model "$LITELLM_MODEL" \
    --arg content "$prompt" \
    '{model: $model, messages: [{role: "user", content: $content}], temperature: 0.95, max_tokens: 512}')

  local response
  local curl_err
  curl_err=$(mktemp)
  if ! response=$(curl -sf \
    --max-time 60 \
    -X POST "$LITELLM_ENDPOINT" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${LITELLM_API_KEY}" \
    -d "$payload" \
    2>"$curl_err"); then
    log_error "curl failed: $(cat "$curl_err")"
    log_error "Endpoint: $LITELLM_ENDPOINT"
    log_error "Payload: $payload"
    rm -f "$curl_err"
    return 1
  fi
  rm -f "$curl_err"
  echo "$response" | jq -r '.choices[0].message.content'
}

# --- JSON Repair ---
repair_json() {
  local input="$1"
  local repair_script
  repair_script=$(mktemp /tmp/imgbatch_repair_XXXXXX.py)

  cat > "$repair_script" << 'PYEOF'
import sys, re, json

raw = sys.stdin.read().strip()

# Strip code fences
raw = re.sub(r'^```json\s*', '', raw)
raw = re.sub(r'^```\s*', '', raw)
raw = re.sub(r'```\s*$', '', raw)
raw = raw.strip()

try:
    obj = json.loads(raw)
    print(json.dumps(obj))
    sys.exit(0)
except Exception:
    pass

# Fix unquoted keys
raw = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', raw)

try:
    obj = json.loads(raw)
    print(json.dumps(obj))
    sys.exit(0)
except Exception:
    pass

sys.exit(1)
PYEOF

  local result
  result=$(echo "$input" | python3 "$repair_script" 2>/dev/null) || true
  rm -f "$repair_script"
  echo "$result"
}

# --- Validate a single batch object ---
validate_batch() {
  local json="$1"
  local required_fields=("angle" "expression" "clothes" "lighting" "prompt")
  for field in "${required_fields[@]}"; do
    if ! echo "$json" | jq -e --arg f "$field" 'has($f) and (.[$f] | type == "string") and (.[$f] | length > 0)' &>/dev/null; then
      return 1
    fi
  done
  return 0
}

# --- Main ---
main() {
  check_deps

  if [[ $# -lt 2 ]]; then
    usage
  fi

  local normal_count="$1"
  local face_count="$2"
  local output_file="${3:-}"

  if ! [[ "$normal_count" =~ ^[0-9]+$ ]] || [[ "$normal_count" -lt 0 ]]; then
    log_error "normal_count must be a non-negative integer"
    usage
  fi
  if ! [[ "$face_count" =~ ^[0-9]+$ ]] || [[ "$face_count" -lt 0 ]]; then
    log_error "face_count must be a non-negative integer"
    usage
  fi
  if [[ "$normal_count" -eq 0 && "$face_count" -eq 0 ]]; then
    log_error "At least one of normal_count or face_count must be positive"
    usage
  fi

if [[ -z "$LITELLM_API_KEY" || "$LITELLM_API_KEY" == "your_api_key_here" ]]; then
    log_error "LITELLM_API_KEY is not set"
    exit 1
  fi
  if [[ -z "$LITELLM_ENDPOINT" || "$LITELLM_ENDPOINT" == "your_endpoint" ]]; then
    log_error "LITELLM_ENDPOINT is not set"
    exit 1
  fi
  if [[ -z "$LITELLM_MODEL" || "$LITELLM_MODEL" == "your_model" ]]; then
    log_error "LITELLM_MODEL is not set"
    exit 1
  fi
  log "Config: endpoint=$LITELLM_ENDPOINT model=$LITELLM_MODEL output=$output_file"

  mkdir -p "$OUTPUT_DIR"

  if [[ -z "$output_file" ]]; then
    output_file="${OUTPUT_DIR}/batches_$(date +%Y%m%d_%H%M%S).json"
  fi

  local total_count=$((normal_count + face_count))
  log "Generating $total_count prompt(s) ($normal_count scene + $face_count face) -> $output_file"

  local batches_json="[]"
  local success=0

  # --- Generate normal scene prompts ---
  if [[ $normal_count -gt 0 ]]; then
    log "--- Generating $normal_count scene prompt(s) ---"
    local i=1
    while [[ $i -le $normal_count ]]; do
      log "Requesting scene prompt $i of $normal_count..."

      local raw_response
      if ! raw_response=$(call_litellm "$i" "$normal_count"); then
        log_error "LiteLLM request failed for scene prompt $i -- skipping"
        i=$((i + 1))
        continue
      fi

      local clean_json
      clean_json=$(repair_json "$raw_response")

      if [[ -z "$clean_json" ]]; then
        log_error "Could not parse response for scene prompt $i -- skipping"
        log_error "Raw response: $raw_response"
        i=$((i + 1))
        continue
      fi

      if ! validate_batch "$clean_json"; then
        log_error "Missing required fields in scene prompt $i -- skipping"
        log_error "Parsed JSON: $clean_json"
        i=$((i + 1))
        continue
      fi

      local seed
      seed=$(random_seed)
      clean_json=$(echo "$clean_json" | jq --argjson seed "$seed" '. + {seed: $seed, count: 1}')

      batches_json=$(echo "$batches_json" | jq --argjson batch "$clean_json" '. + [$batch]')

      log "Scene prompt $i OK (seed: $seed)"
      success=$((success + 1))
      i=$((i + 1))
    done
  fi

  # --- Generate face close-up prompts ---
  if [[ $face_count -gt 0 ]]; then
    log "--- Generating $face_count face close-up prompt(s) ---"
    local j=1
    while [[ $j -le $face_count ]]; do
      log "Requesting face prompt $j of $face_count..."

      local raw_response
      if ! raw_response=$(call_litellm_face "$j" "$face_count"); then
        log_error "LiteLLM request failed for face prompt $j -- skipping"
        j=$((j + 1))
        continue
      fi

      local clean_json
      clean_json=$(repair_json "$raw_response")

      if [[ -z "$clean_json" ]]; then
        log_error "Could not parse response for face prompt $j -- skipping"
        log_error "Raw response: $raw_response"
        j=$((j + 1))
        continue
      fi

      if ! validate_batch "$clean_json"; then
        log_error "Missing required fields in face prompt $j -- skipping"
        log_error "Parsed JSON: $clean_json"
        j=$((j + 1))
        continue
      fi

      local seed
      seed=$(random_seed)
      clean_json=$(echo "$clean_json" | jq --argjson seed "$seed" '. + {seed: $seed, count: 1}')

      batches_json=$(echo "$batches_json" | jq --argjson batch "$clean_json" '. + [$batch]')

      log "Face prompt $j OK (seed: $seed)"
      success=$((success + 1))
      j=$((j + 1))
    done
  fi

  # Wrap in final structure and save
  echo "$batches_json" | jq '{batches: .}' > "$output_file"

  log "Done. $success/$total_count prompts generated ($normal_count scene + $face_count face) -> $output_file"
}

main "$@"
