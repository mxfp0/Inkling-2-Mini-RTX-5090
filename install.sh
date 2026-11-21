#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
command -v docker >/dev/null || { echo "[inkling] Docker is required" >&2; exit 1; }

docker pull "$BASE_IMAGE"
mkdir -p "$MODEL_DIR"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp ${HF_TOKEN:+-e HF_TOKEN} \
  -v "$MODEL_DIR:/model" -v "$SCRIPT_DIR/scripts/fetch_model.py:/opt/fetch_model.py:ro" \
  --entrypoint python3 "$BASE_IMAGE" -I /opt/fetch_model.py "$MODEL_REPO" "$MODEL_REVISION" /model
docker build --pull=false --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --label "inkling.base-image=$BASE_IMAGE" -t "$IMAGE" "$SCRIPT_DIR/image"
docker volume create "$CACHE_VOLUME" >/dev/null
echo "[inkling] installed $MODEL_REPO@$MODEL_REVISION into $MODEL_DIR and built $IMAGE"
