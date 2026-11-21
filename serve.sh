#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
PORT="${PORT:-8031}"
SERVED_MODEL_ID="${SERVED_MODEL_ID:-$MODEL_ID}"
GPU="${GPU_UUID:-${GPU:-0}}"
BIND_ADDRESS="${BIND_ADDRESS:-127.0.0.1}"
die() { echo "[inkling] $*" >&2; exit 1; }

[[ "$GPU" != *,* ]] || die "Serve on exactly one GPU"
[[ -f "$MODEL_DIR/.installed-revision" && "$(cat "$MODEL_DIR/.installed-revision")" == "$MODEL_REVISION" ]] \
  || die "Run $SCRIPT_DIR/install.sh first"
[[ "$(docker image inspect -f '{{index .Config.Labels "inkling.base-image"}}' "$IMAGE" 2>/dev/null)" == "$BASE_IMAGE" ]] \
  || die "Image $IMAGE is missing or not built from the pinned base; run $SCRIPT_DIR/install.sh"
# A SIGKILLed docker client leaves its container running and holding the GPU.
# Match the client by command line: a zombie or a reused PID does not count.
while read -r NAME CLIENT_PID; do
  [[ -n "$NAME" ]] || continue
  ps -o args= -p "$CLIENT_PID" 2>/dev/null | grep -qF -- "--name $NAME " || docker rm -f "$NAME" >/dev/null
done < <(docker ps -a --filter "label=inkling.port=$PORT" --format '{{.Names}} {{.Label "inkling.client-pid"}}')
if curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null; then
  die "Port $PORT is occupied by another server"
fi
for ((ATTEMPT=0; ATTEMPT<=30; ATTEMPT++)); do
  FREE_MIB="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  [[ "$FREE_MIB" -ge "${MIN_FREE_MIB:-30000}" ]] && break
  (( ATTEMPT < 30 )) || die "Insufficient free VRAM on GPU $GPU ($FREE_MIB MiB)"
  sleep 1
done

ARGS=(/models/Inkling-2-Mini --served-model-name "$SERVED_MODEL_ID"
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 1
  --max-model-len "${MAX_MODEL_LEN:-262144}" --max-num-seqs "${MAX_NUM_SEQS:-4}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-1024}" --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
  --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling
  --enable-per-request-metrics --enable-prompt-tokens-details)
[[ -z "${GPU_UTIL:-}" ]] || ARGS+=(--gpu-memory-utilization "$GPU_UTIL")
DSPARK_TOKENS="${DSPARK_TOKENS:-7}"
[[ "$DSPARK_TOKENS" =~ ^[0-9]+$ ]] || die "DSPARK_TOKENS must be a non-negative integer"
if (( DSPARK_TOKENS > 0 )); then
  ARGS+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$DSPARK_TOKENS,\"draft_sample_method\":\"probabilistic\"}"
    --per-request-spec-decode-metrics summary)
fi

# exec keeps this PID for the docker client, so supervisors can stop inkling-2-mini-$PORT-$PID.
exec docker run --rm --name "inkling-2-mini-$PORT-$$" \
  --label "inkling.port=$PORT" --label "inkling.client-pid=$$" \
  --gpus "device=$GPU" --ipc=host -p "$BIND_ADDRESS:$PORT:8000" \
  -v "$MODEL_DIR:/models/Inkling-2-Mini:ro" -v "$CACHE_VOLUME:/root/.cache" \
  -e VLLM_USE_RUST_FRONTEND=1 -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
  -e TRITON_CACHE_DIR=/root/.cache/triton \
  "$IMAGE" "${ARGS[@]}"
