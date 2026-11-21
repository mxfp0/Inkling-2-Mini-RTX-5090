#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
export PORT="${PORT:-8031}"
PID_FILE="$SCRIPT_DIR/server-${PORT}.pid"
LOG_FILE="$SCRIPT_DIR/server-${PORT}.log"
die() { echo "[inkling] $*" >&2; exit 1; }

if [[ -s "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "[inkling] Already running or starting on port $PORT"; exit 0
fi
setsid nohup "$SCRIPT_DIR/serve.sh" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
disown "$SERVER_PID" 2>/dev/null || true
for _ in $(seq 1 "${STARTUP_POLLS:-900}"); do
  if [[ "$(docker inspect -f '{{.State.Running}}' "inkling-2-mini-$PORT-$SERVER_PID" 2>/dev/null)" == true ]] \
    && curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null \
    && curl -sf --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
    echo "[inkling] ${SERVED_MODEL_ID:-$MODEL_ID} ready on port $PORT (pid $SERVER_PID); log: $LOG_FILE"; exit 0
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    tail -n 80 "$LOG_FILE" >&2
    "$SCRIPT_DIR/stop.sh"
    exit 1
  fi
  sleep 1
done
"$SCRIPT_DIR/stop.sh"
die "Startup timed out; inspect $LOG_FILE"
