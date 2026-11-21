#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8031}"
PID_FILE="$SCRIPT_DIR/server-${PORT}.pid"
[[ -f "$PID_FILE" ]] || exit 0
SERVER_PID="$(cat "$PID_FILE")"
[[ "$SERVER_PID" =~ ^[1-9][0-9]*$ ]] || { echo "[inkling] Invalid PID in $PID_FILE" >&2; exit 1; }
CONTAINER="inkling-2-mini-$PORT-$SERVER_PID"

docker stop -t "${STOP_TIMEOUT:-30}" "$CONTAINER" >/dev/null 2>&1 || true
if kill -0 "$SERVER_PID" 2>/dev/null; then
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  for _ in $(seq 1 10); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 1
  done
  kill -KILL "$SERVER_PID" 2>/dev/null || true
fi
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
# A replacement server may have rewritten the PID file while this one stopped.
if [[ "$(cat "$PID_FILE" 2>/dev/null)" == "$SERVER_PID" ]]; then
  rm -f "$PID_FILE"
fi
