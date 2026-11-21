#!/usr/bin/env bash
MODEL_REPO="thinkingmachines/Inkling-2-Mini"
MODEL_REVISION="ed40e7da625f594fc9c4990f5f66be24873c22d7"
MODEL_ID="inkling-2-mini"
MODEL_DIR="${MODEL_DIR:-$HOME/models/$MODEL_REPO}"
BASE_IMAGE="vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e"
IMAGE="${IMAGE:-inkling-2-mini-rtx5090:c2dc759f}"
CACHE_VOLUME="${CACHE_VOLUME:-inkling-2-mini-cache}"
