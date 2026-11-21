# Validation, November 21, 2026

## Environment

| Item | Value |
| --- | --- |
| GPU | RTX 5090 32 GB (SM120), PCIe 4.0 x16, no display attached |
| Driver | 595.99.02 (CUDA 13.2) |
| Docker | 29.1.3 with NVIDIA Container Toolkit |
| Base image | `vllm/vllm-openai@sha256:c2dc759f…1449e` (vLLM `0.1.dev25855+g7cb5dae22`) |
| Checkpoint | `thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7` |
| Settings | Repository defaults: DSpark-7, FP8 KV, 262,144 context, 4 slots, 1,024-token chunks |

## Startup

From [startup.txt](startup.txt): 19.08 GiB of model memory, 1.81 GiB kept for
transient activations, 33,064 KV blocks (8.07 GiB), 311,557 KV tokens, 1.19x
concurrency at 262,144 tokens. Ready in about 40 s with warm JIT caches.

## Functional checks

[smoke-gateway.json](smoke-gateway.json) is the output of
`scripts/smoke.py --gpu <uuid>` through a llama-swap gateway. All ten checks passed.

| Check | Result |
| --- | --- |
| basic | Correct answers, `reasoning` field present, per-request `metrics` present |
| reasoning_effort | `none`, `low`, `high` accepted; effort changes the rendered prompt. `minimal`, `xhigh` and `max` were also accepted in a separate manual request |
| tools | Auto, named and required tool calls; result roundtrip with string `arguments` |
| tools_stream | Streamed deltas assemble into valid JSON, `finish_reason: tool_calls` |
| json_schema | Strict `response_format` JSON schema parsed and matched |
| vision | Four-color synthetic image, quadrants in the right order |
| retrieval | Exact code at 251,734 input tokens, 50.3 s |
| concurrency | Four simultaneous 31K-token retrievals, all correct, 12.4 s wall |
| overlap | 251,697-token and 31,317-token retrievals plus an image request at once; all correct; peak sampled VRAM 29,356 MiB; no OOM or preemption in the log |
| speed | 1,024-token streams at `reasoning_effort: low`: 201, 275, 200, 305 tok/s (mean acceptance 2.2–3.5 tokens) |

[gateway-metrics.json](gateway-metrics.json) is the output of a llama-swap
Activity check: streaming and non-streaming chat and text completions each
produced an Activity row with prefill rate, decode rate, cached tokens and
DSpark draft/accepted tokens.

Manual observations, not saved as artifacts: a repeated 36,529-token prompt
reused 36,480 cached tokens and dropped TTFT from 5.2 s to 49 ms. Gateway unload
removed the container and returned VRAM to 34 MiB. Killing the `docker run`
client with SIGKILL left the container running; the next start removed it, and
`stop.sh` with the stale PID file also removed it. A fresh checkout with no
environment overrides downloaded and verified the weights as a non-root user,
served on GPU 0 at `127.0.0.1:8031` and passed `smoke.py --checks basic`.

## Memory trials

Each row is a startup with `--max-model-len 262144` unless noted.

| DSpark | KV dtype | Prefill chunk | Outcome |
| --- | --- | ---: | --- |
| on | BF16 | 2,048 | Rejected: 13.41 GiB needed, 10.1–11.5 GiB available |
| off | BF16 | default | Rejected after warmup: 11.81 GiB needed, 8.7 GiB available (about 160K maximum) |
| on | BF16 | 2,048 | At 196,608: rejected, 7.8 GiB available (about 99K maximum) |
| on | FP8 | 2,048 | Rejected: 8.45 GiB needed, 7.75 GiB available (about 225K maximum) |
| off | FP8 | default | Fits: 308,006 KV tokens; 174 tok/s decode |
| off | FP8 | 1,024 | Fits: 414,516 KV tokens; 174 tok/s decode ([speed-no-dspark.json](speed-no-dspark.json)) |
| **on** | **FP8** | **1,024** | **Fits: 311,557 KV tokens; 197–350 tok/s decode** |

Before the SM120 patch, every DSpark startup that reached the draft model failed
with `CUDA error: no kernel image is available for execution on the device`.
The unpatched `torch.ops._C_cache_ops.reshape_and_cache_flash` reproduces it alone.

These are smoke checks. They do not establish BF16 equivalence, FP8 KV accuracy,
full-context quality or long-agent reliability.
