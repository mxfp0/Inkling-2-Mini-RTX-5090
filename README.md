# Inkling 2 Mini on a single RTX 5090

Serve [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
on one 32 GB RTX 5090 with the full 262,144-token context, DSpark speculative
decoding, tools, reasoning and image input.

The checkpoint runs as published: mixed NVFP4/MXFP8 weights, nothing requantized.
It uses vLLM's official Inkling Docker image with one small patch, because that
image ships without RTX 5090 (SM120) kernels for an op that DSpark needs.

| | |
| --- | --- |
| Context | 262,144 tokens per request, 311,557 shared KV tokens |
| Decode | 199 tok/s prose and 291 tok/s code, single stream with DSpark-7 |
| Prefill | 17,000 tok/s at 4–8K tokens, 14,500 at 32K, 8,400 at 128K |
| Concurrency | 4 scheduler slots sharing one KV pool; up to 556 (prose) and 778 (code) tok/s aggregate |
| Features | Reasoning, `reasoning_effort`, tool calls, JSON schema, images, prefix caching, per-request metrics |

## Requirements

- One NVIDIA GPU with at least 32 GB of VRAM and nothing else running on it.
  Tested on an RTX 5090 (SM120) with driver 595.99.02.
- Docker with the NVIDIA Container Toolkit (`docker run --gpus` must work).
- About 20 GB for weights and 36 GB for the image. Install needs internet access.

## Quick start

```bash
./install.sh          # pull the pinned image, download + verify weights, build the patched image
./start.sh            # serve on 127.0.0.1:8031 in the background
curl http://127.0.0.1:8031/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "Explain B-trees in two sentences."}],
  "max_tokens": 2048,
  "reasoning_effort": "low"
}'
./stop.sh
```

`./serve.sh` runs the same server in the foreground, for supervisors such as
llama-swap or systemd. The weights go to `~/models/thinkingmachines/Inkling-2-Mini`
unless `MODEL_DIR` is set; set `HF_TOKEN` if your Hugging Face access needs it.

`install.sh` runs the download inside the pinned image, checks every file size,
the SHA256 of every LFS file and the config revision, and records
`.installed-revision`. `serve.sh` refuses to start without that marker or with
an image that was not built from the pinned base.

## API notes

- **Reasoning.** Responses carry the chain of thought in `message.reasoning`.
  Set the top-level `reasoning_effort` to `none`, `minimal`, `low`, `medium`,
  `high`, `xhigh` or `max`; the default is `high`. `none` still reasons briefly,
  as the model card notes. Always set `max_tokens`: the card warns that the
  model can fall into repetitive reasoning loops.
- **Tools.** Standard OpenAI `tools`, including `tool_choice: "required"`, named
  tools and streaming. Assistant tool-call `arguments` may be JSON strings.
- **Structured output.** `response_format` with `json_schema` works.
- **Images.** `image_url` content parts (validated with base64 data URLs). The
  model does not take video or audio.
- **Sampling.** The checkpoint default is temperature 1.0.
- **Metrics.** Every response includes a `metrics` object (TTFT, mean ITL,
  tokens/s, DSpark acceptance) and `usage.prompt_tokens_details.cached_tokens`.
  Streaming clients get them in the final chunk with
  `stream_options: {"include_usage": true}`.

## Benchmarks

Measured with [sparkDash](https://github.com/MiaAI-Lab/sparkDash) on November 21,
2026, directly against the server port with the repository defaults (DSpark-7,
FP8 KV, four scheduler slots). Screenshots are in
[validation/2026-11-21/sparkdash](validation/2026-11-21/sparkdash).

Decode used 400 output tokens, temperature 0.7 and thinking on. Rates are tok/s.

| Streams | Prose TTFT | Prose aggregate | Prose per stream | Code TTFT | Code aggregate | Code per stream |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 96 ms | 198.8 | 198.8 | 89 ms | 290.7 | 290.7 |
| 2 | 245 ms | 333.0 | 175.3 | 160 ms | 547.2 | 308.1 |
| 4 | 2.24 s | 555.6 | 143.3 | 160 ms | 708.2 | 217.2 |
| 8 | 5.56 s | 543.4 | 140.7 | 3.51 s | 777.7 | 212.5 |
| 16 | 4.72 s | 467.4 | 136.0 | 5.53 s | 766.1 | 209.8 |
| 32 | 12.24 s | 494.3 | 134.0 | 9.85 s | 703.2 | 196.2 |

Code decodes faster because DSpark accepts more of its draft tokens. The server
runs four requests at a time (`MAX_NUM_SEQS=4`). Above four streams, the extra
requests wait for a free slot, so TTFT grows and aggregate throughput stops rising.
A separate run with `DSPARK_TOKENS=0` decoded 174 tok/s on one stream.

Prefill uses the server's prompt timing, with cache hits excluded.

| Context | Prompt tokens | TTFT | Prefill tok/s |
| ---: | ---: | ---: | ---: |
| 1K | 1,059 | 104 ms | 20,464 (low confidence) |
| 2K | 2,081 | 168 ms | 18,129 (low confidence) |
| 4K | 4,133 | 295 ms | 17,089 |
| 8K | 8,227 | 537 ms | 16,982 |
| 16K | 16,422 | 1.06 s | 16,369 |
| 32K | 32,804 | 2.31 s | 14,516 |
| 64K | 65,570 | 5.67 s | 11,673 |
| 128K | 131,106 | 15.67 s | 8,396 |

sparkDash marks the 1K and 2K rows as low confidence. In the smoke tests, a
251,734-token retrieval request took 50 s end to end, about 5,000 tok/s.

## Configuration

Environment variables read by `serve.sh` and `start.sh`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PORT` | `8031` | Host port |
| `BIND_ADDRESS` | `127.0.0.1` | Host address to publish on; `0.0.0.0` exposes it to the LAN |
| `GPU_UUID` / `GPU` | `0` | One GPU, by UUID or index |
| `MAX_MODEL_LEN` | `262144` | Context per request |
| `MAX_NUM_SEQS` | `4` | Scheduler slots |
| `MAX_NUM_BATCHED_TOKENS` | `1024` | Prefill chunk; 2048 does not fit 262K with DSpark |
| `KV_CACHE_DTYPE` | `fp8` | `auto` (BF16) fits about 99K with DSpark |
| `DSPARK_TOKENS` | `7` | Draft tokens; `0` disables DSpark |
| `GPU_UTIL` | unset | Passed as `--gpu-memory-utilization` when set |
| `MIN_FREE_MIB` | `30000` | Free VRAM required before starting |
| `SERVED_MODEL_ID` | `inkling-2-mini` | Model name in the API |
| `MODEL_DIR`, `IMAGE`, `CACHE_VOLUME` | see `config.sh` | Weights path, local image tag, JIT cache volume |

Change memory settings only with fresh long-context checks (see below).

## llama-swap

[examples/llama-swap.yaml](examples/llama-swap.yaml) runs `serve.sh` in the
foreground. `serve.sh` execs `docker run`, so llama-swap's `${PID}` is the
Docker client and the container is named `inkling-2-mini-${PORT}-${PID}`; the
example's `cmdStop` stops exactly that container. The container only listens
on localhost, so the entry sets `proxy` explicitly. llama-swap's Activity view
then shows prefill and decode rates, cached tokens and DSpark draft acceptance
for each request.

## How it fits in 32 GB

Weights, the DSpark drafter and the vision tower take 19.1 GiB. vLLM's
extensible KV cache then reserves the measured transient activation peak
(1.8 GiB) and gives the rest, 8.07 GiB, to KV.

Inkling has 8 global-attention layer passes per token (4 global layers, two of
them inside the 3-pass loop) plus sliding-window layers with a 1,024-token
window. In BF16, one 262,144-token request needs 13.4 GiB of KV with DSpark,
so the recipe command fails to start on this card. FP8 KV halves that. With
2,048-token prefill chunks the sliding-window and activation reservations still
leave the cache 0.7 GiB short (225K maximum); 1,024-token chunks fit 262,144 with
room for 1.19 full-length requests. The full trial table is in
[validation/2026-11-21](validation/2026-11-21/README.md).

Differences from the vLLM recipe command:

- `--kv-cache-dtype fp8`, `--max-num-batched-tokens 1024`, `--max-num-seqs 4`
- `--enable-per-request-metrics --enable-prompt-tokens-details --per-request-spec-decode-metrics summary`
- no `--privileged`; the port is published on localhost only by default
- the SM120 patch below

## The SM120 patch

`vllm/vllm-openai:inkling-2-mini` compiles vLLM's CUDA ops for SM80, SM89,
SM90 and SM100 only. The Inkling target model avoids them on SM120 (it uses
Triton and CuTe DSL kernels), but the DSpark draft layers write their KV cache
through `reshape_and_cache_flash`, which then fails with
`no kernel image is available for execution on the device`.

[image/patch_sm120.py](image/patch_sm120.py) makes vLLM's FlashAttention
backend use its own Triton `reshape_and_cache_flash` on SM120. It touches one
import in `fa_utils.py`, and the build fails if that code has changed. The
patched image is tagged `inkling-2-mini-rtx5090:c2dc759f` and labelled with
the base digest.

## Validation

`scripts/smoke.py` checks generation, reasoning effort, tools (also streamed),
JSON schema, vision, 250K-token retrieval, four parallel 32K requests, a
long+short+image overlap with VRAM sampling, and decode speed:

```bash
python3 scripts/smoke.py --base-url http://127.0.0.1:8031/v1 --gpu 0 --output report.json
```

It uses only the standard library, sends synthetic prompts and never executes
model-generated tool calls. The November 21, 2026 run passed every check; see
[validation/2026-11-21](validation/2026-11-21/README.md). These are smoke checks,
not proof of BF16 equivalence, FP8 KV accuracy, full-context quality or
long-agent reliability.

## Files

| Path | Purpose |
| --- | --- |
| `config.sh` | Pinned model revision, base image digest, local image tag |
| `install.sh` | Pull, download and verify, build |
| `serve.sh` | Foreground server (exec `docker run`) |
| `start.sh`, `stop.sh` | Background server with `server-${PORT}.pid` and `.log` |
| `image/` | Dockerfile and the SM120 patch |
| `scripts/fetch_model.py` | Verified snapshot download, run inside the image |
| `scripts/smoke.py` | Functional and speed checks |
| `examples/llama-swap.yaml` | Gateway entry |

`stop.sh` stops the container named after the recorded PID, then the client,
and removes the PID file only if a replacement server has not rewritten it.
A container left behind by a killed client is removed on the next start.

## License

The scripts and patch in this repository are MIT licensed; see [LICENSE](LICENSE).
The model is by Thinking Machines and the serving stack by the vLLM project, both
under Apache 2.0. Neither is redistributed here: `install.sh` downloads them.
