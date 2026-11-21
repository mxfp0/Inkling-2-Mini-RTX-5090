#!/usr/bin/env python3
"""Functional smoke checks for an OpenAI-compatible Inkling 2 Mini endpoint.

Standard library only. Prompts are synthetic, model-generated tool calls are
never executed, and only short answers are kept in the report.
"""
import argparse
import base64
import json
import random
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
            "required": ["city"],
        },
    },
}


class Client:
    def __init__(self, base_url, model, timeout):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _request(self, body):
        body = {"model": self.model, **body}
        return urllib.request.Request(
            f"{self.base_url}/chat/completions", json.dumps(body).encode(),
            {"Content-Type": "application/json"})

    def chat(self, **body):
        started = time.monotonic()
        try:
            with urllib.request.urlopen(self._request(body), timeout=self.timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"HTTP {error.code}: {error.read().decode()[:500]}") from None
        data["_seconds"] = time.monotonic() - started
        return data

    def stream(self, **body):
        body.update(stream=True, stream_options={"include_usage": True})
        started = time.monotonic()
        first = None
        chunks, usage, metrics = [], None, None
        with urllib.request.urlopen(self._request(body), timeout=self.timeout) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                data = json.loads(line[5:])
                usage = data.get("usage") or usage
                metrics = data.get("metrics") or metrics
                for choice in data.get("choices", []):
                    delta = choice.get("delta", {})
                    if first is None and any(delta.get(k) for k in ("content", "reasoning", "reasoning_content", "tool_calls")):
                        first = time.monotonic()
                    chunks.append(choice)
        return {"chunks": chunks, "usage": usage, "metrics": metrics, "started": started,
                "first": first, "finished": time.monotonic()}


def message(data):
    return data["choices"][0]["message"]


def reasoning(msg):
    return msg.get("reasoning") or msg.get("reasoning_content") or ""


def quadrant_png(size=256):
    colors = {(0, 0): (220, 30, 30), (1, 0): (30, 180, 40), (0, 1): (30, 60, 220), (1, 1): (240, 220, 20)}
    half = size // 2
    rows = b"".join(
        b"\x00" + b"".join(bytes(colors[(x >= half, y >= half)]) for x in range(size)) for y in range(size))

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def haystack(words, seed, code):
    rng = random.Random(seed)
    subjects = ["The archivist", "A courier", "The harbor master", "An engineer", "The botanist", "A pilot"]
    verbs = ["catalogued", "inspected", "repaired", "measured", "documented", "relocated"]
    objects = ["the north bridge", "a copper valve", "the signal tower", "a brass compass", "the river lock"]
    sentences, count, needle_at = [], 0, int(words * 0.6)
    while count < words:
        sentence = (f"Record {len(sentences)}: {rng.choice(subjects)} {rng.choice(verbs)} "
                    f"{rng.choice(objects)} on day {rng.randint(1, 9999)}.")
        sentences.append(sentence)
        count += len(sentence.split())
        if needle_at and count >= needle_at:
            sentences.append(f"IMPORTANT: the vault access code is {code}.")
            needle_at = 0
    return " ".join(sentences)


def check_basic(client):
    data = client.chat(messages=[{"role": "user", "content": "What is 17*23? Then name the capital of Australia. Answer briefly."}],
                       max_tokens=2048)
    msg = message(data)
    content = msg.get("content") or ""
    assert "391" in content and "Canberra" in content, content
    assert reasoning(msg), "missing reasoning field"
    assert data.get("metrics"), "missing per-request metrics"
    return {"answer": content[:80], "reasoning_chars": len(reasoning(msg))}


def check_reasoning_effort(client):
    results = {}
    for effort in ("none", "low", "high"):
        data = client.chat(messages=[{"role": "user", "content": "How many r letters are in strawberry? Answer with a number."}],
                           max_tokens=3000, reasoning_effort=effort)
        content = message(data).get("content") or ""
        assert "3" in content, (effort, content)
        results[effort] = {"prompt_tokens": data["usage"]["prompt_tokens"],
                           "completion_tokens": data["usage"]["completion_tokens"]}
    assert results["none"]["prompt_tokens"] != results["high"]["prompt_tokens"], "effort did not reach the template"
    return results


def check_tools(client):
    messages = [{"role": "user", "content": "What's the weather in Lisbon in celsius? Use the tool."}]
    data = client.chat(messages=messages, tools=[WEATHER_TOOL], max_tokens=2048)
    assert data["choices"][0]["finish_reason"] == "tool_calls", data["choices"][0]
    call = message(data)["tool_calls"][0]
    arguments = json.loads(call["function"]["arguments"])
    assert call["function"]["name"] == "get_weather" and arguments["city"] == "Lisbon", call
    messages += [{"role": "assistant", "content": message(data).get("content"), "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": call["id"],
                  "content": json.dumps({"city": "Lisbon", "temp_c": 19, "conditions": "sunny"})}]
    followup = message(client.chat(messages=messages, tools=[WEATHER_TOOL], max_tokens=2048)).get("content") or ""
    assert "19" in followup, followup
    named = client.chat(messages=[{"role": "user", "content": "Check the weather in Paris."}], tools=[WEATHER_TOOL],
                        tool_choice={"type": "function", "function": {"name": "get_weather"}}, max_tokens=1024)
    required = client.chat(messages=[{"role": "user", "content": "Check the weather in Rome."}], tools=[WEATHER_TOOL],
                           tool_choice="required", max_tokens=1024)
    for data in (named, required):
        json.loads(message(data)["tool_calls"][0]["function"]["arguments"])
    return {"arguments": arguments, "followup": followup[:80]}


def check_tools_stream(client):
    result = client.stream(messages=[{"role": "user", "content": "What's the weather in Oslo? Use the tool."}],
                           tools=[WEATHER_TOOL], max_tokens=2048)
    calls, finish = {}, None
    for choice in result["chunks"]:
        finish = choice.get("finish_reason") or finish
        for delta in choice.get("delta", {}).get("tool_calls") or []:
            call = calls.setdefault(delta.get("index", 0), {"name": "", "arguments": ""})
            function = delta.get("function") or {}
            call["name"] += function.get("name") or ""
            call["arguments"] += function.get("arguments") or ""
    assert finish == "tool_calls", finish
    assert len(calls) == 1 and calls[0]["name"] == "get_weather", calls
    arguments = json.loads(calls[0]["arguments"])
    assert arguments["city"] == "Oslo", arguments
    assert result["metrics"], "missing streamed metrics"
    return {"arguments": arguments}


def check_json_schema(client):
    schema = {"type": "object", "additionalProperties": False, "required": ["city", "country", "population_millions"],
              "properties": {"city": {"type": "string"}, "country": {"type": "string"},
                             "population_millions": {"type": "number"}}}
    data = client.chat(messages=[{"role": "user", "content": "Give the largest city in Japan as JSON."}],
                       response_format={"type": "json_schema", "json_schema": {"name": "city", "schema": schema, "strict": True}},
                       max_tokens=2048)
    parsed = json.loads(message(data)["content"])
    assert set(parsed) == set(schema["required"]) and isinstance(parsed["population_millions"], (int, float)), parsed
    return {"parsed": parsed}


def check_vision(client):
    content = [{"type": "image_url", "image_url": {"url": quadrant_png()}},
               {"type": "text", "text": "Name the colors of the top-left, top-right, bottom-left and bottom-right quadrants, in that order."}]
    answer = (message(client.chat(messages=[{"role": "user", "content": content}], max_tokens=2048)).get("content") or "").lower()
    positions = [answer.find(color) for color in ("red", "green", "blue", "yellow")]
    assert all(p >= 0 for p in positions) and positions == sorted(positions), answer
    return {"answer": answer[:100]}


def retrieval(client, words, seed, code, max_tokens):
    prompt = haystack(words, seed, code) + "\n\nWhat is the vault access code mentioned in the records? Reply with the code only."
    data = client.chat(messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens, reasoning_effort="low")
    answer = (message(data).get("content") or "").strip()
    return {"prompt_tokens": data["usage"]["prompt_tokens"], "seconds": round(data["_seconds"], 1),
            "found": code in answer, "answer": answer[:40]}


def check_retrieval(client, words):
    result = retrieval(client, words, 1, "QX-4471-ZETA", 2048)
    assert result["found"], result
    return result


def run_parallel(jobs):
    results, errors = {}, {}

    def run(name, job):
        try:
            results[name] = job()
        except Exception as error:
            errors[name] = repr(error)

    threads = [threading.Thread(target=run, args=item) for item in jobs.items()]
    started = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results, errors, time.monotonic() - started


class VramSampler:
    def __init__(self, gpu):
        self.gpu, self.peak, self.running = gpu, None, False
        self.available = gpu is not None and shutil.which("nvidia-smi") is not None

    def __enter__(self):
        if self.available:
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
        return self

    def _loop(self):
        while self.running:
            output = subprocess.run(["nvidia-smi", "-i", self.gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True).stdout.strip()
            if output.isdigit():
                self.peak = max(self.peak or 0, int(output))
            time.sleep(0.5)

    def __exit__(self, *exc):
        self.running = False


def check_concurrency(client, words, streams):
    jobs = {f"stream{i}": (lambda i=i: retrieval(client, words, 10 + i, f"CODE-{i}{7 * i + 3}-K", 1024)) for i in range(streams)}
    results, errors, wall = run_parallel(jobs)
    assert not errors and all(r["found"] for r in results.values()), (results, errors)
    return {"wall_seconds": round(wall, 1), "streams": results}


def check_overlap(client, long_words, short_words, gpu):
    jobs = {"long": lambda: retrieval(client, long_words, 2, "LONG-9182-K", 1024),
            "short": lambda: retrieval(client, short_words, 3, "SHORT-5521-K", 1024),
            "vision": lambda: check_vision(client)}
    with VramSampler(gpu) as sampler:
        results, errors, wall = run_parallel(jobs)
    assert not errors, errors
    assert results["long"]["found"] and results["short"]["found"], results
    return {"wall_seconds": round(wall, 1), "peak_vram_mib": sampler.peak, **results}


def check_speed(client, rounds):
    prompts = ["Write a long, detailed technical essay about how B-trees handle insertion and deletion.",
               "Implement a complete Python LRU cache class with docstrings, type hints and tests."]
    samples = []
    for _ in range(rounds):
        for prompt in prompts:
            result = client.stream(messages=[{"role": "user", "content": prompt}], max_tokens=1024, reasoning_effort="low")
            tokens = result["usage"]["completion_tokens"]
            spec = (result["metrics"] or {}).get("speculative_decoding") or {}
            samples.append({"completion_tokens": tokens,
                            "ttft_seconds": round(result["first"] - result["started"], 3),
                            "decode_tokens_per_second": round((tokens - 1) / (result["finished"] - result["first"]), 1),
                            "mean_acceptance_length": spec.get("mean_acceptance_length")})
    return {"samples": samples}


CHECKS = ["basic", "reasoning_effort", "tools", "tools_stream", "json_schema", "vision",
          "retrieval", "concurrency", "overlap", "speed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8031/v1")
    parser.add_argument("--model", default="inkling-2-mini")
    parser.add_argument("--checks", nargs="+", choices=CHECKS, default=CHECKS)
    parser.add_argument("--retrieval-words", type=int, default=149000,
                        help="about 250K tokens with the Inkling tokenizer")
    parser.add_argument("--concurrent-words", type=int, default=19000, help="about 32K tokens per stream")
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--speed-rounds", type=int, default=2)
    parser.add_argument("--gpu", help="nvidia-smi index or UUID to sample during the overlap check")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", help="write a JSON report")
    args = parser.parse_args()

    client = Client(args.base_url, args.model, args.timeout)
    runners = {
        "basic": lambda: check_basic(client),
        "reasoning_effort": lambda: check_reasoning_effort(client),
        "tools": lambda: check_tools(client),
        "tools_stream": lambda: check_tools_stream(client),
        "json_schema": lambda: check_json_schema(client),
        "vision": lambda: check_vision(client),
        "retrieval": lambda: check_retrieval(client, args.retrieval_words),
        "concurrency": lambda: check_concurrency(client, args.concurrent_words, args.streams),
        "overlap": lambda: check_overlap(client, args.retrieval_words, args.concurrent_words, args.gpu),
        "speed": lambda: check_speed(client, args.speed_rounds),
    }
    report = {"base_url": args.base_url, "model": args.model,
              "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checks": {}}
    failed = False
    for name in args.checks:
        try:
            result = {"passed": True, **runners[name]()}
        except Exception as error:
            result, failed = {"passed": False, "error": repr(error)[:1000]}, True
        report["checks"][name] = result
        print(f"{'PASS' if result['passed'] else 'FAIL'} {name}: {json.dumps(result)[:300]}", flush=True)
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
