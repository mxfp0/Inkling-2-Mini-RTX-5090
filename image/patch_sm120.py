from pathlib import Path

import vllm

ROOT = Path(vllm.__file__).parent
PATCHES = [
    (
        ROOT / "v1/attention/backends/fa_utils.py",
        "if current_platform.is_cuda():\n    from vllm._custom_ops import reshape_and_cache_flash\n",
        "if current_platform.is_cuda():\n    from vllm._custom_ops import reshape_and_cache_flash\n\n"
        "    # The image's CUDA cache ops have no SM120 kernels; DSpark draft layers use this path.\n"
        "    if current_platform.is_device_capability(120):\n"
        "        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (\n"
        "            triton_reshape_and_cache_flash as reshape_and_cache_flash,\n"
        "        )\n",
    ),
]

for path, old, new in PATCHES:
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"Patch anchor not found exactly once in {path}")
    path.write_text(text.replace(old, new))
    print(f"patched {path}")
