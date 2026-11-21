"""Download a pinned Hugging Face snapshot and verify sizes, LFS checksums and revision.

Runs inside the pinned vLLM image, which already ships huggingface_hub.
"""
import hashlib
from pathlib import Path
import sys

from huggingface_hub import HfApi, snapshot_download


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(repo, revision, directory):
    target = Path(directory)
    print(f"Downloading {repo}@{revision} (about 20 GB)", flush=True)
    snapshot_download(repo, revision=revision, local_dir=target, max_workers=4)
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    for sibling in info.siblings:
        path = target / sibling.rfilename
        if not path.is_file() or path.stat().st_size != sibling.size:
            raise SystemExit(f"Incomplete download: {sibling.rfilename}")
        if sibling.lfs:
            if sha256(path) != sibling.lfs.sha256:
                raise SystemExit(f"Checksum mismatch: {sibling.rfilename}")
            print(f"verified {sibling.rfilename}", flush=True)
    metadata = target / ".cache/huggingface/download/config.json.metadata"
    if metadata.read_text().splitlines()[0] != revision:
        raise SystemExit("Downloaded config revision does not match the pin")
    (target / ".installed-revision").write_text(revision + "\n")
    print(f"Checked {len(info.siblings)} files.")


if __name__ == "__main__":
    main(*sys.argv[1:])
