"""Download and verify the four public event logs used in the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.json"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, entry: dict, force: bool = False) -> Path:
    destination = ROOT / entry["relative_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force and sha256(destination) == entry["sha256"]:
        print(f"[ok] {name}: {destination}")
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(entry["url"], headers={"User-Agent": "Mozilla/5.0"})
    print(f"[download] {name}: {entry['url']}")
    try:
        with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    actual = sha256(temporary)
    if actual != entry["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {name}: expected {entry['sha256']}, got {actual}")
    temporary.replace(destination)
    print(f"[ok] {name}: {destination} ({destination.stat().st_size} bytes)")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="Dataset names; default downloads all")
    parser.add_argument("--force", action="store_true", help="Redownload existing files")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing files")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected = args.datasets or list(manifest)
    unknown = sorted(set(selected) - set(manifest))
    if unknown:
        parser.error(f"unknown datasets: {', '.join(unknown)}")

    failed = False
    for name in selected:
        entry = manifest[name]
        path = ROOT / entry["relative_path"]
        try:
            if args.verify_only:
                if not path.exists():
                    raise FileNotFoundError(path)
                actual = sha256(path)
                if actual != entry["sha256"]:
                    raise RuntimeError(f"SHA-256 mismatch: {actual}")
                print(f"[ok] {name}: {path}")
            else:
                download(name, entry, args.force)
        except Exception as exc:
            failed = True
            print(f"[error] {name}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
