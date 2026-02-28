#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib import parse, request

BUFFER_SIZE = 8 * 1024 * 1024
DEFAULT_MANIFEST = Path("DICOM_16um/16um_uCT_Scanco.zip.release.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(BUFFER_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path, repo: str, tag: str) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    owner, name = repo.split("/", 1)
    url = (
        f"https://github.com/{owner}/{name}/releases/download/"
        f"{parse.quote(tag)}/{parse.quote(path.name)}"
    )
    with request.urlopen(url) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with request.urlopen(url) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(BUFFER_SIZE)
            if not chunk:
                break
            out.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and reassemble the split CT archive.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo", default="Eliguli712/fea_megalodon")
    parser.add_argument("--tag", default="ct-dicom-16um")
    parser.add_argument("--out-dir", type=Path, default=Path("DICOM_16um"))
    parser.add_argument("--keep-parts", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest, repo=args.repo, tag=args.tag)
    owner, repo = args.repo.split("/", 1)
    parts_dir = args.out_dir / "release_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    assembled_path = args.out_dir / manifest["assembled_name"]
    with assembled_path.open("wb") as out:
        for asset in manifest["assets"]:
            part_path = parts_dir / asset["name"]
            if not part_path.exists() or part_path.stat().st_size != int(asset["size"]):
                url = (
                    f"https://github.com/{owner}/{repo}/releases/download/"
                    f"{parse.quote(args.tag)}/{parse.quote(asset['name'])}"
                )
                print(f"Downloading {asset['name']}")
                download_file(url, part_path)

            digest = sha256_file(part_path)
            if digest != asset["sha256"]:
                raise SystemExit(f"Checksum mismatch for {part_path}")

            with part_path.open("rb") as fh:
                shutil.copyfileobj(fh, out, BUFFER_SIZE)

    if assembled_path.stat().st_size != int(manifest["byte_size"]):
        raise SystemExit("Reassembled archive size does not match manifest")
    if sha256_file(assembled_path) != manifest["sha256"]:
        raise SystemExit("Reassembled archive checksum does not match manifest")

    print(f"Wrote {assembled_path}")

    if not args.keep_parts:
        shutil.rmtree(parts_dir, ignore_errors=True)
        print(f"Removed {parts_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
