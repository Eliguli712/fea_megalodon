#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import shutil
import sys
from pathlib import Path
from urllib import error, parse, request

BUFFER_SIZE = 8 * 1024 * 1024
DEFAULT_MANIFEST = Path("DICOM_16um/16um_uCT_Scanco.zip.release.json")
DEFAULT_PARTS_DIR = Path(".release-tmp/ct-dicom-16um")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(BUFFER_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def split_parts(source: Path, assets: list[dict], parts_dir: Path) -> list[Path]:
    parts_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with source.open("rb") as src:
        for asset in assets:
            part_path = parts_dir / asset["name"]
            expected_size = int(asset["size"])
            expected_sha = asset["sha256"]

            digest = hashlib.sha256()
            remaining = expected_size
            with part_path.open("wb") as out:
                while remaining > 0:
                    chunk = src.read(min(BUFFER_SIZE, remaining))
                    if not chunk:
                        raise RuntimeError(f"Unexpected EOF while writing {part_path.name}")
                    out.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)

            if digest.hexdigest() != expected_sha:
                raise RuntimeError(f"Checksum mismatch for {part_path.name}")
            written.append(part_path)

        if src.read(1):
            raise RuntimeError("Source archive is larger than the committed manifest expects")

    return written


def github_json(
    url: str,
    token: str,
    method: str = "GET",
    payload: dict | None = None,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    try:
        with request.urlopen(req) as resp:
            body = resp.read()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API error {exc.code} for {url}: {detail}") from exc
    return {} if not body else json.loads(body.decode("utf-8"))


def github_delete(url: str, token: str) -> None:
    req = request.Request(
        url,
        method="DELETE",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with request.urlopen(req):
            return
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API error {exc.code} for {url}: {detail}") from exc


def github_upload(upload_url: str, asset_path: Path, token: str) -> None:
    target = f"{upload_url}?name={parse.quote(asset_path.name)}"
    parsed = parse.urlsplit(target)
    path_and_query = parsed.path
    if parsed.query:
        path_and_query = f"{path_and_query}?{parsed.query}"

    conn = http.client.HTTPSConnection(parsed.netloc)
    conn.putrequest("POST", path_and_query)
    conn.putheader("Accept", "application/vnd.github+json")
    conn.putheader("Authorization", f"Bearer {token}")
    conn.putheader("Content-Type", "application/octet-stream")
    conn.putheader("Content-Length", str(asset_path.stat().st_size))
    conn.putheader("X-GitHub-Api-Version", "2022-11-28")
    conn.endheaders()

    with asset_path.open("rb") as fh:
        while True:
            chunk = fh.read(BUFFER_SIZE)
            if not chunk:
                break
            conn.send(chunk)

    resp = conn.getresponse()
    body = resp.read().decode("utf-8", errors="replace")
    if not 200 <= resp.status < 300:
        raise RuntimeError(f"GitHub upload error {resp.status} for {asset_path.name}: {body}")
    conn.close()


def get_or_create_release(owner: str, repo: str, tag: str, token: str) -> dict:
    base = f"https://api.github.com/repos/{owner}/{repo}/releases"
    tag_url = f"{base}/tags/{parse.quote(tag)}"
    try:
        return github_json(tag_url, token=token)
    except RuntimeError as exc:
        if " 404 " not in f" {exc} ":
            raise
    return github_json(
        base,
        token=token,
        method="POST",
        payload={
            "tag_name": tag,
            "name": tag,
            "draft": False,
            "prerelease": False,
            "body": (
                "Split release assets for the raw 16um uCT Scanco archive.\n\n"
                "Use `scripts/fetch_dicom_16um_release.py` with the committed manifest "
                "to download and reassemble the dataset."
            ),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish split CT archive assets to a GitHub release.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS_DIR)
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--keep-parts", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    owner, repo = manifest["repo"].split("/", 1)
    source = Path(manifest["source_path"])
    manifest_asset = args.manifest

    if not source.exists():
        raise SystemExit(f"Missing source archive: {source}")
    if source.stat().st_size != int(manifest["byte_size"]):
        raise SystemExit("Source archive size does not match the committed manifest")

    parts = split_parts(source, manifest["assets"], args.parts_dir)
    print(f"Prepared {len(parts)} parts in {args.parts_dir}")

    if args.skip_upload:
        print("Skipping upload.")
        return 0

    if not args.token:
        raise SystemExit("Set GITHUB_TOKEN or pass --token to upload release assets")

    release = get_or_create_release(owner, repo, manifest["release_tag"], args.token)
    upload_url = release["upload_url"].split("{", 1)[0]
    assets_by_name = {asset["name"]: asset for asset in release.get("assets", [])}

    upload_paths = parts + [manifest_asset]
    for path in upload_paths:
        existing = assets_by_name.get(path.name)
        if existing:
            if not args.overwrite:
                print(f"Skipping existing asset: {path.name}")
                continue
            github_delete(existing["url"], args.token)
        print(f"Uploading {path.name}")
        github_upload(upload_url, path, args.token)

    print(f"Release ready: https://github.com/{owner}/{repo}/releases/tag/{manifest['release_tag']}")

    if not args.keep_parts:
        shutil.rmtree(args.parts_dir, ignore_errors=True)
        print(f"Removed temporary parts from {args.parts_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
