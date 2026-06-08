#!/usr/bin/env python3
"""MACP cloud ingestion adapter for /watch output folders.

Registers completed /watch publication assets into MACP using the cloud
ingestion protocol:

  POST /api/brands/:brandId/ingestion/sessions
  → PUT uploads to presigned R2 URLs
  → POST /api/brands/:brandId/ingestion/sessions/:sessionId/commit

Usage:
  python3 scripts/macp_adapter.py register --folder "<UCID-folder>" [--dry-run]
    [--base-url URL] [--brand-id UUID] [--created-by ID]
    [--editorial-brief-id ID]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ADAPTER_VERSION = "0.4.2"
WATCH_ADAPTER_VERSION = "0.4.2"

TEXT_ROLES = {"transcript_md", "article_md"}
MAX_FRAMES = 100
MAX_TEXT_BYTES = 1_048_576  # 1 MB

CONTENT_TYPES: dict[str, str] = {
    ".md": "text/markdown",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

REQUIRED_ROLES = [
    ("transcript_md", "transcript.md"),
    ("article_md", "business assets/business-article.md"),
    ("article_docx", "business assets/business-article.docx"),
    ("article_pdf", "business assets/business-article.pdf"),
]


# ---------------------------------------------------------------------------
# Environment / option loading
# ---------------------------------------------------------------------------

def _load_config(args: argparse.Namespace) -> dict[str, str]:
    base_url = args.base_url or os.environ.get("MACP_BASE_URL", "")
    brand_id = args.brand_id or os.environ.get("MACP_BRAND_ID", "")
    created_by = args.created_by or os.environ.get("MACP_CREATED_BY", "")
    editorial_brief_id = args.editorial_brief_id or os.environ.get("MACP_EDITORIAL_BRIEF_ID", "")
    token = os.environ.get("MACP_REGISTRATION_TOKEN", "")

    missing = [name for name, val in [
        ("MACP_BASE_URL / --base-url", base_url),
        ("MACP_BRAND_ID / --brand-id", brand_id),
        ("MACP_CREATED_BY / --created-by", created_by),
        ("MACP_REGISTRATION_TOKEN", token),
        ("MACP_EDITORIAL_BRIEF_ID / --editorial-brief-id", editorial_brief_id),
    ] if not val]

    if missing:
        for name in missing:
            print(f"[watch] MACP adapter error: missing required config: {name}", file=sys.stderr)
        raise SystemExit(1)

    return {
        "base_url": base_url.rstrip("/"),
        "brand_id": brand_id,
        "created_by": created_by,
        "editorial_brief_id": editorial_brief_id,
        "token": token,
    }


# ---------------------------------------------------------------------------
# Local folder validation
# ---------------------------------------------------------------------------

def _validate_folder(folder: Path) -> dict:
    """Validate folder contents and return gathered facts. Raises SystemExit on failure."""
    if not folder.is_dir():
        raise SystemExit(f"[watch] MACP adapter: folder not found: {folder}")

    sentinel = folder / "business-article.REQUIRED"
    if sentinel.exists():
        raise SystemExit(
            "[watch] MACP registration skipped: business article sentinel still present; "
            "complete Step 4.5 first."
        )

    file_map: dict[str, Path] = {}
    for role, rel in REQUIRED_ROLES:
        p = folder / rel
        if not p.exists():
            raise SystemExit(f"[watch] MACP adapter: required file missing: {rel}")
        file_map[role] = p

    for role in TEXT_ROLES:
        size = file_map[role].stat().st_size
        if size > MAX_TEXT_BYTES:
            raise SystemExit(
                f"[watch] MACP adapter: {role} exceeds 1 MB text cap "
                f"({size} bytes)"
            )

    hires_paths = sorted(glob.glob(str(folder / "hires" / "frame_*.jpg")))
    if not hires_paths:
        raise SystemExit("[watch] MACP adapter: no hi-res frames found in hires/frame_*.jpg")
    if len(hires_paths) > MAX_FRAMES:
        raise SystemExit(
            f"[watch] MACP adapter: {len(hires_paths)} hi-res frames exceeds maximum {MAX_FRAMES}"
        )

    filenames = [Path(p).name for p in hires_paths]
    if len(filenames) != len(set(filenames)):
        raise SystemExit("[watch] MACP adapter: duplicate frame filename detected")

    return {
        "file_map": file_map,
        "hires_paths": [Path(p) for p in hires_paths],
    }


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _read_metadata(folder: Path) -> dict:
    info_json = folder / "download" / "video.info.json"
    raw: dict = {}
    if info_json.exists():
        try:
            raw = json.loads(info_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    source_url = raw.get("webpage_url") or raw.get("url") or ""
    extractor = (raw.get("extractor_key") or raw.get("extractor") or "").lower()
    if extractor == "youtube":
        source_platform = "youtube"
    elif extractor:
        source_platform = extractor
    else:
        source_platform = None

    if not source_url or not source_platform:
        raise SystemExit(
            "[watch] MACP adapter: cannot determine source URL or platform from "
            "download/video.info.json. Local-file sources are not supported."
        )

    external_source_id = raw.get("id") or ""
    upload_date_raw = raw.get("upload_date") or ""
    published_at = (
        f"{upload_date_raw[:4]}-{upload_date_raw[4:6]}-{upload_date_raw[6:]}"
        if len(upload_date_raw) == 8 else upload_date_raw
    )

    transcript_source = _infer_transcript_source(folder)

    hires_dir = folder / "hires"
    hires_frames = sorted(glob.glob(str(hires_dir / "frame_*.jpg")))

    return {
        "source_url": source_url,
        "source_platform": source_platform,
        "external_source_id": external_source_id,
        "title": raw.get("title") or "",
        "creator": raw.get("uploader") or raw.get("channel") or "",
        "published_at": published_at,
        "duration_seconds": raw.get("duration") or 0,
        "transcript_source": transcript_source,
        "frame_count": len(hires_frames),
        "hires_frame_count": len(hires_frames),
    }


def _infer_transcript_source(folder: Path) -> str:
    transcript_md = folder / "transcript.md"
    if not transcript_md.exists():
        return "none"
    try:
        text = transcript_md.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith("**Source:**"):
                src = line.split(":", 1)[1].strip().lower()
                if "groq" in src:
                    return "whisper_groq"
                if "openai" in src:
                    return "whisper_openai"
                if "whisper" in src:
                    return "whisper_openai"
                return "captions"
    except Exception:
        pass
    return "captions"


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------

def _build_manifest(file_map: dict[str, Path], hires_paths: list[Path]) -> list[dict]:
    manifest = []
    for role, path in file_map.items():
        ext = path.suffix.lower()
        manifest.append({
            "role": role,
            "filename": path.name,
            "content_type": CONTENT_TYPES[ext],
            "size_bytes": path.stat().st_size,
        })
    for hp in hires_paths:
        manifest.append({
            "role": "frame_hires",
            "filename": hp.name,
            "content_type": "image/jpeg",
            "size_bytes": hp.stat().st_size,
        })
    return manifest


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post_json(url: str, body: dict, token: str) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_bytes = exc.read()
        try:
            body_json = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            body_json = {"_raw": body_bytes.decode("utf-8", errors="replace")}
        return exc.code, body_json


def _put_file(put_url: str, path: Path, content_type: str) -> int:
    data = path.read_bytes()
    req = urllib.request.Request(
        put_url,
        data=data,
        method="PUT",
        headers={"Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


# ---------------------------------------------------------------------------
# Core registration flow
# ---------------------------------------------------------------------------

def _register(folder: Path, config: dict, dry_run: bool) -> None:
    facts = _validate_folder(folder)
    meta = _read_metadata(folder)
    manifest = _build_manifest(facts["file_map"], facts["hires_paths"])

    non_frame = [f for f in manifest if f["role"] != "frame_hires"]
    frame_count = len([f for f in manifest if f["role"] == "frame_hires"])

    session_payload = {
        "workflow_type": "watch_bridge",
        "source_type": "video",
        "source_url": meta["source_url"],
        "source_platform": meta["source_platform"],
        "external_source_id": meta["external_source_id"],
        "created_by": config["created_by"],
        "files": manifest,
    }

    if dry_run:
        print("[watch] MACP dry run passed")
        print(f"files: {len(non_frame)} roles, {len(manifest)} objects total")
        print(
            f"would POST: {config['base_url']}/api/brands/{config['brand_id']}/ingestion/sessions"
        )
        role_summary = ", ".join(f["role"] for f in non_frame)
        print(
            f"would upload: {role_summary}, frame_hires × {frame_count}"
        )
        print("would commit: yes")
        return

    sessions_url = f"{config['base_url']}/api/brands/{config['brand_id']}/ingestion/sessions"
    print(f"[watch] POST {sessions_url}", file=sys.stderr)
    status, resp = _post_json(sessions_url, session_payload, config["token"])

    if status not in (200, 201):
        error_msg = resp.get("error") or json.dumps(resp)
        raise SystemExit(f"[watch] MACP session creation failed ({status}): {error_msg}")

    if resp.get("idempotent_hit") and not resp.get("uploads"):
        print("[watch] MACP registration complete (existing registration found)")
        _print_entity_ids(resp)
        return

    session_id = resp["session_id"]
    uploads = resp["uploads"]

    _upload_all(facts["file_map"], facts["hires_paths"], uploads)

    source_metadata = {
        "watch_adapter_version": WATCH_ADAPTER_VERSION,
        "source_title": meta["title"],
        "source_creator": meta["creator"],
        "source_platform": meta["source_platform"],
        "external_source_id": meta["external_source_id"],
        "published_at": meta["published_at"],
        "duration_seconds": meta["duration_seconds"],
        "transcript_source": meta["transcript_source"],
        "frame_count": meta["frame_count"],
        "hires_frame_count": meta["hires_frame_count"],
        "local_output_folder_name": folder.name,
        "local_output_relpaths": {
            "transcript_md": "transcript.md",
            "article_md": "business assets/business-article.md",
            "article_docx": "business assets/business-article.docx",
            "article_pdf": "business assets/business-article.pdf",
            "hires_dir": "hires/",
        },
    }
    config_snapshot = {
        "editorial_brief_id": config["editorial_brief_id"],
        "watch_flags": {},
        "adapter": "watch",
        "adapter_version": ADAPTER_VERSION,
    }

    commit_url = (
        f"{config['base_url']}/api/brands/{config['brand_id']}"
        f"/ingestion/sessions/{session_id}/commit"
    )
    print(f"[watch] POST {commit_url}", file=sys.stderr)
    c_status, c_resp = _post_json(
        commit_url,
        {"source_metadata": source_metadata, "config_snapshot": config_snapshot},
        config["token"],
    )

    if c_status == 409 and c_resp.get("error") == "Session already registered":
        print("[watch] MACP registration complete (already registered)")
        _print_entity_ids(c_resp)
        return

    if c_status != 200:
        error_msg = c_resp.get("error") or json.dumps(c_resp)
        raise SystemExit(f"[watch] MACP commit failed ({c_status}): {error_msg}")

    print("[watch] MACP registration complete")
    _print_entity_ids(c_resp)
    admin_base = config["base_url"]
    print(f"admin: {admin_base}/admin/ingestion/sessions/{session_id}")


def _upload_all(
    file_map: dict[str, Path],
    hires_paths: list[Path],
    uploads: dict,
) -> None:
    upload_tasks: list[tuple[str, Path, str]] = []

    for role, path in file_map.items():
        slot = uploads.get(role)
        if not slot or not slot.get("put_url"):
            raise SystemExit(f"[watch] MACP: no upload slot returned for role {role}")
        content_type = CONTENT_TYPES[path.suffix.lower()]
        upload_tasks.append((slot["put_url"], path, content_type))

    frame_slots: list[dict] = uploads.get("frames") or []
    if len(frame_slots) != len(hires_paths):
        raise SystemExit(
            f"[watch] MACP: expected {len(hires_paths)} frame upload slots, "
            f"got {len(frame_slots)}"
        )
    for slot, path in zip(frame_slots, hires_paths):
        if not slot.get("put_url"):
            raise SystemExit(f"[watch] MACP: no put_url in frame slot for {path.name}")
        upload_tasks.append((slot["put_url"], path, "image/jpeg"))

    for put_url, path, content_type in upload_tasks:
        print(f"[watch] uploading {path.name} ({content_type})…", file=sys.stderr)
        http_status = _put_file(put_url, path, content_type)
        if http_status not in range(200, 300):
            raise SystemExit(
                f"[watch] MACP upload failed for {path.name}: HTTP {http_status}\n"
                "Local files are retained. Commit aborted."
            )


def _print_entity_ids(resp: dict) -> None:
    for key in ("session_id", "campaign_id", "source_pack_id", "asset_id",
                "asset_version_id", "workflow_run_id"):
        if resp.get(key):
            print(f"{key}: {resp[key]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--folder", required=True, help="Path to completed UCID output folder")
    p.add_argument("--dry-run", action="store_true", help="Validate and print payload; no network calls")
    p.add_argument("--base-url", default=None, help="MACP base URL (overrides MACP_BASE_URL)")
    p.add_argument("--brand-id", default=None, help="MACP brand ID (overrides MACP_BRAND_ID)")
    p.add_argument("--created-by", default=None, help="MACP user ID (overrides MACP_CREATED_BY)")
    p.add_argument(
        "--editorial-brief-id", default=None,
        help="Editorial brief ID (overrides MACP_EDITORIAL_BRIEF_ID)",
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="macp_adapter",
        description="Register a completed /watch UCID folder into MACP.",
    )
    sub = ap.add_subparsers(dest="command")
    register_p = sub.add_parser("register", help="Register a UCID folder with MACP")
    _add_shared_args(register_p)

    args = ap.parse_args()
    if args.command != "register":
        ap.print_help()
        return 1

    folder = Path(args.folder).expanduser().resolve()
    config = _load_config(args)
    _register(folder, config, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
