#!/usr/bin/env python3
"""Resolve a Fathom (fathom.video) URL into something yt-dlp can download.

yt-dlp's Fathom extractor only supports public share links
(``fathom.video/share/<token>``). The URLs Fathom surfaces in-app and via its
MCP/API are private ``fathom.video/calls/<id>`` pages that redirect to sign-in.

This script bridges the gap. Given any fathom.video URL it prints a JSON
resolution to stdout:

- ``/share/`` URL → passthrough (yt-dlp handles it directly, no auth).
- ``/calls/<id>`` URL → fetch the page with the user's browser cookies
  (exported via yt-dlp's ``--cookies-from-browser``), parse the Inertia
  ``data-page`` props, and return:
    1. the call's ``universalShareable.shareUrl`` when sharing is enabled
       (preferred — downstream yt-dlp needs no cookies), or
    2. the authenticated HLS ``video_url`` plus a fathom.video-only cookie
       jar to pass to ``watch.py --cookies`` (fallback for unshared calls).

Cookie hygiene: the full browser jar is written to a 0600 file inside a
private work dir, filtered down to fathom.video cookies immediately, and the
full jar is deleted before any network request is made. Cookie values are
never printed.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from urllib.request import HTTPCookieProcessor, Request, build_opener

SHARE_RE = re.compile(r"https?://(?:www\.)?fathom\.video/share/[^/?#&\s]+")
CALLS_RE = re.compile(r"https?://(?:www\.)?fathom\.video/calls/(?P<id>\d+)")
DATA_PAGE_RE = re.compile(r'data-page="([^"]+)"')
SIGN_IN_MARKER = "/users/sign_in"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
# Any URL works here — yt-dlp writes the exported jar even when extraction
# fails, and sign_in is a cheap page that never requires auth itself.
COOKIE_EXPORT_URL = "https://fathom.video/users/sign_in"


def classify(url: str) -> str:
    if SHARE_RE.match(url):
        return "share"
    if CALLS_RE.match(url):
        return "calls"
    return "other"


def export_fathom_cookies(browser: str, work_dir: Path) -> Path:
    """Export browser cookies via yt-dlp, keep only fathom.video ones.

    Returns the path to the filtered Netscape jar (0600). The full-browser
    jar is deleted before this function returns.
    """
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    full_jar = work_dir / "browser-cookies-full.txt"
    print(
        f"[fathom] exporting {browser} cookies via yt-dlp "
        "(first run may prompt for keychain access)…",
        file=sys.stderr,
    )
    subprocess.run(
        [
            "yt-dlp",
            "--cookies-from-browser", browser,
            "--cookies", str(full_jar),
            "--simulate",
            "--quiet",
            "--no-warnings",
            "--ignore-errors",
            "--",
            COOKIE_EXPORT_URL,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not full_jar.exists():
        raise SystemExit(
            f"cookie export from {browser} failed — is {browser} installed and "
            "does it have a profile with fathom.video cookies?"
        )
    full_jar.chmod(0o600)

    # Python's MozillaCookieJar treats yt-dlp's "#HttpOnly_" lines as comments
    # and silently drops them — which loses exactly the session cookies.
    raw = full_jar.read_text(encoding="utf-8", errors="replace")
    full_jar.write_text(raw.replace("#HttpOnly_", ""), encoding="utf-8")

    source = MozillaCookieJar(str(full_jar))
    source.load(ignore_discard=True, ignore_expires=True)
    filtered_path = work_dir / "fathom-cookies.txt"
    filtered = MozillaCookieJar(str(filtered_path))
    kept = 0
    for cookie in source:
        if cookie.domain.lstrip(".").endswith("fathom.video"):
            filtered.set_cookie(cookie)
            kept += 1
    filtered.save(ignore_discard=True, ignore_expires=True)
    filtered_path.chmod(0o600)
    full_jar.unlink()

    if kept == 0:
        raise SystemExit(
            f"no fathom.video cookies found in {browser} — sign in to "
            "https://fathom.video in that browser first, or pass a public "
            "share URL instead."
        )
    print(f"[fathom] kept {kept} fathom.video cookies", file=sys.stderr)
    return filtered_path


def fetch_call_page(url: str, jar_path: Path) -> tuple[str, str]:
    """GET the /calls/ page with the filtered jar. Returns (final_url, html)."""
    jar = MozillaCookieJar(str(jar_path))
    jar.load(ignore_discard=True, ignore_expires=True)
    # yt-dlp writes session cookies with expiry 0; CookieJar reads that as
    # "expired in 1970" and drops them at request time. 0 means session here.
    for cookie in jar:
        if not cookie.expires:
            cookie.expires = None
            cookie.discard = True
    opener = build_opener(HTTPCookieProcessor(jar))
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with opener.open(request, timeout=30) as response:
        return response.geturl(), response.read().decode("utf-8", errors="replace")


def extract_props(page_html: str) -> dict | None:
    """Pull the Inertia props JSON out of the page's data-page attribute."""
    match = DATA_PAGE_RE.search(page_html)
    if not match:
        return None
    try:
        page = json.loads(html.unescape(match.group(1)))
    except json.JSONDecodeError:
        return None
    props = page.get("props")
    return props if isinstance(props, dict) else None


def find_key(obj: object, key: str) -> object | None:
    """Depth-first search for the first occurrence of ``key`` in nested JSON."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = find_key(item, key)
            if found is not None:
                return found
    return None


def resolve(url: str, browser: str, work_dir: Path) -> dict:
    kind = classify(url)
    result: dict = {
        "input_url": url,
        "resolved_url": url,
        "method": "passthrough",
        "requires_cookies": False,
        "cookies_file": None,
        "title": None,
        "call_id": None,
    }
    if kind == "share":
        return result
    if kind == "other":
        # Not a URL shape we know how to improve — let yt-dlp try it as-is.
        return result

    call_id = CALLS_RE.match(url).group("id")
    result["call_id"] = call_id

    jar_path = export_fathom_cookies(browser, work_dir)
    result["cookies_file"] = str(jar_path)

    final_url, page_html = fetch_call_page(url, jar_path)
    if SIGN_IN_MARKER in final_url:
        raise SystemExit(
            f"fathom.video rejected the {browser} session (redirected to sign-in). "
            f"Sign in to https://fathom.video in {browser} and retry, or pass a "
            "public share URL."
        )

    # Calls the user can view but didn't record redirect straight to their
    # share URL — no page parsing needed.
    share_match = SHARE_RE.match(final_url)
    if share_match:
        result["resolved_url"] = share_match.group(0)
        result["method"] = "share"
        print("[fathom] call redirected to its share URL", file=sys.stderr)
        return result

    props = extract_props(page_html)
    if props is None:
        raise SystemExit(
            "could not find Inertia data-page props on the call page — Fathom "
            "may have changed their page structure. Try a share URL instead."
        )

    title = find_key(props, "title")
    result["title"] = title if isinstance(title, str) else None
    call = props.get("call") if isinstance(props.get("call"), dict) else {}

    shareable = find_key(props, "universalShareable")
    share_url = shareable.get("shareUrl") if isinstance(shareable, dict) else None
    if isinstance(share_url, str) and SHARE_RE.match(share_url):
        result["resolved_url"] = share_url
        result["method"] = "share"
        print("[fathom] resolved to share URL (no cookies needed downstream)", file=sys.stderr)
        return result

    video_url = call.get("video_url") or f"https://fathom.video/calls/{call_id}/video.m3u8"
    result["resolved_url"] = video_url
    result["method"] = "hls+cookies"
    result["requires_cookies"] = True
    print(
        "[fathom] no share URL on this call — falling back to authenticated HLS; "
        "pass --cookies to watch.py",
        file=sys.stderr,
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="fathom",
        description="Resolve a fathom.video URL into a yt-dlp-downloadable URL.",
    )
    ap.add_argument("url", help="A fathom.video /calls/ or /share/ URL")
    ap.add_argument(
        "--browser",
        default="chrome",
        help="Browser to read fathom.video cookies from (yt-dlp --cookies-from-browser "
             "syntax: chrome, safari, firefox, edge, brave, …; default: chrome)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for the filtered cookie jar (default: a private tmp dir). "
             "Delete it when done — it holds your fathom.video session.",
    )
    args = ap.parse_args()

    if args.out_dir:
        work = Path(args.out_dir).expanduser().resolve()
        work.mkdir(parents=True, exist_ok=True)
    else:
        work = Path(tempfile.mkdtemp(prefix="fathom-"))
    work.chmod(0o700)

    result = resolve(args.url, args.browser, work)
    # Drop the jar when nothing downstream needs it and we created the dir.
    if not result["requires_cookies"] and result["cookies_file"] and not args.out_dir:
        Path(result["cookies_file"]).unlink(missing_ok=True)
        result["cookies_file"] = None

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
