#!/usr/bin/env python3
"""
Watch a YouTube (or other) video via a local HTML5 player.
Designed for Arena Agent Mode's built-in browser.
"""

import argparse
import http.server
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def download_video(url: str, work_dir: Path) -> Path:
    """Download best video+audio with yt-dlp into work_dir."""
    work_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(work_dir / "%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bv*+ba/b",          # best video + best audio, or best combined
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--no-warnings",
        url,
    ]
    print("Downloading video...", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    # Find the downloaded file
    videos = list(work_dir.glob("*.mp4")) + list(work_dir.glob("*.mkv")) + list(work_dir.glob("*.webm"))
    if not videos:
        print("No video file found after download.", file=sys.stderr)
        sys.exit(1)
    return videos[0]

def write_player_html(work_dir: Path, video_filename: str):
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Arena Video Player</title>
  <style>
    body {{ margin: 0; background: #111; display: flex; align-items: center; justify-content: center; height: 100vh; }}
    video {{ max-width: 100%; max-height: 100%; outline: none; }}
  </style>
</head>
<body>
  <video controls autoplay>
    <source src="/{video_filename}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</body>
</html>"""
    (work_dir / "index.html").write_text(html, encoding="utf-8")

def run_server(work_dir: Path, port: int):
    os.chdir(work_dir)
    handler = http.server.SimpleHTTPRequestHandler
    with http.server.HTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"\n=== ARENA BROWSER VIEW URL ===\nhttp://127.0.0.1:{port}/\n", flush=True)
        print("Server running. Open the URL above in Arena's browser.", flush=True)
        print("Press Ctrl+C to stop.\n", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping server.", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Localhost video player for Arena Agent Mode")
    parser.add_argument("url", help="YouTube or other video URL")
    parser.add_argument("--workdir", default="video_work", help="Directory to store the video (default: video_work)")
    args = parser.parse_args()

    work_dir = Path(args.workdir).resolve()
    video_path = download_video(args.url, work_dir)
    write_player_html(work_dir, video_path.name)

    port = find_free_port()
    # Run server in foreground so the process stays alive
    run_server(work_dir, port)

if __name__ == "__main__":
    main()