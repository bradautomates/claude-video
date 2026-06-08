"""Unit tests for scripts/macp_adapter.py.

Run with:  python3 -m pytest tests/test_macp_adapter.py
       or: python3 -m unittest tests/test_macp_adapter.py
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
import unittest
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

# Allow importing from scripts/
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import macp_adapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ucid_folder(tmp_path: Path, *, sentinel: bool = False, frames: int = 3) -> Path:
    folder = tmp_path / "UCID-0001 2026-01-15 Test Video"
    folder.mkdir(parents=True)

    download_dir = folder / "download"
    download_dir.mkdir()
    info = {
        "id": "abc123",
        "title": "Test Video",
        "uploader": "Test Channel",
        "upload_date": "20260115",
        "duration": 600,
        "webpage_url": "https://www.youtube.com/watch?v=abc123",
        "extractor_key": "Youtube",
    }
    (download_dir / "video.info.json").write_text(json.dumps(info), encoding="utf-8")

    (folder / "transcript.md").write_text(
        "# Transcript\n**Source:** captions\n---\nHello world\n", encoding="utf-8"
    )

    assets = folder / "business assets"
    assets.mkdir()
    (assets / "business-article.md").write_text("# Article\nContent.", encoding="utf-8")
    (assets / "business-article.docx").write_bytes(b"PK\x03\x04docx-bytes")
    (assets / "business-article.pdf").write_bytes(b"%PDF-1.4 pdf-bytes")

    hires = folder / "hires"
    hires.mkdir()
    for i in range(1, frames + 1):
        (hires / f"frame_{i:04d}.jpg").write_bytes(b"\xff\xd8\xff\xe0jpeg")

    if sentinel:
        (folder / "business-article.REQUIRED").write_text("pending", encoding="utf-8")

    return folder


# ---------------------------------------------------------------------------
# T1 — Builds manifest from complete UCID folder
# ---------------------------------------------------------------------------

class TestBuildManifest(unittest.TestCase):
    def test_required_roles_present(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp), frames=3)
            facts = macp_adapter._validate_folder(folder)
            manifest = macp_adapter._build_manifest(facts["file_map"], facts["hires_paths"])
            roles = [m["role"] for m in manifest]
            for role in ("transcript_md", "article_md", "article_docx", "article_pdf"):
                self.assertIn(role, roles)
            self.assertEqual(roles.count("frame_hires"), 3)

    def test_sizes_correct(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp), frames=2)
            facts = macp_adapter._validate_folder(folder)
            manifest = macp_adapter._build_manifest(facts["file_map"], facts["hires_paths"])
            for entry in manifest:
                self.assertGreater(entry["size_bytes"], 0)


# ---------------------------------------------------------------------------
# T2 — Missing business-article.pdf fails before session creation
# ---------------------------------------------------------------------------

class TestMissingPdf(unittest.TestCase):
    def test_missing_pdf_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp))
            (folder / "business assets" / "business-article.pdf").unlink()
            with self.assertRaises(SystemExit):
                macp_adapter._validate_folder(folder)


# ---------------------------------------------------------------------------
# T3 — Sentinel still present fails before session creation
# ---------------------------------------------------------------------------

class TestSentinelPresent(unittest.TestCase):
    def test_sentinel_blocks_registration(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp), sentinel=True)
            with self.assertRaises(SystemExit) as ctx:
                macp_adapter._validate_folder(folder)
            self.assertIn("sentinel", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# T4 — More than 100 hi-res frames fails before session creation
# ---------------------------------------------------------------------------

class TestFrameLimitHigh(unittest.TestCase):
    def test_101_frames_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp), frames=101)
            with self.assertRaises(SystemExit) as ctx:
                macp_adapter._validate_folder(folder)
            self.assertIn("100", str(ctx.exception))


# ---------------------------------------------------------------------------
# T5 — Zero hi-res frames fails before session creation
# ---------------------------------------------------------------------------

class TestFrameLimitLow(unittest.TestCase):
    def test_zero_frames_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp), frames=0)
            with self.assertRaises(SystemExit) as ctx:
                macp_adapter._validate_folder(folder)
            self.assertIn("frame", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# T6 — Missing env var fails with redacted diagnostic
# ---------------------------------------------------------------------------

class TestMissingEnvVar(unittest.TestCase):
    def test_missing_token_raises(self):
        import argparse
        import tempfile
        args = argparse.Namespace(
            base_url="https://example.com",
            brand_id="brand-uuid",
            created_by="user-uuid",
            editorial_brief_id="brief-uuid",
        )
        # MACP_REGISTRATION_TOKEN not set
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("MACP_")}
        with patch.dict(os.environ, clean_env, clear=True):
            with self.assertRaises(SystemExit):
                macp_adapter._load_config(args)

    def test_error_message_does_not_contain_value(self):
        import argparse
        args = argparse.Namespace(
            base_url=None, brand_id=None, created_by=None, editorial_brief_id=None,
        )
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("MACP_")}
        stderr_capture = StringIO()
        with patch.dict(os.environ, clean_env, clear=True):
            with patch("sys.stderr", stderr_capture):
                try:
                    macp_adapter._load_config(args)
                except SystemExit:
                    pass
        output = stderr_capture.getvalue()
        self.assertNotIn("secret", output.lower())
        self.assertNotIn("password", output.lower())


# ---------------------------------------------------------------------------
# T7 — Dry-run makes no HTTP calls and prints redacted summary
# ---------------------------------------------------------------------------

class TestDryRun(unittest.TestCase):
    def test_no_http_calls_in_dry_run(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp))
            config = {
                "base_url": "https://macp.example.com",
                "brand_id": "brand-uuid",
                "created_by": "user-uuid",
                "editorial_brief_id": "brief-uuid",
                "token": "tok-secret",
            }
            with patch("urllib.request.urlopen") as mock_open:
                with patch("sys.stdout", new_callable=StringIO) as mock_out:
                    macp_adapter._register(folder, config, dry_run=True)
                mock_open.assert_not_called()
            output = mock_out.getvalue()
            self.assertIn("dry run", output.lower())
            self.assertNotIn("tok-secret", output)

    def test_dry_run_prints_role_summary(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp), frames=3)
            config = {
                "base_url": "https://macp.example.com",
                "brand_id": "brand-uuid",
                "created_by": "user-uuid",
                "editorial_brief_id": "brief-uuid",
                "token": "tok",
            }
            with patch("sys.stdout", new_callable=StringIO) as mock_out:
                macp_adapter._register(folder, config, dry_run=True)
            output = mock_out.getvalue()
            self.assertIn("frame_hires", output)
            self.assertIn("transcript_md", output)


# ---------------------------------------------------------------------------
# T8 — Session create success → uploads all slots and commits
# ---------------------------------------------------------------------------

class TestSessionCreateSuccess(unittest.TestCase):
    def test_uploads_all_slots_then_commits(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp), frames=2)
            config = {
                "base_url": "https://macp.example.com",
                "brand_id": "brand-uuid",
                "created_by": "user-uuid",
                "editorial_brief_id": "brief-uuid",
                "token": "tok",
            }
            session_resp = {
                "session_id": "sess-001",
                "expires_at": "2026-01-16T00:00:00Z",
                "idempotent_hit": False,
                "uploads": {
                    "transcript_md": {"r2_key": "k1", "put_url": "https://r2.example.com/k1", "expires_in": 900},
                    "article_md": {"r2_key": "k2", "put_url": "https://r2.example.com/k2", "expires_in": 900},
                    "article_docx": {"r2_key": "k3", "put_url": "https://r2.example.com/k3", "expires_in": 900},
                    "article_pdf": {"r2_key": "k4", "put_url": "https://r2.example.com/k4", "expires_in": 900},
                    "frames": [
                        {"filename": "frame_0001.jpg", "r2_key": "f1", "put_url": "https://r2.example.com/f1", "expires_in": 900},
                        {"filename": "frame_0002.jpg", "r2_key": "f2", "put_url": "https://r2.example.com/f2", "expires_in": 900},
                    ],
                },
            }
            commit_resp = {
                "campaign_id": "camp-1",
                "source_pack_id": "spck-1",
                "asset_id": "asst-1",
                "asset_version_id": "asvr-1",
                "workflow_run_id": "wfr-1",
            }
            put_calls = []

            def fake_put(put_url, path, content_type):
                put_calls.append((put_url, path.name, content_type))
                return 200

            with patch.object(macp_adapter, "_post_json", side_effect=[(201, session_resp), (200, commit_resp)]):
                with patch.object(macp_adapter, "_put_file", side_effect=fake_put):
                    with patch("sys.stdout", new_callable=StringIO):
                        macp_adapter._register(folder, config, dry_run=False)

            self.assertEqual(len(put_calls), 6)  # 4 roles + 2 frames
            put_names = [name for _, name, _ in put_calls]
            self.assertIn("frame_0001.jpg", put_names)
            self.assertIn("frame_0002.jpg", put_names)


# ---------------------------------------------------------------------------
# T9 — Completed idempotent hit → no upload/commit; admin link printed
# ---------------------------------------------------------------------------

class TestCompletedIdempotentHit(unittest.TestCase):
    def test_no_upload_or_commit_on_hit(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp))
            config = {
                "base_url": "https://macp.example.com",
                "brand_id": "brand-uuid",
                "created_by": "user-uuid",
                "editorial_brief_id": "brief-uuid",
                "token": "tok",
            }
            hit_resp = {
                "idempotent_hit": True,
                "campaign_id": "camp-existing",
                "asset_id": "asst-existing",
                "workflow_run_id": "wfr-existing",
            }
            put_calls = []
            post_calls = []

            def fake_post(url, body, token):
                post_calls.append(url)
                return (200, hit_resp)

            with patch.object(macp_adapter, "_post_json", side_effect=fake_post):
                with patch.object(macp_adapter, "_put_file", side_effect=lambda *a: put_calls.append(a) or 200):
                    with patch("sys.stdout", new_callable=StringIO) as mock_out:
                        macp_adapter._register(folder, config, dry_run=False)

            self.assertEqual(len(put_calls), 0)
            self.assertEqual(len(post_calls), 1)
            self.assertIn("camp-existing", mock_out.getvalue())


# ---------------------------------------------------------------------------
# T10 — 409 manifest mismatch → clear failure, no upload/commit
# ---------------------------------------------------------------------------

class TestManifestMismatch(unittest.TestCase):
    def test_409_mismatch_raises_before_upload(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp))
            config = {
                "base_url": "https://macp.example.com",
                "brand_id": "brand-uuid",
                "created_by": "user-uuid",
                "editorial_brief_id": "brief-uuid",
                "token": "tok",
            }
            with patch.object(macp_adapter, "_post_json", return_value=(409, {"error": "Active session exists with different manifest"})):
                with patch.object(macp_adapter, "_put_file") as mock_put:
                    with self.assertRaises(SystemExit) as ctx:
                        macp_adapter._register(folder, config, dry_run=False)
                    mock_put.assert_not_called()
            self.assertIn("409", str(ctx.exception))


# ---------------------------------------------------------------------------
# T11 — Upload failure → stops before commit; local files retained
# ---------------------------------------------------------------------------

class TestUploadFailure(unittest.TestCase):
    def test_upload_failure_aborts_before_commit(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp), frames=1)
            config = {
                "base_url": "https://macp.example.com",
                "brand_id": "brand-uuid",
                "created_by": "user-uuid",
                "editorial_brief_id": "brief-uuid",
                "token": "tok",
            }
            session_resp = {
                "session_id": "sess-001",
                "idempotent_hit": False,
                "uploads": {
                    "transcript_md": {"r2_key": "k1", "put_url": "https://r2.example.com/k1", "expires_in": 900},
                    "article_md": {"r2_key": "k2", "put_url": "https://r2.example.com/k2", "expires_in": 900},
                    "article_docx": {"r2_key": "k3", "put_url": "https://r2.example.com/k3", "expires_in": 900},
                    "article_pdf": {"r2_key": "k4", "put_url": "https://r2.example.com/k4", "expires_in": 900},
                    "frames": [{"filename": "frame_0001.jpg", "r2_key": "f1", "put_url": "https://r2.example.com/f1", "expires_in": 900}],
                },
            }
            commit_calls = []

            def fake_post(url, body, token):
                if "commit" in url:
                    commit_calls.append(url)
                    return (200, {})
                return (201, session_resp)

            with patch.object(macp_adapter, "_post_json", side_effect=fake_post):
                with patch.object(macp_adapter, "_put_file", return_value=503):
                    with self.assertRaises(SystemExit) as ctx:
                        macp_adapter._register(folder, config, dry_run=False)
            self.assertEqual(len(commit_calls), 0)
            self.assertIn("503", str(ctx.exception))
            # local files retained
            self.assertTrue((folder / "transcript.md").exists())


# ---------------------------------------------------------------------------
# T12 — Commit success → prints IDs and admin link
# ---------------------------------------------------------------------------

class TestCommitSuccess(unittest.TestCase):
    def test_prints_ids_and_admin_link(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp), frames=1)
            config = {
                "base_url": "https://macp-blond.vercel.app",
                "brand_id": "brand-uuid",
                "created_by": "user-uuid",
                "editorial_brief_id": "brief-uuid",
                "token": "tok",
            }
            session_resp = {
                "session_id": "sess-abc",
                "idempotent_hit": False,
                "uploads": {
                    "transcript_md": {"r2_key": "k1", "put_url": "https://r2.example.com/k1", "expires_in": 900},
                    "article_md": {"r2_key": "k2", "put_url": "https://r2.example.com/k2", "expires_in": 900},
                    "article_docx": {"r2_key": "k3", "put_url": "https://r2.example.com/k3", "expires_in": 900},
                    "article_pdf": {"r2_key": "k4", "put_url": "https://r2.example.com/k4", "expires_in": 900},
                    "frames": [{"filename": "frame_0001.jpg", "r2_key": "f1", "put_url": "https://r2.example.com/f1", "expires_in": 900}],
                },
            }
            commit_resp = {
                "campaign_id": "camp-1",
                "source_pack_id": "spck-1",
                "asset_id": "asst-1",
                "asset_version_id": "asvr-1",
                "workflow_run_id": "wfr-1",
            }

            with patch.object(macp_adapter, "_post_json", side_effect=[(201, session_resp), (200, commit_resp)]):
                with patch.object(macp_adapter, "_put_file", return_value=200):
                    with patch("sys.stdout", new_callable=StringIO) as mock_out:
                        macp_adapter._register(folder, config, dry_run=False)

            output = mock_out.getvalue()
            self.assertIn("camp-1", output)
            self.assertIn("admin/ingestion/sessions/sess-abc", output)


# ---------------------------------------------------------------------------
# T13 — Commit 409 registered replay → treats as already registered
# ---------------------------------------------------------------------------

class TestCommitRegisteredReplay(unittest.TestCase):
    def test_409_registered_replay_treated_as_success(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp), frames=1)
            config = {
                "base_url": "https://macp.example.com",
                "brand_id": "brand-uuid",
                "created_by": "user-uuid",
                "editorial_brief_id": "brief-uuid",
                "token": "tok",
            }
            session_resp = {
                "session_id": "sess-001",
                "idempotent_hit": False,
                "uploads": {
                    "transcript_md": {"r2_key": "k1", "put_url": "https://r2.example.com/k1", "expires_in": 900},
                    "article_md": {"r2_key": "k2", "put_url": "https://r2.example.com/k2", "expires_in": 900},
                    "article_docx": {"r2_key": "k3", "put_url": "https://r2.example.com/k3", "expires_in": 900},
                    "article_pdf": {"r2_key": "k4", "put_url": "https://r2.example.com/k4", "expires_in": 900},
                    "frames": [{"filename": "frame_0001.jpg", "r2_key": "f1", "put_url": "https://r2.example.com/f1", "expires_in": 900}],
                },
            }
            replay_resp = {
                "error": "Session already registered",
                "campaign_id": "camp-existing",
                "asset_id": "asst-existing",
            }
            with patch.object(macp_adapter, "_post_json", side_effect=[(201, session_resp), (409, replay_resp)]):
                with patch.object(macp_adapter, "_put_file", return_value=200):
                    with patch("sys.stdout", new_callable=StringIO) as mock_out:
                        macp_adapter._register(folder, config, dry_run=False)
            self.assertIn("already registered", mock_out.getvalue().lower())


# ---------------------------------------------------------------------------
# T14 — Token and PUT URLs not logged to stdout/stderr
# ---------------------------------------------------------------------------

class TestSecretsNotLogged(unittest.TestCase):
    def test_token_not_in_stdout(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp))
            config = {
                "base_url": "https://macp.example.com",
                "brand_id": "brand-uuid",
                "created_by": "user-uuid",
                "editorial_brief_id": "brief-uuid",
                "token": "super-secret-tok-xyz",
            }
            with patch("sys.stdout", new_callable=StringIO) as mock_out:
                with patch("sys.stderr", new_callable=StringIO) as mock_err:
                    macp_adapter._register(folder, config, dry_run=True)
            self.assertNotIn("super-secret-tok-xyz", mock_out.getvalue())
            self.assertNotIn("super-secret-tok-xyz", mock_err.getvalue())

    def test_put_url_not_in_stdout(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp), frames=1)
            config = {
                "base_url": "https://macp.example.com",
                "brand_id": "brand-uuid",
                "created_by": "user-uuid",
                "editorial_brief_id": "brief-uuid",
                "token": "tok",
            }
            session_resp = {
                "session_id": "sess-001",
                "idempotent_hit": False,
                "uploads": {
                    "transcript_md": {"r2_key": "k1", "put_url": "https://presigned.r2.example.com/secret-k1", "expires_in": 900},
                    "article_md": {"r2_key": "k2", "put_url": "https://presigned.r2.example.com/k2", "expires_in": 900},
                    "article_docx": {"r2_key": "k3", "put_url": "https://presigned.r2.example.com/k3", "expires_in": 900},
                    "article_pdf": {"r2_key": "k4", "put_url": "https://presigned.r2.example.com/k4", "expires_in": 900},
                    "frames": [{"filename": "frame_0001.jpg", "r2_key": "f1", "put_url": "https://presigned.r2.example.com/f1", "expires_in": 900}],
                },
            }
            commit_resp = {"campaign_id": "camp-1"}
            with patch.object(macp_adapter, "_post_json", side_effect=[(201, session_resp), (200, commit_resp)]):
                with patch.object(macp_adapter, "_put_file", return_value=200):
                    with patch("sys.stdout", new_callable=StringIO) as mock_out:
                        with patch("sys.stderr", new_callable=StringIO) as mock_err:
                            macp_adapter._register(folder, config, dry_run=False)
            self.assertNotIn("presigned.r2.example.com", mock_out.getvalue())
            self.assertNotIn("presigned.r2.example.com", mock_err.getvalue())


# ---------------------------------------------------------------------------
# T15 — Content types correct
# ---------------------------------------------------------------------------

class TestContentTypes(unittest.TestCase):
    def test_md_content_type(self):
        self.assertEqual(macp_adapter.CONTENT_TYPES[".md"], "text/markdown")

    def test_docx_content_type(self):
        self.assertEqual(
            macp_adapter.CONTENT_TYPES[".docx"],
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    def test_pdf_content_type(self):
        self.assertEqual(macp_adapter.CONTENT_TYPES[".pdf"], "application/pdf")

    def test_jpg_content_type(self):
        self.assertEqual(macp_adapter.CONTENT_TYPES[".jpg"], "image/jpeg")


# ---------------------------------------------------------------------------
# T16 — Metadata extraction from video.info.json
# ---------------------------------------------------------------------------

class TestMetadataExtraction(unittest.TestCase):
    def test_youtube_platform_extracted(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp))
            meta = macp_adapter._read_metadata(folder)
            self.assertEqual(meta["source_platform"], "youtube")
            self.assertEqual(meta["external_source_id"], "abc123")
            self.assertEqual(meta["title"], "Test Video")
            self.assertEqual(meta["creator"], "Test Channel")

    def test_published_at_formatted(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp))
            meta = macp_adapter._read_metadata(folder)
            self.assertEqual(meta["published_at"], "2026-01-15")


# ---------------------------------------------------------------------------
# T17 — Local-file source unsupported → clear skip/stop message
# ---------------------------------------------------------------------------

class TestLocalFileUnsupported(unittest.TestCase):
    def test_local_file_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp))
            # Overwrite info.json with local-file style data (no webpage_url/extractor)
            info_json = folder / "download" / "video.info.json"
            info_json.write_text(json.dumps({"title": "local.mp4"}), encoding="utf-8")
            with self.assertRaises(SystemExit) as ctx:
                macp_adapter._read_metadata(folder)
            self.assertIn("Local-file", str(ctx.exception))


# ---------------------------------------------------------------------------
# T18 — Text files over MACP cap fail before session creation
# ---------------------------------------------------------------------------

class TestTextCapExceeded(unittest.TestCase):
    def test_oversized_transcript_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp))
            # Write a transcript over 1 MB
            (folder / "transcript.md").write_bytes(b"x" * (macp_adapter.MAX_TEXT_BYTES + 1))
            with self.assertRaises(SystemExit) as ctx:
                macp_adapter._validate_folder(folder)
            self.assertIn("1 MB", str(ctx.exception))


# ---------------------------------------------------------------------------
# T19 — Duplicate frame filename fails before session creation
# ---------------------------------------------------------------------------

class TestDuplicateFrameFilename(unittest.TestCase):
    def test_duplicate_frame_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp), frames=3)
            hires = folder / "hires"
            # Simulate duplicate by writing an extra file that globs to a dup
            # (can't literally have two identical names; test duplicate detection via
            # injecting a list with duplicates via monkeypatching glob)
            with patch("glob.glob", return_value=[
                str(hires / "frame_0001.jpg"),
                str(hires / "frame_0001.jpg"),  # duplicate
                str(hires / "frame_0002.jpg"),
            ]):
                with self.assertRaises(SystemExit) as ctx:
                    macp_adapter._validate_folder(folder)
            self.assertIn("duplicate", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# T20 — Unsupported source identity fails before session creation
# ---------------------------------------------------------------------------

class TestUnsupportedSourceIdentity(unittest.TestCase):
    def test_no_extractor_raises(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            folder = _make_ucid_folder(Path(tmp))
            info_json = folder / "download" / "video.info.json"
            info_json.write_text(
                json.dumps({"title": "Something", "id": "123"}),
                encoding="utf-8",
            )
            with self.assertRaises(SystemExit) as ctx:
                macp_adapter._read_metadata(folder)
            msg = str(ctx.exception).lower()
            self.assertTrue("local" in msg or "platform" in msg or "url" in msg)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
