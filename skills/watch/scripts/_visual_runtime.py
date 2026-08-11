#!/usr/bin/env python3
"""Trusted process and publication boundary for the visual harness."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import os
import platform
import signal
import stat
import subprocess  # nosec B404
import threading
import time

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import _visual_stream as visual_stream
from _visual_failures import ChildFailure, HarnessError

MAX_EXECUTABLE_BYTES = 512 * 1024 * 1024
MAX_CLI_PROBE_BYTES = 64 * 1024
MAX_CHILD_STDOUT_BYTES = 512 * 1024
MAX_CHILD_STDERR_BYTES = 128 * 1024
MAX_CHILD_INPUT_BYTES = 8 * 1024 * 1024
CONTEXT_LIMIT_MARKERS = (b"prompt is too long",)
ELF_MAGIC = b"\x7fELF"
SEAL_FLAGS = (
    fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
)
CONTAINMENT_HELPER_PATHS = (Path("/usr/bin/setpriv"), Path("/usr/bin/unshare"))
PR_SET_PDEATHSIG = 1
MIN_CLAUDE_VERSION = (2, 1, 220)
REQUIRED_CLAUDE_FLAGS = {
    "--safe-mode",
    "--tools",
    "--permission-mode",
    "--disable-slash-commands",
    "--input-format",
    "--output-format",
    "--json-schema",
    "--no-session-persistence",
    "--model",
    "--effort",
    "--max-budget-usd",
    "--system-prompt",
}
SAFE_ENVIRONMENT_KEYS = {
    "PATH",
    "HOME",
    "TMPDIR",
    "TEMP",
    "LANG",
    "LC_ALL",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
}


@dataclass(frozen=True)
class PinnedDirectory:
    path: Path
    descriptor: int
    identity: tuple[int, int]

    @property
    def descriptor_path(self) -> Path:
        return Path(f"/proc/self/fd/{self.descriptor}")

    def assert_path_identity(self) -> None:
        try:
            current = os.stat(self.path, follow_symlinks=False)
        except OSError as exc:
            raise HarnessError("work directory identity changed") from exc
        if (
            not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino) != self.identity
        ):
            raise HarnessError("work directory identity changed")


@dataclass(frozen=True)
class AuditedExecutable:
    path: Path
    descriptor: int
    identity: tuple[int, int, int, int, int, int, int]
    sha256: str
    capability_sha256: str
    snapshot_descriptor: int

    @property
    def command_path(self) -> str:
        return f"/proc/self/fd/{self.snapshot_descriptor}"

    def assert_identity(self) -> None:
        try:
            current = os.fstat(self.descriptor)
            snapshot = os.fstat(self.snapshot_descriptor)
        except OSError as exc:
            raise HarnessError("audited executable identity changed") from exc
        identity = _executable_identity(current)
        if (
            identity != self.identity
            or _hash_descriptor(self.descriptor, current.st_size) != self.sha256
            or _capability_sha256(self.descriptor) != self.capability_sha256
            or _hash_descriptor(self.snapshot_descriptor, snapshot.st_size)
            != self.sha256
            or fcntl.fcntl(self.snapshot_descriptor, fcntl.F_GET_SEALS) != SEAL_FLAGS
        ):
            raise HarnessError("audited executable identity changed")


@dataclass(frozen=True)
class ChildResult:
    structured_output: dict[str, Any]
    cost_usd: float


@dataclass(frozen=True)
class _Capture:
    stdout: bytes
    stderr: bytes
    returncode: int
    elapsed_seconds: float
    timed_out: bool
    overflow: bool
    input_failed: bool
    gate_failure: tuple[str, str | None] | None = None


def _hash_descriptor(descriptor: int, size: int) -> str:
    if size < 1 or size > MAX_EXECUTABLE_BYTES:
        raise HarnessError("Claude executable size is unsafe")
    digest = hashlib.sha256()
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, min(1024 * 1024, size - offset), offset)
        if not chunk:
            raise HarnessError("Claude executable changed during hashing")
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _executable_identity(
    info: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        stat.S_IMODE(info.st_mode),
        info.st_uid,
        info.st_gid,
    )


def _capability_sha256(descriptor: int) -> str:
    """Bind Linux file capabilities without exposing their value."""
    try:
        value = os.getxattr(f"/proc/self/fd/{descriptor}", "security.capability")
    except OSError as exc:
        if exc.errno in {
            errno.ENODATA,
            getattr(errno, "ENOATTR", errno.ENODATA),
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
        }:
            value = b""
        else:
            raise HarnessError(
                "Claude executable capability state is unavailable"
            ) from exc
    return hashlib.sha256(b"watch-file-capability-v1\0" + value).hexdigest()


def _is_native_elf(descriptor: int) -> bool:
    return os.pread(descriptor, len(ELF_MAGIC), 0) == ELF_MAGIC


def _sealed_snapshot(source: int, size: int) -> int:
    try:
        snapshot = os.memfd_create(
            "watch-audited-executable",
            os.MFD_ALLOW_SEALING,
        )
    except (AttributeError, OSError) as exc:
        raise HarnessError("sealed executable snapshots are unsupported") from exc
    try:
        offset = 0
        while offset < size:
            chunk = os.pread(source, min(1024 * 1024, size - offset), offset)
            if not chunk:
                raise HarnessError("executable changed during snapshot")
            view = memoryview(chunk)
            while view:
                count = os.write(snapshot, view)
                if count < 1:
                    raise HarnessError("executable snapshot failed")
                view = view[count:]
            offset += len(chunk)
        os.fchmod(snapshot, 0o500)
        fcntl.fcntl(snapshot, fcntl.F_ADD_SEALS, SEAL_FLAGS)
        if fcntl.fcntl(snapshot, fcntl.F_GET_SEALS) != SEAL_FLAGS:
            raise HarnessError("executable snapshot sealing failed")
        os.set_inheritable(snapshot, True)
        return snapshot
    except BaseException:
        os.close(snapshot)
        raise


@contextmanager
def pin_directory(path: Path) -> Iterator[PinnedDirectory]:
    if os.name != "posix" or not Path("/proc/self/fd").is_dir():
        raise HarnessError("trusted directory pinning is unsupported")
    try:
        if path.is_symlink():
            raise HarnessError("work directory must be a regular non-symlink directory")
        resolved = path.resolve(strict=True)
        descriptor = os.open(
            resolved,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise HarnessError("work directory could not be pinned") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise HarnessError("work directory must be a regular non-symlink directory")
        pinned = PinnedDirectory(resolved, descriptor, (info.st_dev, info.st_ino))
        pinned.assert_path_identity()
        yield pinned
    finally:
        os.close(descriptor)


@contextmanager
def audit_executable(path: Path) -> Iterator[AuditedExecutable]:
    if os.name != "posix" or not Path("/proc/self/fd").is_dir():
        raise HarnessError("trusted executable pinning is unsupported")
    try:
        if path.is_symlink():
            raise HarnessError("Claude executable must be a regular non-symlink file")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise HarnessError("Claude executable is unavailable") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not info.st_mode & 0o111:
            raise HarnessError("Claude executable must be a regular executable file")
        if not _is_native_elf(descriptor):
            raise HarnessError("Claude executable must be a native executable")
        identity = _executable_identity(info)
        executable_hash = _hash_descriptor(descriptor, info.st_size)
        snapshot = _sealed_snapshot(descriptor, info.st_size)
        try:
            audited = AuditedExecutable(
                path.resolve(strict=True),
                descriptor,
                identity,
                executable_hash,
                _capability_sha256(descriptor),
                snapshot,
            )
            audited.assert_identity()
            yield audited
        finally:
            os.close(snapshot)
    finally:
        os.close(descriptor)


ContainmentHelpers = tuple[AuditedExecutable, AuditedExecutable]


@contextmanager
def audit_containment_helpers() -> Iterator[ContainmentHelpers]:
    with ExitStack() as stack:
        setpriv = stack.enter_context(
            audit_executable(CONTAINMENT_HELPER_PATHS[0]),
        )
        unshare = stack.enter_context(
            audit_executable(CONTAINMENT_HELPER_PATHS[1]),
        )
        yield setpriv, unshare


def child_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in SAFE_ENVIRONMENT_KEYS
        or key.startswith("ANTHROPIC_")
        or key.startswith("CLAUDE_CODE_USE_")
    }
    return {**environment, "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}


def _runtime_executable_record(executable: AuditedExecutable) -> dict[str, Any]:
    executable.assert_identity()
    return {
        "sha256": executable.sha256,
        "privilege": {
            "mode": executable.identity[4],
            "uid": executable.identity[5],
            "gid": executable.identity[6],
            "capability_sha256": executable.capability_sha256,
        },
    }


def runtime_digest(
    executable: AuditedExecutable,
    helpers: ContainmentHelpers,
) -> str:
    routing: dict[str, Any] = {}
    for key, value in sorted(child_environment().items()):
        if any(marker in key for marker in ("TOKEN", "KEY", "SECRET", "PASSWORD")):
            routing[key] = {"present": bool(value)}
        else:
            routing[key] = value
    payload = json.dumps(
        {
            "executable": _runtime_executable_record(executable),
            "containment_helpers": [
                _runtime_executable_record(helper) for helper in helpers
            ],
            "environment": routing,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(b"watch-runtime-v3\0" + payload).hexdigest()


def inspect_runtime(
    executable_path: Path,
    *,
    publication_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    with (
        audit_executable(executable_path) as executable,
        audit_containment_helpers() as helpers,
    ):
        check_claude_cli(executable, helpers)
        if publication_check is not None:
            publication_check()
        return {
            "schema_version": 1,
            "executable_sha256": executable.sha256,
            "runtime_digest": runtime_digest(executable, helpers),
        }


def _arm_parent_death_signal(parent_pid: int) -> None:
    """Close the fork-to-setpriv parent-death race in the child."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl failed")
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def _kill_group(process: subprocess.Popen[bytes], sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
    except (OSError, ProcessLookupError):
        pass


def _stop_and_reap(process: subprocess.Popen[bytes]) -> bool:
    _kill_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=0.25)
    except subprocess.TimeoutExpired:
        _kill_group(process, signal.SIGKILL)
        try:
            process.wait(timeout=0.75)
        except subprocess.TimeoutExpired:
            return False
    return True


def _pid_namespace_prefix(helpers: ContainmentHelpers) -> list[str]:
    if os.name != "posix" or platform.system() != "Linux":
        raise HarnessError("trusted process-tree containment is unsupported")
    setpriv, unshare = helpers
    return [
        setpriv.command_path,
        "--pdeathsig",
        "KILL",
        "--no-new-privs",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        unshare.command_path,
        "--user",
        "--map-current-user",
        "--pid",
        "--fork",
        "--kill-child=KILL",
        "--mount-proc",
        "--",
    ]


def _execute(
    executable: AuditedExecutable,
    helpers: ContainmentHelpers,
    arguments: list[str],
    *,
    raw_input: bytes | None,
    timeout_seconds: float,
    stdout_limit: int,
    stderr_limit: int,
    model: str | None = None,
    effort: str | None = None,
) -> _Capture:
    executable.assert_identity()
    for helper in helpers:
        helper.assert_identity()
    command = [
        *_pid_namespace_prefix(helpers),
        executable.command_path,
        *arguments,
    ]
    snapshots = (
        executable.snapshot_descriptor,
        *(helper.snapshot_descriptor for helper in helpers),
    )
    started = time.monotonic()
    parent_pid = os.getpid()

    def arm_parent_death_signal() -> None:
        _arm_parent_death_signal(parent_pid)

    try:
        process = subprocess.Popen(  # nosec B603
            command,
            stdin=subprocess.PIPE if raw_input is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=child_environment(),
            start_new_session=True,
            pass_fds=snapshots,
            preexec_fn=arm_parent_death_signal,  # nosec B604
        )
    except OSError as exc:
        raise HarnessError("Claude child could not start") from exc
    if (
        process.stdout is None
        or process.stderr is None
        or (raw_input is not None and process.stdin is None)
    ):
        _stop_and_reap(process)
        raise ChildFailure(
            "cleanup", elapsed_seconds=round(time.monotonic() - started, 3)
        )

    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    overflow = threading.Event()
    input_failed = threading.Event()
    gate = (
        visual_stream.EffortGate(model, effort)
        if raw_input and model and effort
        else None
    )

    def drain(name: str, pipe: Any, limit: int) -> None:
        try:
            read_chunk = getattr(pipe, "read1", pipe.read)
            while True:
                chunk = read_chunk(65536)
                if not chunk:
                    if name == "stdout" and gate:
                        gate.finish(incomplete=overflow.is_set())
                    return
                if len(buffers[name]) + len(chunk) > limit:
                    overflow.set()
                    if gate:
                        gate.cancel("output_limit")
                    _kill_group(process, signal.SIGKILL)
                    return
                buffers[name].extend(chunk)
                if name == "stdout" and gate:
                    gate.feed(chunk)
                    if gate.failure:
                        if gate.failure[0] == "output_limit":
                            overflow.set()
                        _kill_group(process, signal.SIGKILL)
                        return
        except (OSError, ValueError):
            if gate:
                gate.cancel()

    readers = [
        threading.Thread(
            target=drain, args=("stdout", process.stdout, stdout_limit), daemon=True
        ),
        threading.Thread(
            target=drain, args=("stderr", process.stderr, stderr_limit), daemon=True
        ),
    ]
    for reader in readers:
        reader.start()

    writer: threading.Thread | None = None
    if raw_input is not None:

        def write_input() -> None:
            try:
                input_stream = process.stdin
                if input_stream is None:
                    input_failed.set()
                    return
                if gate:
                    input_stream.write(gate.request)
                    input_stream.flush()
                    gate.wait(timeout_seconds)
                    if not gate.dispatch(input_stream.fileno(), raw_input):
                        input_stream.close()
                        return
                else:
                    input_stream.write(raw_input)
                input_stream.close()
            except (BrokenPipeError, OSError, ValueError):
                input_failed.set()

        writer = threading.Thread(target=write_input, daemon=True)
        writer.start()

    timed_out = False
    cleanup_ok = True
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        if gate:
            gate.cancel()
        cleanup_ok = _stop_and_reap(process)
    except BaseException as interruption:
        if gate:
            gate.cancel()
        cleanup_ok = _stop_and_reap(process)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        interrupted_threads = [*readers, *([writer] if writer is not None else [])]
        for thread in interrupted_threads:
            thread.join(timeout=0.25)
        if not cleanup_ok or any(thread.is_alive() for thread in interrupted_threads):
            raise ChildFailure(
                "cleanup",
                elapsed_seconds=round(time.monotonic() - started, 3),
            ) from interruption
        raise

    threads = [*readers, *([writer] if writer is not None else [])]
    for thread in threads:
        thread.join(timeout=0.75)
    if any(thread.is_alive() for thread in threads):
        cleanup_ok = _stop_and_reap(process) and cleanup_ok
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        for thread in threads:
            thread.join(timeout=0.25)
    if any(thread.is_alive() for thread in threads) or not cleanup_ok:
        raise ChildFailure(
            "cleanup", elapsed_seconds=round(time.monotonic() - started, 3)
        )
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
    return _Capture(
        stdout=bytes(buffers["stdout"]),
        stderr=bytes(buffers["stderr"]),
        returncode=process.returncode
        if process.returncode is not None
        else -signal.SIGKILL,
        elapsed_seconds=round(time.monotonic() - started, 3),
        timed_out=timed_out,
        overflow=overflow.is_set(),
        input_failed=input_failed.is_set(),
        gate_failure=(None if overflow.is_set() else gate.failure if gate else None),
    )


def _probe(
    executable: AuditedExecutable,
    helpers: ContainmentHelpers,
    argument: str,
    label: str,
) -> tuple[int, str]:
    capture = _execute(
        executable,
        helpers,
        [argument],
        raw_input=None,
        timeout_seconds=10.0,
        stdout_limit=MAX_CLI_PROBE_BYTES,
        stderr_limit=MAX_CLI_PROBE_BYTES,
    )
    if capture.timed_out:
        raise HarnessError(f"{label} check timed out")
    if capture.overflow:
        raise HarnessError(f"{label} output limit exceeded")
    return capture.returncode, capture.stdout.decode("utf-8", "replace")


def check_claude_cli(
    executable: AuditedExecutable,
    helpers: ContainmentHelpers,
) -> None:
    version_code, version_stdout = _probe(
        executable,
        helpers,
        "--version",
        "Claude CLI version",
    )
    import re

    match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_stdout)
    if (
        version_code
        or not match
        or tuple(map(int, match.groups())) < MIN_CLAUDE_VERSION
    ):
        raise HarnessError("Claude CLI 2.1.220 or newer is required")
    help_code, help_stdout = _probe(
        executable,
        helpers,
        "--help",
        "Claude CLI capability",
    )
    missing = sorted(flag for flag in REQUIRED_CLAUDE_FLAGS if flag not in help_stdout)
    if help_code or missing:
        raise HarnessError("Claude CLI required flags are unavailable")


def _is_context_limit(stderr: bytes) -> bool:
    folded = stderr.lower()
    return any(marker in folded for marker in CONTEXT_LIMIT_MARKERS)


def run_child(
    executable: AuditedExecutable,
    helpers: ContainmentHelpers,
    *,
    model: str,
    effort: str,
    budget: float,
    schema: dict[str, Any],
    system_prompt: str,
    message: dict[str, Any],
    timeout_seconds: float,
    phase: str,
    child_ordinal: int,
    deadline_limited: bool = False,
) -> ChildResult:
    arguments = [
        "--safe-mode",
        "-p",
        "--tools",
        "",
        "--permission-mode",
        "dontAsk",
        "--disable-slash-commands",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--json-schema",
        json.dumps(schema, separators=(",", ":")),
        "--no-session-persistence",
        "--model",
        model,
        "--effort",
        effort,
        "--max-budget-usd",
        f"{budget:.6f}",
        "--system-prompt",
        system_prompt,
    ]
    raw_input = (
        json.dumps(message, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    base = {
        "phase": phase,
        "child_ordinal": child_ordinal,
        "model": model,
        "effort": effort,
    }
    if len(raw_input) > MAX_CHILD_INPUT_BYTES:
        raise ChildFailure("context_limit", **base)
    capture = _execute(
        executable,
        helpers,
        arguments,
        raw_input=raw_input,
        timeout_seconds=timeout_seconds,
        stdout_limit=MAX_CHILD_STDOUT_BYTES,
        stderr_limit=MAX_CHILD_STDERR_BYTES,
        model=model,
        effort=effort,
    )
    base = {**base, "elapsed_seconds": capture.elapsed_seconds}
    if capture.overflow:
        raise ChildFailure("output_limit", **base)
    if capture.timed_out:
        raise ChildFailure("review_deadline" if deadline_limited else "timeout", **base)
    if capture.returncode and _is_context_limit(capture.stderr):
        raise ChildFailure("context_limit", **base)
    if capture.gate_failure:
        category, dimension = capture.gate_failure
        raise ChildFailure(category, **base, identity_dimension=dimension)
    if capture.returncode < 0:
        raise ChildFailure("signal", **base, signal=-capture.returncode)
    if not capture.stdout.strip() and capture.returncode:
        raise ChildFailure("process_exit", **base, exit_code=capture.returncode)
    structured, cost = visual_stream.validate_child_events(
        capture.stdout,
        context=base,
        model=model,
        budget=budget,
        returncode=capture.returncode,
        input_failed=capture.input_failed,
    )
    return ChildResult(structured, cost)
