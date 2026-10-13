"""Native libraries from a script (feature 109): `native("name")` and what stands behind it.

A library the author built runs in a child host process (`native_host.py`), so a crash, an
abort or a hang in it reaches the script as a `NativeError` while the app keeps running.
Buffers are files mapped into both processes; the HANDLE owns them, so they outlive a host
crash and a reload, and an array keeps its mapping alive for as long as anything holds it.

Every handle and buffer belongs to an OWNER: the compiled script's module code, or one script
instance. `PythonBehavior` sets the owner around each entry into user code, and releasing an
owner stops its hosts and deletes its files. That is what makes a reload, a reset, a dry run
or an export leave nothing behind.
"""

import collections
import contextlib
import contextvars
import ctypes
import math
import mmap
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from loguru import logger

from shaderbox.scripting import native_host

NativeErrorKind = Literal["failed", "crashed", "timeout", "missing", "load"]

DEFAULT_TIMEOUT = 2.0
# How long a changed library file must hold still before a reload. A linker grows the file
# over tens of milliseconds, and loading it mid-write crashes the loading host.
SETTLE_SECONDS = 0.5
_STDERR_TAIL = 12
_STDERR_CHUNK = 65536
_FILE_PREFIX = "sbnative-"
_STALE_RE = re.compile(rf"^{_FILE_PREFIX}(\d+)-")
_HOST_PATH = Path(native_host.__file__)

_LIBRARY_PATTERNS: dict[str, tuple[str, ...]] = {
    "linux": ("{}.so", "lib{}.so"),
    "win32": ("{}.dll",),
    "darwin": ("lib{}.dylib", "{}.dylib"),
}


class NativeError(Exception):
    """A native library call that did not succeed. `kind` says how:

    `failed` -- the export returned nonzero (the message is its `sb_fail` text);
    `crashed` -- the library's process died; `timeout` -- it outlived the deadline and was
    stopped; `missing` -- no such export; `load` -- the library could not be found, loaded or
    verified, or the handle was released.
    """

    def __init__(self, kind: NativeErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind: NativeErrorKind = kind


@dataclass
class _Buffer:
    path: Path
    size: int
    dtype: str
    shape: tuple[int, ...]
    array: np.ndarray
    library: "NativeLibrary"


class NativeOwner:
    """What a handle or a buffer is released with: a script's module code or one instance.

    `timeout_cap` bounds every deadline under this owner -- the completion probe runs on the
    UI thread, where a library's full default deadline would freeze the editor.
    """

    def __init__(
        self, scripts_dir: Path | None, timeout_cap: float | None = None
    ) -> None:
        self.scripts_dir = scripts_dir
        self.timeout_cap = timeout_cap
        self.libraries: dict[tuple[str, float, bool], NativeLibrary] = {}
        # A load that failed, with the library file's stamp at the time: `native()` raises it
        # again without starting a host until the file changes, so a broken library named in
        # `update` costs a stat per frame rather than a host start.
        self.failures: dict[
            tuple[str, float, bool], tuple[tuple[int, int] | None, NativeError]
        ] = {}
        self.buffers: list[_Buffer] = []
        self.released = False

    def release(self) -> None:
        self.released = True
        for buffer in self.buffers:
            buffer.library.forget_buffer(buffer)
        self.buffers.clear()
        for library in self.libraries.values():
            library.release()
        self.libraries.clear()

    def revive(self) -> None:
        for library in self.libraries.values():
            library.revive()


_OWNER: contextvars.ContextVar[NativeOwner | None] = contextvars.ContextVar(
    "native_owner", default=None
)


@contextlib.contextmanager
def owning(owner: NativeOwner) -> Iterator[None]:
    token = _OWNER.set(owner)
    try:
        yield
    finally:
        _OWNER.reset(token)


def native(
    name: str, *, timeout: float = DEFAULT_TIMEOUT, debug: bool = False
) -> "NativeLibrary":
    """Load the library `name` from the document's `scripts/` directory and return its handle.

    `name` is a bare name (`"fluid"` finds `fluid.so`/`libfluid.so`, `fluid.dll`,
    `libfluid.dylib`) or a path, relative to `scripts/` or absolute. The same name in the same
    script phase returns the same handle, so calling it inside `update` costs nothing after the
    first frame. `timeout` is the per-call deadline in seconds; `debug=True` removes it and lets
    a debugger attach to `handle.pid`.
    """
    owner = _OWNER.get()
    if owner is None:
        raise NativeError(
            "load",
            "native() runs in a script's module code, __init__ or update -- create the handle "
            "there and pass it to a thread",
        )
    if owner.released:
        raise NativeError("load", "native() on a script that has been released")
    key = (name, timeout, debug)
    existing = owner.libraries.get(key)
    if existing is not None:
        return existing
    path = _resolve(name, owner.scripts_dir)
    stamp = _file_stamp(path)
    failure = owner.failures.get(key)
    if failure is not None and failure[0] == stamp:
        raise NativeError(failure[1].kind, str(failure[1]))
    effective: float | None = None if debug else timeout
    if owner.timeout_cap is not None:
        effective = (
            owner.timeout_cap
            if effective is None
            else min(effective, owner.timeout_cap)
        )
    try:
        library = NativeLibrary(path, effective, debug)
    except NativeError as e:
        owner.failures[key] = (stamp, e)
        raise
    owner.failures.pop(key, None)
    owner.libraries[key] = library
    return library


def _file_stamp(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _resolve(name: str, scripts_dir: Path | None) -> Path:
    as_path = Path(name)
    if as_path.suffix or len(as_path.parts) > 1:
        if as_path.is_absolute():
            candidates = [as_path]
        elif scripts_dir is not None:
            candidates = [scripts_dir / as_path]
        else:
            candidates = []
    else:
        patterns = _LIBRARY_PATTERNS.get(sys.platform, _LIBRARY_PATTERNS["linux"])
        candidates = (
            [scripts_dir / pattern.format(name) for pattern in patterns]
            if scripts_dir is not None
            else []
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if not candidates:
        raise NativeError(
            "load", f"cannot find `{name}`: this script has no document directory"
        )
    tried = ", ".join(str(c) for c in candidates)
    raise NativeError("load", f"cannot find `{name}` -- tried {tried}")


def _scratch_dir() -> Path:
    shm = Path("/dev/shm")
    if sys.platform.startswith("linux") and shm.is_dir() and os.access(shm, os.W_OK):
        return shm
    return Path(tempfile.gettempdir())


def _unlink(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


@dataclass
class _Host:
    process: subprocess.Popen[bytes]
    replies: "queue.Queue[object]"
    stderr_tail: collections.deque[str]
    stderr_reader: threading.Thread


_EOF = object()
_CORRUPT = object()


@dataclass
class _Resources:
    """Everything a handle must give back, held apart from the handle so its finalizer can."""

    host: _Host | None = None
    copies: set[Path] = field(default_factory=set)
    files: set[Path] = field(default_factory=set)

    def stop_host(self) -> None:
        host = self.host
        self.host = None
        if host is not None:
            _stop(host)
        for copy in list(self.copies):
            _unlink(copy)
            self.copies.discard(copy)

    def close(self) -> None:
        self.stop_host()
        for path in list(self.files):
            _unlink(path)
        self.files.clear()


def _stop(host: _Host) -> None:
    # Only stdin is closed here. The reader threads own stdout and stderr, and closing a
    # stream a reader is blocked on waits for the pipe's EOF -- which never comes while a
    # process the library started still holds it.
    if host.process.poll() is None:
        host.process.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        host.process.wait(timeout=2.0)
    if host.process.stdin is not None:
        with contextlib.suppress(OSError):
            host.process.stdin.close()


def _read_replies(stream: Any, replies: "queue.Queue[object]") -> None:
    try:
        while True:
            frame = native_host.read_frame(stream)
            if frame is None:
                replies.put(_EOF)
                return
            replies.put(frame)
    except ValueError:
        replies.put(_CORRUPT)
    except OSError:
        replies.put(_EOF)


def _read_stderr(stream: Any, label: str, tail: collections.deque[str]) -> None:
    # One log record per chunk read, not per line: the app's log sinks are slow enough per
    # record that a chatty library would otherwise be throttled by its own stderr pipe.
    pending = b""
    with contextlib.suppress(OSError, ValueError):
        while True:
            chunk = stream.read1(_STDERR_CHUNK)
            if not chunk:
                break
            *lines, pending = (pending + chunk).split(b"\n")
            text = [line.decode("utf-8", "replace").rstrip() for line in lines]
            tail.extend(text)
            if text:
                logger.info(f"[native {label}] " + "\n".join(text))
    if pending:
        tail.append(pending.decode("utf-8", "replace").rstrip())


def _exit_description(code: int | None) -> str:
    if code is None:
        return "did not exit"
    if code < 0:
        try:
            return f"killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"killed by signal {-code}"
    return f"exited with code {code}"


class NativeLibrary:
    """A loaded native library. `call(export, **args)` runs one export; `buffer(shape, dtype)`
    allocates memory the library can read and write."""

    def __init__(self, path: Path, timeout: float | None, debug: bool) -> None:
        self.path = path
        self.timeout = timeout
        self.debug = debug
        self._lock = threading.Lock()
        self._buffers: dict[int, _Buffer] = {}
        self._dead: NativeError | None = None
        self._released = False
        self._loaded_stamp: tuple[int, int] | None = None
        self._pending_stamp: tuple[int, int] | None = None
        self._pending_since = 0.0
        self._resources = _Resources()
        self._finalizer = weakref.finalize(self, self._resources.close)
        with self._lock:
            self._start()
            if self._dead is not None:
                error = self._dead
                self.release()
                raise error

    @property
    def pid(self) -> int | None:
        host = self._resources.host
        return host.process.pid if host is not None else None

    # ---- lifecycle ---------------------------------------------------------------------

    def _stamp(self) -> tuple[int, int] | None:
        return _file_stamp(self.path)

    def _start(self) -> None:
        self._loaded_stamp = self._stamp()
        copy = _scratch_dir() / (
            f"{_FILE_PREFIX}{os.getpid()}-{uuid.uuid4().hex}{self.path.suffix}"
        )
        self._resources.copies.add(copy)
        try:
            shutil.copyfile(self.path, copy)
        except OSError as e:
            self._die(NativeError("load", f"cannot copy {self.path}: {e}"))
            return
        executable = getattr(sys, "_base_executable", sys.executable) or sys.executable
        try:
            process = subprocess.Popen(
                [executable, "-I", "-S", str(_HOST_PATH)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as e:
            self._die(
                NativeError("load", f"cannot start the library's host process: {e}")
            )
            return
        replies: queue.Queue[object] = queue.Queue()
        tail: collections.deque[str] = collections.deque(maxlen=_STDERR_TAIL)
        threading.Thread(
            target=_read_replies, args=(process.stdout, replies), daemon=True
        ).start()
        stderr_reader = threading.Thread(
            target=_read_stderr,
            args=(process.stderr, self.path.name, tail),
            daemon=True,
        )
        stderr_reader.start()
        self._resources.host = _Host(process, replies, tail, stderr_reader)
        reply = self._request(
            {"op": "load", "path": str(copy), "debug": self.debug}, "loading"
        )
        if reply is None:
            return
        if not reply.get("ok"):
            self._die(NativeError(reply.get("kind", "load"), str(reply.get("message"))))

    def _die(self, error: NativeError) -> None:
        self._dead = error
        self._resources.stop_host()

    def _check_reload(self) -> None:
        stamp = self._stamp()
        if stamp is None or stamp == self._loaded_stamp:
            # A missing file is a linker between unlink and create: keep what is running.
            self._pending_stamp = None
            return
        now = time.monotonic()
        if stamp != self._pending_stamp:
            self._pending_stamp = stamp
            self._pending_since = now
            return
        if now - self._pending_since < SETTLE_SECONDS:
            return
        self._pending_stamp = None
        self._resources.stop_host()
        self._dead = None
        self._start()

    def revive(self) -> None:
        with self._lock:
            if self._dead is not None and not self._released:
                self._dead = None
                self._start()

    def release(self) -> None:
        self._released = True
        self._resources.stop_host()
        for buffer in list(self._buffers.values()):
            self.forget_buffer(buffer)
        self._finalizer()

    def forget_buffer(self, buffer: _Buffer) -> None:
        if self._buffers.get(id(buffer.array)) is buffer:
            del self._buffers[id(buffer.array)]
        self._resources.files.discard(buffer.path)
        _unlink(buffer.path)
        # The host maps a buffer on first use and keeps it; unmapping is what returns the
        # pages, since an unlinked file's memory lives as long as any mapping of it. A host
        # busy in another thread's call keeps the mapping until it stops instead.
        if self._resources.host is not None and self._lock.acquire(timeout=0.1):
            try:
                self._request({"op": "unmap", "path": str(buffer.path)}, "unmapping")
            finally:
                self._lock.release()

    # ---- the protocol ------------------------------------------------------------------

    def _request(self, message: dict[str, Any], what: str) -> dict[str, Any] | None:
        """The host's reply, or None after recording why there is none in `_dead`."""
        host = self._resources.host
        if host is None:
            self._dead = self._dead or NativeError(
                "crashed", "the library is not running"
            )
            return None
        stdin = host.process.stdin
        assert stdin is not None
        # A host that is gone already has its EOF on the way; a stream closed under us by a
        # release from another thread raises ValueError.
        with contextlib.suppress(OSError, ValueError):
            native_host.write_frame(stdin, message)
        try:
            reply = host.replies.get(timeout=self.timeout)
        except queue.Empty:
            self._die(
                NativeError(
                    "timeout",
                    f"{self.path.name}: {what} took longer than {self.timeout} s -- stopped",
                )
            )
            return None
        if reply is _CORRUPT:
            self._die(
                NativeError(
                    "crashed",
                    f"{self.path.name}: protocol corrupted while {what} -- stopped",
                )
            )
            return None
        if reply is _EOF:
            try:
                code = host.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                code = None
            host.stderr_reader.join(timeout=0.5)
            message_text = (
                f"{self.path.name} crashed while {what} ({_exit_description(code)})"
            )
            if host.stderr_tail:
                message_text += "; last output: " + " | ".join(host.stderr_tail)
            self._die(NativeError("crashed", message_text))
            return None
        assert isinstance(reply, dict)
        return reply

    # ---- the script's surface ----------------------------------------------------------

    def buffer(self, shape: int | tuple[int, ...], dtype: str = "f4") -> np.ndarray:
        """A zero-filled C-contiguous array the library sees as an `SB_BUFFER` argument.

        Pass the returned object itself to `call` -- a slice or a copy is refused. `dtype` is
        one of f4 f8 i4 u4 i8 u1; `shape` has one to four extents, each at least 1.
        """
        dims = (shape,) if isinstance(shape, int) else tuple(shape)
        if not 1 <= len(dims) <= native_host.MAX_DIMS:
            raise NativeError(
                "load", f"a buffer has 1 to 4 dimensions, not {len(dims)}"
            )
        if any(not isinstance(d, int) or d < 1 for d in dims):
            raise NativeError(
                "load", f"every buffer extent must be an int >= 1, got {dims}"
            )
        normalized = np.dtype(dtype).str.lstrip("<>|=")
        if normalized not in native_host.DTYPE_CODES:
            raise NativeError(
                "load",
                f"buffer dtype {dtype!r} -- use one of {', '.join(native_host.DTYPE_CODES)}",
            )
        if self._released:
            raise NativeError(
                "load", f"{self.path.name}: this handle has been released"
            )
        size = math.prod(dims) * np.dtype(normalized).itemsize
        path = _scratch_dir() / f"{_FILE_PREFIX}{os.getpid()}-{uuid.uuid4().hex}.buf"
        fd = os.open(
            path, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        )
        try:
            try:
                if hasattr(os, "posix_fallocate"):
                    # Reserve the pages now: a sparse file on a full tmpfs maps fine and then
                    # kills this process with SIGBUS on the first write to an unbacked page.
                    os.posix_fallocate(fd, 0, size)
                else:
                    os.ftruncate(fd, size)
            except OSError as e:
                os.close(fd)
                fd = -1
                _unlink(path)
                raise NativeError(
                    "load", f"cannot allocate a {size}-byte buffer: {e}"
                ) from e
            mapping = mmap.mmap(fd, size)
        finally:
            if fd >= 0:
                os.close(fd)
        array = np.frombuffer(mapping, dtype=normalized).reshape(dims)
        owner = _OWNER.get()
        record = _Buffer(path, size, normalized, dims, array, self)
        self._buffers[id(array)] = record
        self._resources.files.add(path)
        if owner is not None and not owner.released:
            owner.buffers.append(record)
        return array

    def call(self, export: str, /, **args: object) -> dict[str, Any]:
        """Run `export` with named arguments; return the results it set with `sb_ret_*`.

        Numbers, integers, strings and this handle's buffers cross; `None` leaves an argument
        out. Raises `NativeError` on any failure.
        """
        packed = [self._encode(name, value) for name, value in args.items()]
        with self._lock:
            if self._released:
                raise NativeError(
                    "load", f"{self.path.name}: this handle has been released"
                )
            self._check_reload()
            if self._dead is not None:
                raise NativeError(self._dead.kind, str(self._dead))
            reply = self._request(
                {
                    "op": "call",
                    "export": export,
                    "args": [a for a in packed if a is not None],
                },
                f"running `{export}`",
            )
            if reply is None:
                if self._released:
                    raise NativeError(
                        "load", f"{self.path.name}: this handle has been released"
                    )
                assert self._dead is not None
                raise NativeError(self._dead.kind, str(self._dead))
        if not reply.get("ok"):
            raise NativeError(
                reply.get("kind", "failed"), f"{self.path.name}: {reply.get('message')}"
            )
        rc = int(reply["rc"])
        if rc != 0:
            text = reply.get("error") or f"`{export}` returned {rc} without sb_fail"
            raise NativeError("failed", f"{self.path.name}.{export}: {text}")
        results = reply.get("results")
        return dict(results) if isinstance(results, dict) else {}

    def info(self) -> dict[str, Any]:
        """The host's pid and imported modules -- what a test asserts the host is."""
        with self._lock:
            reply = self._request({"op": "info"}, "reporting")
            if reply is None:
                assert self._dead is not None
                raise NativeError(self._dead.kind, str(self._dead))
            return reply

    def _encode(self, name: str, value: object) -> list[Any] | None:
        if value is None:
            return None
        if isinstance(value, bool | np.bool_ | int | np.integer):
            number = int(value)
            if not -(2**63) <= number < 2**63:
                raise NativeError(
                    "failed", f"argument `{name}` = {number} does not fit int64"
                )
            return [name, native_host.SB_INTEGER, number]
        if isinstance(value, float | np.floating):
            return [name, native_host.SB_NUMBER, float(value)]
        if isinstance(value, str):
            if "\0" in value:
                raise NativeError(
                    "failed", f"argument `{name}` contains a NUL character"
                )
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as e:
                raise NativeError(
                    "failed", f"argument `{name}` is not valid UTF-8: {e}"
                ) from e
            return [name, native_host.SB_STRING, value]
        if isinstance(value, np.ndarray):
            record = self._buffers.get(id(value))
            if record is None or record.array is not value:
                raise NativeError(
                    "failed",
                    f"argument `{name}` is not a buffer of this library -- pass the array "
                    "`lib.buffer(...)` returned, not a slice, a copy or another library's",
                )
            return [
                name,
                native_host.SB_BUFFER,
                {
                    "path": str(record.path),
                    "size": record.size,
                    "dtype": native_host.DTYPE_CODES[record.dtype],
                    "shape": list(record.shape),
                },
            ]
        raise NativeError(
            "failed",
            f"argument `{name}` is a {type(value).__name__} -- pass a number, an int, a str "
            "or a buffer from lib.buffer",
        )


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        kernel32 = ctypes.WinDLL("kernel32")
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def sweep_stale_files() -> None:
    """Delete buffer files and library copies left by a ShaderBox process that is gone."""
    for directory in {_scratch_dir(), Path(tempfile.gettempdir())}:
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            match = _STALE_RE.match(entry.name)
            if match is None:
                continue
            pid = int(match.group(1))
            if pid != os.getpid() and not _pid_alive(pid):
                _unlink(entry)
