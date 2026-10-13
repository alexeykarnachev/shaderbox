"""Native libraries from a script (feature 109): the gates the spec's Verification section names.

The fixture library is C, compiled at test time with the system `cc` against the shipped
header. With no compiler every test here is SKIPPED, which is not a pass. Each test's
docstring names the break it must catch.
"""

import contextlib
import itertools
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import threading
import time
import types
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

from shaderbox.constants import RESOURCES_DIR
from shaderbox.project_session import ProjectSession
from shaderbox.scripting import ScriptContext, ScriptEngine, native_library
from shaderbox.scripting.behavior import PythonBehavior
from shaderbox.scripting.native_library import (
    SETTLE_SECONDS,
    NativeError,
    NativeLibrary,
    NativeOwner,
    native,
    owning,
)

_NATIVE_DIR = RESOURCES_DIR / "native"
_CC = shutil.which("cc")
_ODIN = shutil.which("odin")

pytestmark = pytest.mark.skipif(
    _CC is None,
    reason="no C compiler: the native-library gates are SKIPPED, not passed",
)

_FIXTURE = r"""
#include "shaderbox_native.h"
#include <stdlib.h>
#ifndef K
#define K 111
#endif
SB_LIBRARY()

#ifdef LOAD_FAIL
SB_EXPORT int32_t sb_load(sb_call *c) { return sb_fail(c, "no license file"); }
#endif
#ifdef LOAD_HANG
SB_EXPORT int32_t sb_load(sb_call *c) { (void)c; for (;;) {} }
#endif
#ifdef LOCKFILE
#include <fcntl.h>
#include <sys/file.h>
SB_EXPORT int32_t sb_load(sb_call *c) {
    int fd = open(LOCKFILE, O_CREAT | O_RDWR, 0600);
    if (fd < 0 || flock(fd, LOCK_EX | LOCK_NB) != 0) return sb_fail(c, "device busy");
    return 0;
}
#endif

SB_EXPORT int32_t chatty(sb_call *c) { (void)c; printf("hello from printf\n"); return 0; }

SB_EXPORT int32_t printcrash(sb_call *c) {
    (void)c;
    printf("last words\n");
    volatile int *p = 0;
    *p = 1;
    return 0;
}

SB_EXPORT int32_t noeol(sb_call *c) { (void)c; fputs("final-no-newline", stderr); return 0; }

SB_EXPORT int32_t flood(sb_call *c) {
    int64_t n = sb_int(c, "bytes", 0);
    char block[4096];
    memset(block, 'x', sizeof block);
    for (int64_t done = 0; done < n; done += (int64_t)sizeof block)
        fwrite(block, 1, sizeof block, stderr);
    return 0;
}

SB_EXPORT int32_t spawn_bg(sb_call *c) { (void)c; return system("sleep 3 &") == 0 ? 0 : 1; }

SB_EXPORT int32_t fill(sb_call *c) {
    sb_arg *b = sb_buf(c, "buf");
    if (!b || b->dtype != SB_F32) return sb_fail(c, "buf must be an f4 buffer");
    float *p = (float *)b->data;
    for (int64_t i = 0; i < b->shape[0]; i++) p[i] = (float)sb_num(c, "value", 1.0);
    sb_ret_int(c, "k", K);
    sb_ret_str(c, "who", "fixture");
    sb_ret_num(c, "half", 0.5);
    return 0;
}

SB_EXPORT int32_t first(sb_call *c) {
    sb_arg *b = sb_buf(c, "buf");
    if (!b) return sb_fail(c, "no buf");
    ((float *)b->data)[0] = (float)K;
    return 0;
}

SB_EXPORT int32_t constant(sb_call *c) { sb_ret_int(c, "k", K); return 0; }

SB_EXPORT int32_t boom(sb_call *c) {
    (void)c;
    fprintf(stderr, "about to crash\n");
    fflush(stderr);
    volatile int *p = 0;
    *p = 1;
    return 0;
}

SB_EXPORT int32_t hang(sb_call *c) { (void)c; for (;;) {} return 0; }
SB_EXPORT int32_t silent(sb_call *c) { (void)c; return 3; }

SB_EXPORT int32_t shout(sb_call *c) {
    static const char frame[] = "\x0b\x00\x00\x00{\"ok\":true}\n";
    fwrite(frame, 1, sizeof frame - 1, stdout);
    fflush(stdout);
    sb_ret_int(c, "k", 5);
    return 0;
}

SB_EXPORT int32_t echo(sb_call *c) {
    sb_ret_str(c, "s", sb_str(c, "s", "?"));
    sb_ret_int(c, "n", sb_int(c, "n", -1));
    return 0;
}
"""


def _build(
    out: Path,
    defines: tuple[str, ...] = (),
    header_dir: Path = _NATIVE_DIR,
    source: str = _FIXTURE,
) -> Path:
    assert _CC is not None
    out.parent.mkdir(parents=True, exist_ok=True)
    src = out.with_suffix(".c")
    src.write_text(source)
    built = out.with_name(out.name + ".tmp")
    subprocess.run(
        [
            _CC,
            "-shared",
            "-fPIC",
            "-O1",
            f"-I{header_dir}",
            *defines,
            "-o",
            str(built),
            str(src),
        ],
        check=True,
    )
    built.replace(out)
    return out


def _pid_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@pytest.fixture
def lib_dir(tmp_path: Path) -> Path:
    _build(tmp_path / "fix.so")
    return tmp_path


@pytest.fixture
def owner(lib_dir: Path) -> Iterator[NativeOwner]:
    owner = NativeOwner(lib_dir)
    yield owner
    owner.release()


def _open(owner: NativeOwner, name: str = "fix", timeout: float = 2.0) -> NativeLibrary:
    with owning(owner):
        return native(name, timeout=timeout)


def _buffer(owner: NativeOwner, lib: NativeLibrary, n: int = 8) -> np.ndarray:
    with owning(owner):
        return lib.buffer(n, "f4")


def _settle_reload(lib: NativeLibrary, export: str = "constant") -> dict[str, object]:
    """Drive calls across the settle window so a changed file reloads. The first call only
    notices the change; a dead handle still raises there."""
    with contextlib.suppress(NativeError):
        lib.call(export)
    time.sleep(SETTLE_SECONDS + 0.1)
    return lib.call(export)


def test_results_and_arguments_cross(owner: NativeOwner) -> None:
    """Scalars and a buffer go in, named results come back. Break: drop results -> {}."""
    lib = _open(owner)
    buf = _buffer(owner, lib)
    assert lib.call("fill", buf=buf, value=2.5) == {
        "k": 111,
        "who": "fixture",
        "half": 0.5,
    }
    assert buf.tolist() == [2.5] * 8
    assert lib.call("echo", s="héllo", n=True) == {"s": "héllo", "n": 1}
    assert lib.call("echo", n=np.int64(-7))["n"] == -7


def test_crash_is_isolated_and_buffers_survive(owner: NativeOwner) -> None:
    """Gate 1. The library writes a pattern, then crashes: the test process lives, the error
    is `crashed` naming the export, and the pattern reads back through the array and through
    a fresh map of the buffer's file. Break: load the library in-process -- this process dies."""
    lib = _open(owner)
    buf = _buffer(owner, lib)
    lib.call("fill", buf=buf, value=3.0)
    with pytest.raises(NativeError) as raised:
        lib.call("boom")
    assert raised.value.kind == "crashed"
    assert "boom" in str(raised.value)
    assert "about to crash" in str(raised.value)
    assert buf.tolist() == [3.0] * 8
    path = owner.buffers[0].path
    assert np.fromfile(path, dtype="f4").tolist() == [3.0] * 8


def test_a_crash_is_not_reported_as_a_timeout(owner: NativeOwner) -> None:
    """Gate 2. Under a 0.5 s deadline a crash is `crashed`, seen well inside it. Break: drop
    PR_SET_DUMPABLE in the host. That break shows only where `core_pattern` pipes to a handler
    that holds the dying process (apport, systemd-coredump); elsewhere this passes either way."""
    lib = _open(owner, timeout=0.5)
    start = time.monotonic()
    with pytest.raises(NativeError) as raised:
        lib.call("boom")
    assert raised.value.kind == "crashed"
    assert time.monotonic() - start < 0.3


def test_a_hang_is_stopped_at_the_deadline(owner: NativeOwner) -> None:
    """Gate 3. Break: remove the kill -- the bounded wait in `_stop` leaves the pid alive."""
    lib = _open(owner, timeout=0.2)
    pid = lib.pid
    assert pid is not None
    start = time.monotonic()
    with pytest.raises(NativeError) as raised:
        lib.call("hang")
    assert raised.value.kind == "timeout"
    assert time.monotonic() - start < 0.2 + 1.0
    assert _pid_gone(pid)


def test_a_dead_host_stays_dead_until_something_changes(
    lib_dir: Path, owner: NativeOwner
) -> None:
    """Gate 4. After a crash, calls raise without starting a host; a changed file brings one
    back once it settles, and so does `revive` (what reset does). Break: restart on every call
    (a pid appears); separately, drop `revive` (the last call raises)."""
    lib = _open(owner)
    with pytest.raises(NativeError):
        lib.call("boom")
    for _ in range(10):
        with pytest.raises(NativeError) as raised:
            lib.call("constant")
        assert raised.value.kind == "crashed"
        assert lib.pid is None
    _build(lib_dir / "fix.so", ("-DK=222",))
    assert _settle_reload(lib) == {"k": 222}

    with pytest.raises(NativeError):
        lib.call("boom")
    owner.revive()
    assert lib.call("constant") == {"k": 222}


def test_reload_runs_new_code_and_keeps_buffers(
    lib_dir: Path, owner: NativeOwner
) -> None:
    """Gate 5. A nonzero pattern survives the swap and the new constant lands. Break:
    reallocate buffers on reload -- the pattern is gone (a zero pattern would hide that)."""
    lib = _open(owner)
    buf = _buffer(owner, lib)
    buf[:] = 7.0
    lib.call("first", buf=buf)
    assert buf[0] == 111.0
    _build(lib_dir / "fix.so", ("-DK=222",))
    lib.call("first", buf=buf)
    time.sleep(SETTLE_SECONDS + 0.1)
    lib.call("first", buf=buf)
    assert buf[0] == 222.0
    assert buf[1:].tolist() == [7.0] * 7


def test_an_in_place_overwrite_does_not_reach_the_live_host(
    lib_dir: Path, owner: NativeOwner
) -> None:
    """Gate 6. Overwrite the library IN PLACE (same inode) and call before the settle window:
    the live host still runs the old code. Break: load in place -- the host returns the new
    constant or dies."""
    lib = _open(owner)
    assert lib.call("constant") == {"k": 111}
    other = _build(lib_dir / "other" / "fix.so", ("-DK=222",))
    inode = (lib_dir / "fix.so").stat().st_ino
    shutil.copyfile(other, lib_dir / "fix.so")
    assert (lib_dir / "fix.so").stat().st_ino == inode
    assert lib.call("constant") == {"k": 111}


@pytest.mark.parametrize(
    ("edit", "named"),
    [
        (("int32_t     dtype;", "int32_t     ndim; /*x*/"), "sb_arg.dtype"),
        (("#define SB_ABI 1", "#define SB_ABI 2"), "ABI 2"),
        (("SB_F32 = 1,", "SB_F32 = 9,"), "SB_F32"),
        (("SB__ENTRY_ENUM(SB_U8),", ""), "SB_U8"),
    ],
)
def test_the_abi_is_proven_entry_by_entry(
    tmp_path: Path, edit: tuple[str, str], named: str
) -> None:
    """Gate 7. Each altered header is refused, naming what differs. Break: compare struct
    sizes only (the field swap passes), or only the names the library reports (the removed
    entry passes)."""
    header = (_NATIVE_DIR / "shaderbox_native.h").read_text()
    old, new = edit
    assert old in header
    if old == "int32_t     dtype;":
        # Swap two same-width fields: same size, different offsets.
        header = header.replace(
            "    int32_t     dtype;    /* SB_BUFFER only, else 0                                        */\n"
            "    int32_t     ndim;     /* SB_BUFFER only: 1..4, else 0                                  */\n",
            "    int32_t     ndim;\n    int32_t     dtype;\n",
        )
        assert "int32_t     ndim;\n    int32_t     dtype;" in header
    else:
        header = header.replace(old, new)
    (tmp_path / "hdr").mkdir()
    (tmp_path / "hdr" / "shaderbox_native.h").write_text(header)
    _build(tmp_path / "fix.so", header_dir=tmp_path / "hdr")
    owner = NativeOwner(tmp_path)
    with pytest.raises(NativeError) as raised:
        _open(owner)
    assert raised.value.kind == "load"
    assert named in str(raised.value)


def test_a_library_without_the_contract_is_named(tmp_path: Path) -> None:
    """Gate 7. Break: skip the `sb_abi` check -- the load succeeds."""
    _build(tmp_path / "plain.so", source="int plain(void) { return 1; }\n")
    with pytest.raises(NativeError) as raised:
        _open(NativeOwner(tmp_path), "plain")
    assert "not a ShaderBox library" in str(raised.value)


def test_released_buffers_are_safe_and_pages_are_reserved(owner: NativeOwner) -> None:
    """Gate 9. Writing an array after its handle is released does not kill the process, and a
    buffer's pages are reserved when it is made. Break: back arrays by SharedMemory.buf (the
    write segfaults); separately, replace posix_fallocate with ftruncate (the file is sparse)."""
    lib = _open(owner)
    buf = _buffer(owner, lib, 1 << 16)
    path = owner.buffers[0].path
    if hasattr(os, "posix_fallocate"):
        assert path.stat().st_blocks * 512 >= buf.nbytes
    owner.release()
    assert not path.exists()
    buf[:] = 9.0
    assert float(buf[0]) == 9.0


def test_only_the_handles_own_buffers_pass(owner: NativeOwner, lib_dir: Path) -> None:
    """Gate 13. Break: accept a buffer by its base chain -- the slice passes and the library
    would write from the file's start."""
    lib = _open(owner)
    other = _open(NativeOwner(lib_dir))
    buf = _buffer(owner, lib)
    foreign = other.buffer(8, "f4")
    for bad in (np.zeros(8, "f4"), buf[1:], buf.reshape(2, 4), foreign):
        with pytest.raises(NativeError) as raised:
            lib.call("fill", buf=bad)
        assert "lib.buffer" in str(raised.value)
    with pytest.raises(NativeError, match="NUL"):
        lib.call("echo", s="a\0b")
    with pytest.raises(NativeError, match="int64"):
        lib.call("echo", n=2**64)
    with pytest.raises(NativeError, match="UTF-8"):
        lib.call("echo", s="\ud800")
    other.release()


def test_errors_name_what_went_wrong(owner: NativeOwner, lib_dir: Path) -> None:
    """Gates 11-12. Break: ignore the return value (no `failed`); reorder the lookup (the
    listed order changes)."""
    lib = _open(owner)
    with pytest.raises(NativeError) as raised:
        lib.call("silent")
    assert raised.value.kind == "failed"
    assert "returned 3 without sb_fail" in str(raised.value)
    with pytest.raises(NativeError) as raised:
        lib.call("fill")
    assert "buf must be an f4 buffer" in str(raised.value)
    with pytest.raises(NativeError) as raised:
        lib.call("nope")
    assert raised.value.kind == "missing"
    assert "nope" in str(raised.value)
    with pytest.raises(NativeError) as raised:
        _open(owner, "absent")
    assert raised.value.kind == "load"
    message = str(raised.value)
    assert message.index("absent.so") < message.index("libabsent.so")


def test_the_host_imports_only_the_stdlib(owner: NativeOwner) -> None:
    """Gate 14. Break: start the host through multiprocessing from a parent whose __main__
    imports the app -- numpy, moderngl and shaderbox appear."""
    modules = _open(owner).info()["modules"]
    roots = {name.split(".")[0] for name in modules}
    assert roots.isdisjoint({"shaderbox", "numpy", "moderngl", "glfw", "imgui_bundle"})


def _captured_log() -> tuple[list[str], int]:
    lines: list[str] = []
    sink = logger.add(lambda message: lines.append(str(message)), format="{message}")
    return lines, sink


def _wait_for(lines: list[str], text: str) -> bool:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if any(text in line for line in lines):
            return True
        time.sleep(0.02)
    return False


def test_library_stdout_cannot_corrupt_the_protocol(owner: NativeOwner) -> None:
    """Gate 15. The export writes a well-formed fake reply to stdout; it reaches the log, and
    the real reply is what returns. Break: leave the protocol on fd 1 -- the fake frame is
    read as the reply, or the call is `crashed` with "protocol corrupted"."""
    lines, sink = _captured_log()
    try:
        assert _open(owner).call("shout") == {"k": 5}
        assert _wait_for(lines, '{"ok":true}')
    finally:
        logger.remove(sink)


def test_unflushed_printf_reaches_the_log(owner: NativeOwner) -> None:
    """D12a. A library's printf is fully buffered on a pipe and the host dies by SIGKILL, so
    the host flushes C stdio after every call. Break: drop that flush."""
    lines, sink = _captured_log()
    try:
        _open(owner).call("chatty")
        assert _wait_for(lines, "hello from printf")
    finally:
        logger.remove(sink)


def test_printf_before_a_crash_is_in_the_crash_message(owner: NativeOwner) -> None:
    """D12a. The host makes C stdout unbuffered, so output written just before a crash is not
    lost with the process. Break: drop the setvbuf -- the line dies in the buffer."""
    with pytest.raises(NativeError) as raised:
        _open(owner).call("printcrash")
    assert "last words" in str(raised.value)


def test_a_final_line_without_newline_is_logged(owner: NativeOwner) -> None:
    """D12a. Break: drop the emit of the unterminated remainder at EOF."""
    lines, sink = _captured_log()
    try:
        lib = _open(owner)
        lib.call("noeol")
        owner.release()
        assert _wait_for(lines, "final-no-newline")
    finally:
        logger.remove(sink)


def test_long_output_without_newlines_is_logged_as_it_streams(
    owner: NativeOwner,
) -> None:
    """D12a. Output with no newline (a progress bar) is logged in capped pieces while it
    streams, so the reader holds at most one piece. Break: drop the cap -- nothing is logged
    until the host exits, and the whole stream sits in the reader's memory."""
    lines, sink = _captured_log()
    try:
        lib = _open(owner, timeout=10.0)
        start = time.monotonic()
        lib.call("flood", bytes=8 << 20)
        assert time.monotonic() - start < 2.0
        assert _wait_for(lines, "x" * 1000)
    finally:
        logger.remove(sink)


def test_names_must_be_clean_utf8(owner: NativeOwner) -> None:
    """A surrogate or a NUL in an export or argument name is refused in the parent, and the
    host keeps running. Break: drop the name checks -- the host dies, or a NUL truncates."""
    lib = _open(owner)
    for export, args in (
        ("con\ud800", {}),
        ("echo", {"s\0x": 1}),
        ("echo", {"\ud800": 1}),
    ):
        with pytest.raises(NativeError) as raised:
            lib.call(export, **args)
        assert raised.value.kind == "failed"
    assert lib.call("constant") == {"k": 111}


def test_a_failed_load_is_retried_after_a_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D6 at `native()`. A transient failure heals without a file change. Break: memoize a
    failure until the file changes."""
    _build(tmp_path / "fix.so", ("-DLOAD_FAIL",))
    monkeypatch.setattr(native_library, "RETRY_SECONDS", 0.2)
    started: list[Path] = []
    real_start = NativeLibrary._start

    def _counting(library: NativeLibrary) -> None:
        started.append(library.path)
        real_start(library)

    monkeypatch.setattr(NativeLibrary, "_start", _counting)
    owner = NativeOwner(tmp_path)
    for _ in range(2):
        with pytest.raises(NativeError):
            _open(owner)
    assert len(started) == 1
    time.sleep(0.3)
    with pytest.raises(NativeError):
        _open(owner)
    assert len(started) == 2
    owner.release()


def test_releasing_buffers_never_waits_for_a_busy_host(
    owner: NativeOwner, lib_dir: Path
) -> None:
    """D3. While another thread is in a call, a phase's buffers are released at once and the
    host keeps their mappings until it stops. Break: wait for the lock -- the release blocks
    on every buffer."""
    lib = _open(owner, timeout=1.0)
    instance = NativeOwner(lib_dir)
    with owning(instance):
        for _ in range(10):
            lib.buffer(64, "f4")
    busy = threading.Thread(target=_call_ignoring, args=(lib, "hang"))
    busy.start()
    time.sleep(0.1)
    start = time.monotonic()
    instance.release()
    assert time.monotonic() - start < 0.3
    busy.join(timeout=3.0)


def _call_ignoring(lib: NativeLibrary, export: str) -> None:
    with contextlib.suppress(NativeError):
        lib.call(export)


def test_released_buffers_are_unmapped_by_a_live_host(
    owner: NativeOwner, lib_dir: Path
) -> None:
    """D3. A module-level handle outlives an instance's buffers; their pages return only when
    the host unmaps them. Break: drop the unmap request -- the host keeps the mapping."""
    lib = _open(owner)
    instance = NativeOwner(lib_dir)
    with owning(instance):
        buf = lib.buffer(1024, "f4")
    lib.call("fill", buf=buf)
    assert lib.info()["maps"] == 1
    instance.release()
    assert lib.info()["maps"] == 0


def test_a_failed_load_is_not_retried_until_the_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D6 at `native()`. A broken library named in `update` must not start a host per frame.
    Break: drop the failure memo -- every call starts one."""
    _build(tmp_path / "fix.so", ("-DLOAD_FAIL",))
    started: list[Path] = []
    real_start = NativeLibrary._start

    def _counting(library: NativeLibrary) -> None:
        started.append(library.path)
        real_start(library)

    monkeypatch.setattr(NativeLibrary, "_start", _counting)
    owner = NativeOwner(tmp_path)
    for _ in range(5):
        with pytest.raises(NativeError, match="no license file"):
            _open(owner)
    assert len(started) == 1
    _build(tmp_path / "fix.so")
    assert _open(owner).call("constant") == {"k": 111}
    assert len(started) == 2
    owner.release()


def test_stopping_a_host_does_not_wait_for_its_children(owner: NativeOwner) -> None:
    """D12. A process the library started keeps the host's stderr open. Break: close the
    readers' streams in `_stop` -- the close waits for that pipe's EOF, three seconds here."""
    lib = _open(owner)
    lib.call("spawn_bg")
    start = time.monotonic()
    owner.release()
    assert time.monotonic() - start < 1.5


def test_a_parent_killed_mid_call_leaves_no_host(lib_dir: Path) -> None:
    """D12. A host busy in a call never reads its request pipe's EOF, so it watches for its
    parent. Break: drop the watchdog -- the hung host outlives the parent."""
    script = textwrap.dedent(
        f"""
        import os, signal, threading, time
        from pathlib import Path
        from shaderbox.scripting.native_library import NativeOwner, native, owning
        with owning(NativeOwner(Path({str(lib_dir)!r}))):
            lib = native("fix", timeout=30.0)
        print(lib.pid, flush=True)
        threading.Thread(target=lambda: lib.call("hang"), daemon=True).start()
        time.sleep(0.3)
        os.kill(os.getpid(), signal.SIGKILL)
        """
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == -signal.SIGKILL
    host_pid = int(run.stdout.strip())
    deadline = time.monotonic() + 3.0
    while not _pid_gone(host_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _pid_gone(host_pid)


def test_sb_load_failure_and_hang_are_load_errors(tmp_path: Path) -> None:
    """Gate 16. Break: ignore sb_load's return; drop the deadline on loading."""
    _build(tmp_path / "fail" / "fix.so", ("-DLOAD_FAIL",))
    with pytest.raises(NativeError) as raised:
        _open(NativeOwner(tmp_path / "fail"))
    assert raised.value.kind == "load"
    assert "no license file" in str(raised.value)
    _build(tmp_path / "hang" / "fix.so", ("-DLOAD_HANG",))
    with pytest.raises(NativeError) as raised:
        _open(NativeOwner(tmp_path / "hang"), timeout=0.3)
    assert raised.value.kind == "timeout"


def test_a_missing_file_keeps_the_running_host(
    lib_dir: Path, owner: NativeOwner
) -> None:
    """Gate 16. A linker between unlink and create leaves no file. Break: treat a missing file
    as changed -- the reload fails and the handle dies."""
    lib = _open(owner)
    (lib_dir / "fix.so").unlink()
    lib.call("constant")
    time.sleep(SETTLE_SECONDS + 0.1)
    assert lib.call("constant") == {"k": 111}


def test_handles_memoize_and_refuse_outside_a_script(owner: NativeOwner) -> None:
    """Gate 16. Same name, same phase, same handle; a released handle and a thread's
    `native()` each say what happened. Break: no memo (two handles, two hosts)."""
    first = _open(owner)
    assert _open(owner) is first

    caught: list[NativeError] = []

    def _in_thread() -> None:
        try:
            native("fix")
        except NativeError as e:
            caught.append(e)

    worker = threading.Thread(target=_in_thread)
    worker.start()
    worker.join()
    assert caught and "pass it to a thread" in str(caught[0])

    owner.release()
    with pytest.raises(NativeError, match="released"):
        first.call("constant")


def test_a_killed_parent_leaves_no_host(lib_dir: Path) -> None:
    """Gate 16. An idle host whose parent dies exits, through either of two paths: EOF on its
    request pipe, or the parent watchdog. Break: drop both -- the host outlives the parent."""
    script = textwrap.dedent(
        f"""
        import os, signal, sys
        from pathlib import Path
        from shaderbox.scripting.native_library import NativeOwner, native, owning
        with owning(NativeOwner(Path({str(lib_dir)!r}))):
            lib = native("fix")
        print(lib.pid, flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
        """
    )
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert run.returncode == -signal.SIGKILL
    host_pid = int(run.stdout.strip())
    deadline = time.monotonic() + 3.0
    while not _pid_gone(host_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _pid_gone(host_pid)


def test_debug_handles_have_no_deadline(owner: NativeOwner, lib_dir: Path) -> None:
    """Open question 2: a debug handle's call is not stopped, except under a probe's cap.
    Break: keep the deadline under debug (the hang ends at 2 s); separately, let debug escape
    the probe cap."""
    with owning(owner):
        lib = native("fix", debug=True, timeout=0.2)
    outcome: list[NativeError] = []

    def _hang() -> None:
        try:
            lib.call("hang")
        except NativeError as e:
            outcome.append(e)

    worker = threading.Thread(target=_hang)
    worker.start()
    worker.join(timeout=0.6)
    assert worker.is_alive(), "a debug call was stopped by a deadline"
    owner.release()
    worker.join(timeout=3.0)
    assert not worker.is_alive()

    capped = NativeOwner(lib_dir, timeout_cap=0.25)
    with owning(capped):
        assert native("fix", debug=True).timeout == 0.25
    capped.release()


# ---- the engine: release sites, isolation, errors as script errors -------------------------


_GL_FLOAT = 0x1406

_SCRIPT = """\
from shaderbox.scripting import ScriptBehavior, ScriptContext, native

MODULE_LIB = native("fix")

class Behavior(ScriptBehavior):
    def __init__(self) -> None:
        self.lib = native("fix", timeout=1.5)
        self.buf = self.lib.buffer(4, "f4")

    def update(self, context: ScriptContext) -> dict:
        self.lib.call("fill", buf=self.buf, value=context.frame + 1.0)
        return {"u_x": float(self.buf[0])}
"""


def _u(name: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        name=name, dimension=1, array_length=1, gl_type=_GL_FLOAT, value=0.0
    )


class _FakePass:
    def __init__(self) -> None:
        self.uniform_values: dict[str, object] = {}
        self.script_ready = True

    def get_active_uniforms(self) -> list[types.SimpleNamespace]:
        return [_u("u_x")]


class _FakeDocument:
    def __init__(self) -> None:
        self.passes: dict[str, _FakePass] = {"main": _FakePass()}


def _scripts_dir(tmp_path: Path, script: str = _SCRIPT) -> Path:
    scripts = tmp_path / "scripts"
    _build(scripts / "fix.so")
    (scripts / "script.py").write_text(script)
    return scripts


def _hosts(behavior: PythonBehavior) -> tuple[list[int], list[int]]:
    """(module-phase pids, instance-phase pids) the behavior currently runs."""
    module = [lib.pid for lib in behavior._module_owner.libraries.values() if lib.pid]
    instance_owner = behavior._instance_owner
    instance = (
        [lib.pid for lib in instance_owner.libraries.values() if lib.pid]
        if instance_owner is not None
        else []
    )
    return module, instance


def _instance_buf(behavior: PythonBehavior) -> np.ndarray:
    instance = behavior._instance
    assert instance is not None
    buf = instance.__dict__["buf"]
    assert isinstance(buf, np.ndarray)
    return buf


def _files(behavior: PythonBehavior) -> list[Path]:
    owners = [behavior._module_owner, behavior._instance_owner]
    return [b.path for o in owners if o is not None for b in o.buffers]


def _engine(
    tmp_path: Path, script: str = _SCRIPT
) -> tuple[ScriptEngine, PythonBehavior]:
    engine = ScriptEngine()
    engine.reload("n0", _scripts_dir(tmp_path, script))
    behavior = engine._documents["n0"].behavior
    assert behavior is not None and behavior.error is None, behavior and behavior.error
    return engine, behavior


def _assert_released(pids: list[int], files: list[Path]) -> None:
    assert pids, "the fixture started no host -- nothing to check"
    for pid in pids:
        assert _pid_gone(pid), f"host {pid} still runs"
    for path in files:
        assert not path.exists(), f"{path} was not deleted"


def test_reload_releases_the_old_script(tmp_path: Path) -> None:
    """Gate 8 (reload). Holding the old behavior, so the finalizer cannot stand in. Break:
    drop the release in `engine.reload`."""
    engine, old = _engine(tmp_path)
    module, instance = _hosts(old)
    files = _files(old)
    script = tmp_path / "scripts" / "script.py"
    script.write_text(_SCRIPT + "\n# edited\n")
    os.utime(script, (time.time() + 5, time.time() + 5))
    engine.reload("n0", tmp_path / "scripts")
    assert engine._documents["n0"].behavior is not old
    _assert_released(module + instance, files)
    engine.release_all()


def test_reset_releases_the_instance_and_keeps_the_module(tmp_path: Path) -> None:
    """Gate 8 (reset). Break: release the module phase on reset (its pid goes), or keep the
    instance phase (its pid stays)."""
    engine, behavior = _engine(tmp_path)
    module, instance = _hosts(behavior)
    instance_owner = behavior._instance_owner
    assert instance_owner is not None
    instance_files = [b.path for b in instance_owner.buffers]
    engine.reset("n0")
    _assert_released(instance, instance_files)
    assert all(not _pid_gone(pid) for pid in module)
    assert _hosts(behavior)[1], "reset started no new instance host"
    engine.release_all()


@pytest.mark.parametrize(
    "site",
    [
        "drop_document",
        "release_all",
        "resolve_scripts",
        "script_deleted",
        "scripts_dir_moved",
    ],
)
def test_document_and_engine_teardown_release(tmp_path: Path, site: str) -> None:
    """Gate 8 (drop, engine release, project load). Break: drop the release at that site."""
    engine, behavior = _engine(tmp_path)
    module, instance = _hosts(behavior)
    files = _files(behavior)
    if site == "drop_document":
        engine.drop_document("n0")
    elif site == "release_all":
        engine.release_all()
    elif site == "script_deleted":
        (tmp_path / "scripts" / "script.py").unlink()
        engine.reload("n0", tmp_path / "scripts")
    elif site == "scripts_dir_moved":
        engine.reload("n0", tmp_path / "elsewhere")
    else:
        session = ProjectSession.__new__(ProjectSession)
        session._on_key_warning = lambda *args: None
        session.script_engine = engine
        session.ui_documents = {}
        session._resolve_scripts()
        assert session.script_engine is not engine
    _assert_released(module + instance, files)


@pytest.mark.parametrize("site", ["dry_run", "returned_value_types"])
def test_probes_release_their_fresh_script(
    tmp_path: Path, site: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate 8 (dry run, completion probe). Break: drop the `finally: release()` there."""
    engine, live = _engine(tmp_path)
    made: list[PythonBehavior] = []
    original = engine.fresh_behavior_for

    def _spy(
        document_id: str, native_timeout_cap: float | None = None
    ) -> PythonBehavior | None:
        behavior = original(document_id, native_timeout_cap)
        if behavior is not None:
            made.append(behavior)
            made_pids.extend(list(itertools.chain(*_hosts(behavior))))
            made_files.extend(_files(behavior))
        return behavior

    made_pids: list[int] = []
    made_files: list[Path] = []
    monkeypatch.setattr(engine, "fresh_behavior_for", _spy)
    if site == "dry_run":
        engine.dry_run("n0", _FakeDocument(), (0.0, 0.1), 10)
    else:
        engine.returned_value_types("n0")
    assert made
    _assert_released(made_pids, made_files)
    assert all(not _pid_gone(pid) for pid in list(itertools.chain(*_hosts(live))))
    engine.release_all()


def test_an_export_releases_its_fresh_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate 8 (export). Break: drop the release in `_make_export_isolation`."""
    engine, _ = _engine(tmp_path)
    made: list[PythonBehavior] = []
    original = engine.fresh_behavior_for

    def _spy(
        document_id: str, native_timeout_cap: float | None = None
    ) -> PythonBehavior | None:
        behavior = original(document_id, native_timeout_cap)
        if behavior is not None:
            made.append(behavior)
        return behavior

    monkeypatch.setattr(engine, "fresh_behavior_for", _spy)
    session = ProjectSession.__new__(ProjectSession)
    session.script_engine = engine
    session.ui_documents = {
        "n0": types.SimpleNamespace(document=types.SimpleNamespace(on_pre_render=None))
    }
    with session._make_export_isolation("n0")():
        assert made
        pids = list(itertools.chain(*_hosts(made[0])))
        files = _files(made[0])
    _assert_released(pids, files)
    engine.release_all()


def test_reset_revives_a_crashed_module_handle(tmp_path: Path) -> None:
    """D6 through the engine. Break: drop `revive` from `PythonBehavior.reset`."""
    engine, behavior = _engine(tmp_path)
    module_lib = next(iter(behavior._module_owner.libraries.values()))
    with pytest.raises(NativeError):
        module_lib.call("boom")
    engine.reset("n0")
    assert module_lib.call("constant") == {"k": 111}
    engine.release_all()


def test_reload_stops_the_old_hosts_before_the_new_script_loads(tmp_path: Path) -> None:
    """A library that holds something exclusive in `sb_load` loads again after a script edit.
    Break: build the new behavior before releasing the old -- `device busy`."""
    lock = tmp_path / "device.lock"
    scripts = tmp_path / "scripts"
    _build(scripts / "fix.so", (f'-DLOCKFILE="{lock}"',))
    script = (
        'from shaderbox.scripting import ScriptBehavior, native\nLIB = native("fix")\n'
    )
    script += (
        "class Behavior(ScriptBehavior):\n    def update(self, context) -> dict:\n"
    )
    script += "        return {}\n"
    (scripts / "script.py").write_text(script)
    engine = ScriptEngine()
    engine.reload("n0", scripts)
    first = engine._documents["n0"].behavior
    assert first is not None and first.error is None, first and first.error
    (scripts / "script.py").write_text(script + "# edited\n")
    os.utime(scripts / "script.py", (time.time() + 5, time.time() + 5))
    engine.reload("n0", scripts)
    second = engine._documents["n0"].behavior
    assert second is not None and second is not first
    assert second.error is None, second.error
    engine.release_all()


def test_a_fresh_script_cannot_touch_the_live_buffers(tmp_path: Path) -> None:
    """Gate 10. A fresh behavior's native calls write its own buffer, never the live one.
    Break: share handles across behaviors -- the live buffer changes."""
    engine, live = _engine(tmp_path)
    document = _FakeDocument()
    engine.tick(
        "n0", document, types.SimpleNamespace(t=0.0, dt=0.0, frame=0, mouse=None)
    )
    live_buf = _instance_buf(live)
    assert live_buf.tolist() == [1.0] * 4
    fresh = engine.fresh_behavior_for("n0")
    assert fresh is not None
    try:
        fresh.run(ScriptContext(t=0.0, dt=0.0, frame=41))
        assert _instance_buf(fresh).tolist() == [42.0] * 4
        assert live_buf.tolist() == [1.0] * 4
    finally:
        fresh.release()
        engine.release_all()


def test_an_uncaught_native_error_is_a_script_error_on_the_call_line(
    tmp_path: Path,
) -> None:
    """Gate 11. Break: let NativeError escape the tick, or lose the user's line."""
    script = _SCRIPT.replace(
        'self.lib.call("fill", buf=self.buf, value=context.frame + 1.0)',
        'self.lib.call("silent")',
    )
    engine, _ = _engine(tmp_path, script)
    engine.tick(
        "n0", _FakeDocument(), types.SimpleNamespace(t=0.0, dt=0.0, frame=0, mouse=None)
    )
    errors = [e for key, e in engine.errors.items() if key[0] == "n0"]
    assert errors, "the native failure did not become a script error"
    error = errors[0]
    assert "without sb_fail" in error.message
    call_line = script.splitlines().index('        self.lib.call("silent")')
    assert error.line == call_line
    engine.release_all()


# ---- the shipped contract ------------------------------------------------------------------


def test_the_shipped_c_example_builds_and_loads(tmp_path: Path) -> None:
    """Gate 17. Break: rename a table entry in the header -- the load refuses it."""
    assert _CC is not None
    subprocess.run(
        [
            _CC,
            "-shared",
            "-fPIC",
            "-O1",
            f"-I{_NATIVE_DIR}",
            "-o",
            str(tmp_path / "example.so"),
            str(_NATIVE_DIR / "example_c" / "example.c"),
        ],
        check=True,
    )
    _check_example(tmp_path)


@pytest.mark.skipif(
    _ODIN is None, reason="no odin compiler: the Odin example is SKIPPED"
)
def test_the_shipped_odin_example_builds_and_loads(tmp_path: Path) -> None:
    """Gate 17. Break: rename a table entry in the Odin binding -- the load refuses it."""
    assert _ODIN is not None
    subprocess.run(
        [
            _ODIN,
            "build",
            str(_NATIVE_DIR / "example_odin"),
            "-build-mode:shared",
            f"-out:{tmp_path / 'example.so'}",
        ],
        check=True,
    )
    _check_example(tmp_path)


def _check_example(directory: Path) -> None:
    owner = NativeOwner(directory)
    try:
        lib = _open(owner, "example")
        values = _buffer(owner, lib, 4)
        values[:] = [1.0, 2.0, 3.0, 4.0]
        assert lib.call("scale", values=values, factor=2.0) == {"sum": 20.0}
        assert values.tolist() == [2.0, 4.0, 6.0, 8.0]
    finally:
        owner.release()
