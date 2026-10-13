"""The child process a native library runs in (feature 109).

Started by `native.py` as `python -I -S <this file>`, by PATH, so no package `__init__` runs
and nothing from site-packages loads: the host imports the standard library and nothing
else, and a gate asserts that. It never touches GL.

The protocol is length-prefixed JSON frames: the parent writes requests to the host's stdin,
and the host replies on the file descriptor that was stdout when it started. Before loading
the library the host points fd 1 at stderr, so a library's `printf` lands in the log and
cannot corrupt a reply.

The ABI lives here and only here: the ctypes structures below are what the host builds, and
`LAYOUT` is what a library's `sb_layout_*` table must equal. `native.py` imports this module
for those definitions, which is safe because it is stdlib-only.
"""

import ctypes
import json
import mmap
import os
import struct
import sys
from typing import IO, Any

SB_ABI = 1

SB_NUMBER = 1
SB_INTEGER = 2
SB_STRING = 3
SB_BUFFER = 4

SB_F32 = 1
SB_F64 = 2
SB_I32 = 3
SB_U32 = 4
SB_I64 = 5
SB_U8 = 6

# numpy dtype string -> ABI dtype code.
DTYPE_CODES: dict[str, int] = {
    "f4": SB_F32,
    "f8": SB_F64,
    "i4": SB_I32,
    "u4": SB_U32,
    "i8": SB_I64,
    "u1": SB_U8,
}

MAX_DIMS = 4
ERROR_CAP = 1024
RESULTS_CAP = 32


class SbArg(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("kind", ctypes.c_int32),
        ("dtype", ctypes.c_int32),
        ("ndim", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        ("shape", ctypes.c_int64 * MAX_DIMS),
        ("data", ctypes.c_void_p),
        ("number", ctypes.c_double),
        ("integer", ctypes.c_int64),
    ]


class SbCall(ctypes.Structure):
    _fields_ = [
        ("abi", ctypes.c_int32),
        ("nargs", ctypes.c_int32),
        ("args", ctypes.POINTER(SbArg)),
        ("error", ctypes.c_void_p),
        ("error_cap", ctypes.c_int32),
        ("nresults", ctypes.c_int32),
        ("results", ctypes.POINTER(SbArg)),
        ("results_cap", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
    ]


def _layout() -> dict[str, int]:
    table: dict[str, int] = {
        "sizeof.sb_arg": ctypes.sizeof(SbArg),
        "sizeof.sb_call": ctypes.sizeof(SbCall),
    }
    for struct_name, struct_type in (("sb_arg", SbArg), ("sb_call", SbCall)):
        for field_name, *_ in struct_type._fields_:
            table[f"{struct_name}.{field_name}"] = getattr(
                struct_type, field_name
            ).offset
    for name in (
        "SB_NUMBER",
        "SB_INTEGER",
        "SB_STRING",
        "SB_BUFFER",
        "SB_F32",
        "SB_F64",
        "SB_I32",
        "SB_U32",
        "SB_I64",
        "SB_U8",
    ):
        table[name] = globals()[name]
    return table


LAYOUT: dict[str, int] = _layout()

# prctl options (linux/prctl.h).
_PR_SET_DUMPABLE = 4
_PR_SET_PTRACER = 0x59616D61
_PR_SET_PTRACER_ANY = ctypes.c_ulong(-1).value


def read_frame(stream: IO[bytes]) -> Any:
    """One length-prefixed JSON frame, or None at EOF."""
    head = stream.read(4)
    if len(head) < 4:
        return None
    (length,) = struct.unpack("<I", head)
    body = stream.read(length)
    if len(body) < length:
        return None
    return json.loads(body)


def write_frame(stream: IO[bytes], message: object) -> None:
    body = json.dumps(message).encode("utf-8")
    stream.write(struct.pack("<I", len(body)) + body)
    stream.flush()


def _protect(debug: bool) -> None:
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None)
        if debug:
            # A debugger attaches by pid; yama's ptrace_scope=1 otherwise refuses any
            # process that is not our ancestor.
            libc.prctl(_PR_SET_PTRACER, ctypes.c_ulong(_PR_SET_PTRACER_ANY), 0, 0, 0)
        else:
            # A crash then skips the core-dump handler (apport, systemd-coredump), which held
            # the dying process for about a second -- longer than a short deadline, so the
            # parent read a crash as a timeout.
            libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0)
    elif sys.platform == "win32":
        sem_failcriticalerrors = 0x0001
        sem_nogpfaulterrorbox = 0x0002
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.SetErrorMode(sem_failcriticalerrors | sem_nogpfaulterrorbox)


def _redirect_stdout() -> IO[bytes]:
    # The protocol keeps the ORIGINAL stdout; fd 1 becomes stderr so anything the library
    # prints is log text, never a reply.
    protocol = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    if sys.platform == "win32":
        # A library writing through GetStdHandle bypasses the C runtime's fd 1.
        kernel32 = ctypes.WinDLL("kernel32")
        std_output_handle = ctypes.c_ulong(-11).value
        std_error_handle = ctypes.c_ulong(-12).value
        kernel32.SetStdHandle(
            std_output_handle, kernel32.GetStdHandle(std_error_handle)
        )
    return protocol


class _Host:
    def __init__(self) -> None:
        self.lib: ctypes.CDLL | None = None
        self.maps: dict[str, tuple[mmap.mmap, int]] = {}
        self.error = ctypes.create_string_buffer(ERROR_CAP)
        self.results = (SbArg * RESULTS_CAP)()

    def load(self, path: str) -> dict[str, Any]:
        try:
            lib = ctypes.CDLL(path)
        except OSError as e:
            return {"ok": False, "kind": "load", "message": f"cannot load: {e}"}
        if sys.platform != "win32":
            # The copy is ours alone; unlinking it now means a host killed later leaves
            # nothing behind. The parent deletes it too, for a host that dies in dlopen.
            try:
                os.unlink(path)
            except OSError:
                pass
        refusal = _verify(lib)
        if refusal is not None:
            return {"ok": False, "kind": "load", "message": refusal}
        self.lib = lib
        if _has_export(lib, "sb_load"):
            reply = self._invoke("sb_load", [], no_args=True)
            if not reply["ok"] or reply["rc"] != 0:
                message = reply.get("message") or reply.get("error") or ""
                return {
                    "ok": False,
                    "kind": "load",
                    "message": f"sb_load failed: {message}",
                }
        return {"ok": True, "pid": os.getpid()}

    def _map(self, path: str, size: int) -> int:
        cached = self.maps.get(path)
        if cached is not None and len(cached[0]) == size:
            return cached[1]
        fd = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
        try:
            mapping = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        self.maps[path] = (mapping, address)
        return address

    def _invoke(
        self, export: str, args: list[list[Any]], no_args: bool = False
    ) -> dict[str, Any]:
        assert self.lib is not None
        if not _has_export(self.lib, export):
            return {"ok": False, "kind": "missing", "message": f"no export `{export}`"}
        function = getattr(self.lib, export)
        function.restype = ctypes.c_int32
        function.argtypes = [ctypes.POINTER(SbCall)]

        keep: list[object] = []
        packed = (SbArg * max(len(args), 1))()
        for i, (name, kind, value) in enumerate(args):
            arg = packed[i]
            encoded = name.encode("utf-8")
            keep.append(encoded)
            arg.name = encoded
            arg.kind = kind
            if kind == SB_NUMBER:
                arg.number = float(value)
            elif kind == SB_INTEGER:
                arg.integer = int(value)
                arg.number = float(value)
            elif kind == SB_STRING:
                text = ctypes.create_string_buffer(value.encode("utf-8"))
                keep.append(text)
                arg.data = ctypes.addressof(text)
            elif kind == SB_BUFFER:
                arg.dtype = value["dtype"]
                arg.ndim = len(value["shape"])
                for d, extent in enumerate(value["shape"]):
                    arg.shape[d] = extent
                arg.data = self._map(value["path"], value["size"])

        call = SbCall()
        call.abi = SB_ABI
        call.nargs = 0 if no_args else len(args)
        call.args = (
            ctypes.POINTER(SbArg)()
            if no_args
            else ctypes.cast(packed, ctypes.POINTER(SbArg))
        )
        self.error[0] = b"\0"
        call.error = ctypes.addressof(self.error)
        call.error_cap = ERROR_CAP
        call.nresults = 0
        call.results = ctypes.cast(self.results, ctypes.POINTER(SbArg))
        call.results_cap = RESULTS_CAP

        rc = function(ctypes.byref(call))

        results: dict[str, Any] = {}
        for i in range(max(0, min(call.nresults, RESULTS_CAP))):
            result = self.results[i]
            if not result.name:
                continue
            key = result.name.decode("utf-8", "replace")
            if result.kind == SB_NUMBER:
                results[key] = result.number
            elif result.kind == SB_INTEGER:
                results[key] = result.integer
            elif result.kind == SB_STRING and result.data:
                results[key] = ctypes.string_at(result.data).decode("utf-8", "replace")
        error = self.error.value.decode("utf-8", "replace") if rc != 0 else ""
        return {"ok": True, "rc": rc, "error": error, "results": results}

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request["op"]
        if op == "call":
            return self._invoke(request["export"], request["args"])
        if op == "info":
            return {"ok": True, "pid": os.getpid(), "modules": sorted(sys.modules)}
        return {"ok": False, "kind": "load", "message": f"unknown op {op!r}"}


def _has_export(lib: ctypes.CDLL, name: str) -> bool:
    try:
        getattr(lib, name)
    except AttributeError:
        return False
    return True


def _verify(lib: ctypes.CDLL) -> str | None:
    """None when the library's table equals `LAYOUT`, else the refusal naming the entry."""
    if not _has_export(lib, "sb_abi"):
        return "not a ShaderBox library: no `sb_abi` export -- add `SB_LIBRARY()`"
    lib.sb_abi.restype = ctypes.c_int32
    version = lib.sb_abi()
    if version != SB_ABI:
        return (
            f"built against ABI {version}, ShaderBox speaks ABI {SB_ABI} -- "
            "rebuild with the current header"
        )
    for name in ("sb_layout_name", "sb_layout_value"):
        if not _has_export(lib, name):
            return f"no `{name}` export -- add `SB_LIBRARY()`"
    lib.sb_layout_name.restype = ctypes.c_char_p
    lib.sb_layout_name.argtypes = [ctypes.c_int32]
    lib.sb_layout_value.restype = ctypes.c_int64
    lib.sb_layout_value.argtypes = [ctypes.c_int32]
    theirs: dict[str, int] = {}
    for i in range(1024):
        raw = lib.sb_layout_name(i)
        if raw is None:
            break
        theirs[raw.decode("utf-8", "replace")] = lib.sb_layout_value(i)
    missing = sorted(LAYOUT.keys() - theirs.keys())
    if missing:
        return f"ABI mismatch: the library's layout table has no `{missing[0]}`"
    extra = sorted(theirs.keys() - LAYOUT.keys())
    if extra:
        return f"ABI mismatch: the library's layout table has an unknown `{extra[0]}`"
    for name in sorted(LAYOUT):
        if theirs[name] != LAYOUT[name]:
            return (
                f"ABI mismatch at `{name}`: the library has {theirs[name]}, "
                f"ShaderBox has {LAYOUT[name]}"
            )
    return None


def main() -> None:
    protocol = _redirect_stdout()
    requests = sys.stdin.buffer
    first = read_frame(requests)
    if first is None:
        return
    _protect(bool(first.get("debug")))
    host = _Host()
    write_frame(protocol, host.load(first["path"]))
    if host.lib is None:
        return
    while True:
        request = read_frame(requests)
        if request is None:
            return
        write_frame(protocol, host.handle(request))


if __name__ == "__main__":
    main()
