"""The frame profiler: a tree of CPU and GPU spans for one frame (feature 088).

A `Profiler` opens a root per frame and hands back a `FrameProfile`; `cpu(name)` and
`gpu(name)` are context managers that push a `Span` under whatever span is open. A span's
parent is decided at entry, so a document rendered inside another document's pass nests by
construction and nothing here knows what a document is.

Imports `moderngl` for the GPU timer queries and nothing else -- no imgui, no `App`, no
`Document` -- so any caller can take one. The GL context is resolved at the first `gpu()`
entry of an ENABLED profiler, never at construction: `NULL_PROFILER` is built at import time
and importing this module must work with no window.

Two GL facts shape the module, both measured (RTX 3090, GL 3.3 core, moderngl 5.12) and both
silent when violated:

- `GL_TIME_ELAPSED` queries do NOT nest. Two of them one inside the other leave the outer
  reading garbage, the inner reading 0 and `ctx.error` at `GL_INVALID_OPERATION`, with no
  exception -- and `ui.update_and_draw` calls `clear_errors()` every frame, so even that
  late signal is gone. `gpu()` asserts instead.
- Reading a query blocks until the GPU has drained past it. Read one frame late under a real
  fragment load the read stalled 22.3 ms; read two frames late, 0.009 ms. So each span PATH
  owns a three-deep ring of query objects, frame N runs in slot `N % RING_DEPTH`, and frame
  N's numbers are read at frame N+2's `begin_frame` -- a `FrameProfile` is complete two
  frames after it closes.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import moderngl

# Frames of margin between a query's block and its read. Measured, not tunable: a two-deep
# ring stalled 22.3 ms on a loaded GPU, three-deep 0.009 ms, four-deep 0.007 ms.
RING_DEPTH: int = 3

_MS_PER_NS: float = 1e-6
_MS_PER_S: float = 1e3


@dataclass
class Span:
    """One measured block: its name, what it cost, and the spans opened inside it.

    `cpu_ms` is wall time around the block. `gpu_ms` is `None` on a CPU span, and on a GPU
    span until the frame's queries are read two frames later. `count` is how many times the
    block's work ran inside the one span (an iterated pass draws N times in its one turn).
    """

    name: str
    cpu_ms: float = 0.0
    gpu_ms: float | None = None
    count: int = 1
    children: list["Span"] = field(default_factory=list)


@dataclass
class FrameProfile:
    """One frame's span tree; `complete` once its GPU numbers have been read back."""

    root: Span
    index: int = 0
    complete: bool = False

    @property
    def cpu_ms(self) -> float:
        return self.root.cpu_ms

    @property
    def gpu_ms(self) -> float:
        return gpu_total(self.root)

    @property
    def children(self) -> list[Span]:
        return self.root.children


def gpu_total(span: Span) -> float:
    """Every GPU span's time in the subtree. GPU spans cannot nest, so nothing double-counts."""
    total: float = span.gpu_ms or 0.0
    for child in span.children:
        total += gpu_total(child)
    return total


def other_ms(span: Span) -> float:
    """The part of `span`'s wall its children do not account for."""
    return max(0.0, span.cpu_ms - sum(child.cpu_ms for child in span.children))


class Profiler:
    """Builds one `FrameProfile` per frame while `enabled`; a null object while not.

    A disabled profiler's `cpu` / `gpu` cost one attribute read, create no query and build no
    tree -- which is what lets `Document.render` default to one and every export, probe and
    test stay silent. Disabling drops the whole query ring, the module's only eviction rule:
    `moderngl.Query` has neither `release()` nor `__del__`, so a query is a permanent GL name
    and a pass rename or a newly opened document would otherwise grow the ring for the life
    of the process.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._enabled: bool = enabled
        self._gl: moderngl.Context | None = None
        self._root: Span | None = None
        self._root_started: float = 0.0
        self._stack: list[Span] = []
        # The open stack as `name#ordinal` segments, parallel to `_stack` minus the root.
        self._path: list[str] = []
        self._frame_index: int = 0
        self._gpu_open: bool = False
        self._ring: dict[tuple[str, ...], list[moderngl.Query | None]] = {}
        # Frames awaiting their read: ring slot -> (profile, its GPU spans by path).
        self._pending: dict[int, tuple[FrameProfile, dict[tuple[str, ...], Span]]] = {}
        # This frame's GPU spans by path, so the read two frames on knows where to write.
        self._gpu_spans: dict[tuple[str, ...], Span] = {}
        # (parent path, name) -> how many siblings of that name have opened this frame, so
        # two same-named spans under one parent are two distinct spans with two ring keys.
        self._ordinals: dict[tuple[tuple[str, ...], str], int] = {}
        # The newest profile whose GPU numbers have been read back: two frames behind live
        # while GPU spans are open, this frame's own once a frame opens none.
        self.last_complete: FrameProfile | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        if value == self._enabled:
            return
        self._enabled = value
        if not value:
            self._ring = {}
            self._pending = {}
            self._gpu_spans = {}
            self.last_complete = None
            self._ordinals = {}
            self._stack = []
            self._path = []
            self._root = None
            self._gpu_open = False
            self._gl = None

    def begin_frame(self) -> None:
        """Open this frame's root, first reading back the frame two behind it.

        Frame F reads slot `(F + 1) % RING_DEPTH`, which frame F-2 wrote: two frames of
        margin, and one full frame still standing between the read and that slot's reuse at
        frame F+1.
        """
        if not self._enabled:
            return
        self._read_pending((self._frame_index + 1) % RING_DEPTH)
        self._root = Span("frame")
        self._stack = [self._root]
        self._path = []
        self._gpu_spans = {}
        self._ordinals = {}
        self._root_started = time.perf_counter()

    def end_frame(self) -> FrameProfile | None:
        """Close the root and return the frame's profile, incomplete until its read lands."""
        if not self._enabled or self._root is None:
            return None
        self._root.cpu_ms = (time.perf_counter() - self._root_started) * _MS_PER_S
        profile = FrameProfile(self._root, self._frame_index)
        if self._gpu_spans:
            self._pending[self._frame_index % RING_DEPTH] = (profile, self._gpu_spans)
        else:
            profile.complete = True
            self._publish(profile)
        self._frame_index += 1
        self._stack = []
        self._path = []
        self._root = None
        return profile

    @contextmanager
    def frame(self) -> Iterator[None]:
        """`begin_frame` / `end_frame` as a block, so an abort still closes the root.

        The root closes whether or not the body returned early, which is what keeps the ring
        aligned with the frames that actually drew.
        """
        self.begin_frame()
        try:
            yield
        finally:
            self.end_frame()

    @contextmanager
    def cpu(self, name: str) -> Iterator[None]:
        if not self._enabled or not self._stack:
            yield
            return
        span = self._push(name)
        started = time.perf_counter()
        try:
            yield
        finally:
            span.cpu_ms = (time.perf_counter() - started) * _MS_PER_S
            self._pop()

    @contextmanager
    def gpu(self, name: str, count: int = 1) -> Iterator[None]:
        if not self._enabled or not self._stack:
            yield
            return
        assert not self._gpu_open, (
            f"GPU span {name!r} opened inside another: GL_TIME_ELAPSED queries do not nest, "
            "and a nested pair reports wrong numbers with no error left to read"
        )
        if self._gl is None:
            self._gl = moderngl.get_context()
        span = self._push(name)
        span.count = count
        path = tuple(self._path)
        self._gpu_spans[path] = span
        query = self._query_for(path)
        started = time.perf_counter()
        self._gpu_open = True
        query.mglo.begin()
        try:
            yield
        finally:
            query.mglo.end()
            self._gpu_open = False
            span.cpu_ms = (time.perf_counter() - started) * _MS_PER_S
            self._pop()

    def _push(self, name: str) -> Span:
        key = (tuple(self._path), name)
        ordinal = self._ordinals.get(key, 0)
        self._ordinals[key] = ordinal + 1
        span = Span(name)
        self._stack[-1].children.append(span)
        self._stack.append(span)
        self._path.append(f"{name}#{ordinal}")
        return span

    def _pop(self) -> None:
        self._stack.pop()
        self._path.pop()

    def _query_for(self, path: tuple[str, ...]) -> moderngl.Query:
        assert self._gl is not None
        ring: list[moderngl.Query | None] | None = self._ring.get(path)
        if ring is None:
            ring = [None for _ in range(RING_DEPTH)]
            self._ring[path] = ring
        index = self._frame_index % RING_DEPTH
        query = ring[index]
        if query is None:
            query = self._gl.query(time=True)
            ring[index] = query
        return query

    def _publish(self, profile: FrameProfile) -> None:
        """Hand `profile` over as the newest complete one, never going backwards in time.

        A frame that opens no GPU span completes the instant it closes, while the frame two
        behind it completes at the NEXT `begin_frame` -- so without the ordering check an
        aborted frame would be replaced by an older one on the very next frame.
        """
        if self.last_complete is None or profile.index >= self.last_complete.index:
            self.last_complete = profile

    def _read_pending(self, slot: int) -> None:
        """Give the frame that owned `slot` its GPU milliseconds, before the slot is reused."""
        pending = self._pending.pop(slot, None)
        if pending is None:
            return
        profile, gpu_spans = pending
        for path, span in gpu_spans.items():
            ring = self._ring.get(path)
            query = ring[slot] if ring is not None else None
            if query is not None:
                span.gpu_ms = query.elapsed * _MS_PER_NS
        profile.complete = True
        self._publish(profile)


NULL_PROFILER: Profiler = Profiler(enabled=False)
