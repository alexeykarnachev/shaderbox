"""Headless verification for the render/publish copilot tools (feature 020·18).

Covers the pure-logic decisions the GUI smoke can't reach (`scripts/smoke.py` never builds
the registry or runs a turn):

  H1. The registry builds with the render/publish tools: 13 eager specs (9 shader + 4 new),
      each new tool's arg schema builds, render/publish tools are GatePolicy.ALWAYS, the
      publish tools carry a `precheck`.
  H2. Render-to-file (needs a live GL context, like smoke): the App render closure produces a
      real file at the REQUESTED size (the FIXED_DIMS/RENDER_AT_TARGET path), not the canvas
      size — the regression that an inert preset would silently render at canvas size.
  H3. Precheck logic: each publish tool's precheck returns a guided handoff when not connected
      / no pack, and None when ready.

Usage: `uv run python scripts/copilot_render_check.py` (exit 0 on success, non-zero on failure).
H1/H3 are GL-free + network-free; H2 opens an invisible glfw window (like scripts/smoke.py).
"""

import sys
import tempfile
from pathlib import Path

import glfw
from loguru import logger
from platformdirs import user_data_dir

from shaderbox.app import App
from shaderbox.constants import VIDEO_RESOLUTION_ALIGNMENT
from shaderbox.logging_setup import configure_logging
from shaderbox.tabs import share_state

from shaderbox.copilot.capabilities import (
    CompileErrorInfo,
    CopilotCapabilities,
    DeleteNodeResult,
    EditResult,
    GrepHit,
    LibCatalogEntry,
    LibFunctionBody,
    NodeTreeEntry,
    PublishResult,
    RenderResult,
    SetUniformResult,
    ShaderView,
)
from shaderbox.copilot.tools.base import GatePolicy
from shaderbox.copilot.tools.registry import build_registry

_RENDER_PUBLISH_TOOLS = {
    "render_image",
    "render_video",
    "publish_telegram",
    "publish_youtube",
}
_PUBLISH_TOOLS = {"publish_telegram", "publish_youtube"}


def _stub_caps(
    *, telegram_connected: bool, telegram_pack: bool, youtube_connected: bool
) -> CopilotCapabilities:
    # Only the precheck reads vary across H3 cases; the rest are inert stubs satisfying the
    # frozen dataclass (no tool here actually renders/publishes — that needs GL + a network).
    def _no_views(_ids: list[str]) -> list[ShaderView]:
        return []

    def _no_tree() -> list[NodeTreeEntry]:
        return []

    def _no_catalog() -> list[LibCatalogEntry]:
        return []

    def _no_grep(_q: str) -> list[GrepHit]:
        return []

    def _no_lib(_n: list[str]) -> list[LibFunctionBody]:
        return []

    def _edit(_a: str, _b: str, _c: bool, _t: str) -> EditResult:
        return EditResult(matches=0, errors=[])

    def _line(_s: int, _e: int, _t: str, _g: str) -> EditResult:
        return EditResult(matches=0, errors=[])

    def _set(_n: str, _v: object, _node: str) -> SetUniformResult:
        return SetUniformResult(ok=False, error="n/a")

    def _create(_n: str, _s: str, _w: bool) -> tuple[str, list[CompileErrorInfo]]:
        return "", []

    def _delete(_node: str) -> DeleteNodeResult:
        return DeleteNodeResult(ok=False, error="n/a")

    def _render_image(_n: str, _w: int, _h: int) -> RenderResult:
        return RenderResult(ok=False, error="n/a")

    def _render_video(_n: str, _s: float, _f: int, _w: int, _h: int) -> RenderResult:
        return RenderResult(ok=False, error="n/a")

    def _pub_tg(_n: str, _e: str) -> PublishResult:
        return PublishResult(ok=False, error="n/a", kind="telegram")

    def _pub_yt(_n: str, _t: str, _d: str, _s: bool) -> PublishResult:
        return PublishResult(ok=False, error="n/a", kind="youtube")

    return CopilotCapabilities(
        node_tree=_no_tree,
        lib_catalog=_no_catalog,
        read_shaders=_no_views,
        grep=_no_grep,
        read_lib=_no_lib,
        apply_shader_edit=_edit,
        apply_line_edit=_line,
        set_uniform=_set,
        create_node=_create,
        delete_node=_delete,
        render_image=_render_image,
        render_video=_render_video,
        publish_telegram=_pub_tg,
        publish_youtube=_pub_yt,
        has_current_node=lambda: True,
        telegram_connected=lambda: telegram_connected,
        youtube_connected=lambda: youtube_connected,
        telegram_has_default_pack=lambda: telegram_pack,
    )


def _check_registry_builds() -> None:
    caps = _stub_caps(
        telegram_connected=True, telegram_pack=True, youtube_connected=True
    )
    registry = build_registry(caps)
    eager = registry.eager_specs()
    assert len(eager) == 13, f"expected 13 eager tools (9 shader + 4 new), got {len(eager)}"

    names = {s.name for s in eager}
    assert _RENDER_PUBLISH_TOOLS <= names, (
        f"render/publish tools missing from eager set: {_RENDER_PUBLISH_TOOLS - names}"
    )

    # Each new tool's arg schema builds (a pydantic Field/constraint typo would raise here).
    for spec in eager:
        if spec.name in _RENDER_PUBLISH_TOOLS:
            assert isinstance(spec.parameters, dict) and spec.parameters, (
                f"{spec.name} produced no JSON schema"
            )

    for name in _RENDER_PUBLISH_TOOLS:
        assert registry.requires_gate_always(name), (
            f"{name} must be GatePolicy.ALWAYS"
        )
    logger.info("H1 ok: 13 eager tools build, render/publish are ALWAYS-gated, schemas build")


def _check_precheck_logic() -> None:
    # Ready: prechecks return None.
    ready = build_registry(
        _stub_caps(telegram_connected=True, telegram_pack=True, youtube_connected=True)
    )
    assert ready.precheck("publish_telegram", {}) is None, (
        "telegram precheck should pass when connected + pack set"
    )
    assert ready.precheck("publish_youtube", {}) is None, (
        "youtube precheck should pass when connected"
    )
    # Render tools have no precheck.
    assert ready.precheck("render_image", {}) is None, "render_image has no precheck"

    # Telegram not connected -> a guided handoff (no gate fires for this call).
    no_tg = build_registry(
        _stub_caps(telegram_connected=False, telegram_pack=False, youtube_connected=True)
    )
    msg = no_tg.precheck("publish_telegram", {})
    assert msg is not None and "connect" in msg.lower(), (
        f"unconnected telegram precheck should hand off: {msg!r}"
    )

    # Telegram connected but no pack -> a different handoff.
    no_pack = build_registry(
        _stub_caps(telegram_connected=True, telegram_pack=False, youtube_connected=True)
    )
    msg = no_pack.precheck("publish_telegram", {})
    assert msg is not None and "pack" in msg.lower(), (
        f"no-pack telegram precheck should hand off: {msg!r}"
    )

    # YouTube not connected -> a handoff.
    no_yt = build_registry(
        _stub_caps(telegram_connected=True, telegram_pack=True, youtube_connected=False)
    )
    msg = no_yt.precheck("publish_youtube", {})
    assert msg is not None and "connect" in msg.lower(), (
        f"unconnected youtube precheck should hand off: {msg!r}"
    )
    logger.info("H3 ok: publish prechecks hand off when unready, pass when ready")


_DEV_PROJECT = Path(__file__).resolve().parent.parent / "projects" / "dev"
_POINTER = Path(user_data_dir("shaderbox")) / "project_dir"


def _align(v: int) -> int:
    a = VIDEO_RESOLUTION_ALIGNMENT
    return max(a, (v + a - 1) // a * a)


def _check_render_to_file() -> None:
    # H2: the App render closure honors the REQUESTED size. Builds a real App + GL context
    # (like smoke), renders the current node at a size that differs from its canvas, asserts
    # the output file exists + its dimensions equal the requested size (snapped to alignment).
    if not _DEV_PROJECT.exists():
        logger.warning("H2 skipped: dev fixture project not found")
        return
    saved: str | None = _POINTER.read_text() if _POINTER.exists() else None
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    app: App | None = None
    try:
        app = App(project_dir=_DEV_PROJECT)
        node_id: str = app.current_node_id
        assert node_id and node_id in app.ui_nodes, "no current node in the dev fixture"
        ui_node = app.ui_nodes[node_id]
        cw, ch = ui_node.node.canvas.texture.size
        # A size deliberately != the canvas, not already aligned, to prove the request drives it.
        req_w, req_h = cw + 37, ch + 37
        preset = app._copilot_render_preset(False, None, req_w, req_h)
        with tempfile.TemporaryDirectory() as tmp:
            out: Path = Path(tmp) / "h2.png"
            art = share_state.render_to(ui_node.node, preset, 0.0, out)
            assert art is not None and out.exists(), "render produced no file"
            exp: tuple[int, int] = (_align(req_w), _align(req_h))
            assert art.size == exp, (
                f"render ignored the requested size: got {art.size}, expected {exp} "
                f"(canvas was {(cw, ch)} — an inert preset would render at canvas size)"
            )
            logger.info(
                f"H2 ok: render honors the requested size ({art.size}, canvas {(cw, ch)})"
            )
    finally:
        if app is not None:
            app.release()
        glfw.terminate()
        if saved is not None:
            _POINTER.parent.mkdir(parents=True, exist_ok=True)
            _POINTER.write_text(saved)


def main() -> int:
    configure_logging()
    try:
        _check_registry_builds()
        _check_render_to_file()
        _check_precheck_logic()
    except AssertionError as e:
        logger.error(f"copilot_render_check: FAIL — {e}")
        return 1
    except Exception as e:
        logger.exception(f"copilot_render_check: ERROR — {e}")
        return 1
    logger.info("copilot_render_check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
