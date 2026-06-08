"""Headless verification for the copilot render/publish (020·18) + Telegram (020·19) tools.

Covers the pure-logic decisions the GUI smoke can't reach (`scripts/smoke.py` never builds
the registry or runs a turn):

  H1. The registry builds: 20 eager specs (10 shader + 4 render/publish + 6 telegram), each new
      tool's arg schema builds, render/publish are GatePolicy.ALWAYS, set_telegram_token is
      gate_kind=CREDENTIAL.
  H2. Render-to-file (needs a live GL context, like smoke): the App render closure produces a
      real file at the REQUESTED size (the FIXED_DIMS/RENDER_AT_TARGET path), not the canvas
      size — the regression that an inert preset would silently render at canvas size.
  H3. Precheck logic: each publish tool's precheck returns a guided handoff when not connected
      / no pack, and None when ready.
  H4. CREDENTIAL redaction (security): the pasted token never appears in the handler msg, the
      resolved card text, the payload, or a persisted conversation round-trip — only the mask.

Usage: `uv run python scripts/copilot_render_check.py` (exit 0 on success, non-zero on failure).
H1/H3/H4 are GL-free + network-free; H2 opens an invisible glfw window (like scripts/smoke.py).
"""

import sys
import tempfile
from pathlib import Path

import glfw
from loguru import logger
from platformdirs import user_data_dir

from shaderbox.app import App
from shaderbox.constants import VIDEO_RESOLUTION_ALIGNMENT
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
    SwitchNodeResult,
    TelegramConnectResult,
    TelegramOpResult,
)
from shaderbox.copilot.gate import GateKind
from shaderbox.copilot.llm.openrouter import OpenRouterLLMClient
from shaderbox.copilot.persistence import ConversationStore
from shaderbox.copilot.session import CopilotSession
from shaderbox.copilot.state import ChatState, Message
from shaderbox.copilot.tools.base import mask_secret
from shaderbox.copilot.tools.registry import build_registry
from shaderbox.logging_setup import configure_logging
from shaderbox.tabs import share_state

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
        switch_node=lambda _n: SwitchNodeResult(ok=False),
        render_image=_render_image,
        render_video=_render_video,
        publish_telegram=_pub_tg,
        publish_youtube=_pub_yt,
        has_current_node=lambda: True,
        telegram_connected=lambda: telegram_connected,
        youtube_connected=lambda: youtube_connected,
        set_telegram_token=lambda _s: TelegramConnectResult(ok=False),
        telegram_connect=lambda: TelegramConnectResult(ok=False),
        telegram_token_set=lambda: False,
        list_telegram_packs=lambda: [],
        select_telegram_pack=lambda _n: TelegramOpResult(ok=False),
        create_telegram_pack=lambda _t: TelegramOpResult(ok=False),
        delete_telegram_pack=lambda _n: TelegramOpResult(ok=False),
        telegram_has_default_pack=lambda: telegram_pack,
    )


def _check_registry_builds() -> None:
    caps = _stub_caps(
        telegram_connected=True, telegram_pack=True, youtube_connected=True
    )
    registry = build_registry(caps)
    eager = registry.eager_specs()
    assert (
        len(eager) == 20
    ), f"expected 20 eager tools (10 shader + 4 render/publish + 6 telegram), got {len(eager)}"

    names = {s.name for s in eager}
    assert (
        names >= _RENDER_PUBLISH_TOOLS
    ), f"render/publish tools missing from eager set: {_RENDER_PUBLISH_TOOLS - names}"

    # Each new tool's arg schema builds (a pydantic Field/constraint typo would raise here).
    for spec in eager:
        if spec.name in _RENDER_PUBLISH_TOOLS:
            assert (
                isinstance(spec.parameters, dict) and spec.parameters
            ), f"{spec.name} produced no JSON schema"

    for name in _RENDER_PUBLISH_TOOLS:
        assert registry.requires_gate_always(name), f"{name} must be GatePolicy.ALWAYS"
    # set_telegram_token is the one CREDENTIAL-gated tool (feature 020·19).
    set_token = registry.definition_for("set_telegram_token")
    assert (
        set_token is not None and set_token.gate_kind is GateKind.CREDENTIAL
    ), "set_telegram_token must be gate_kind=CREDENTIAL"
    assert set_token.secret_field == "telegram_bot_token", "secret_field marker missing"
    logger.info(
        "H1 ok: 20 eager tools build; render/publish ALWAYS-gated; set_telegram_token CREDENTIAL"
    )


def _check_precheck_logic() -> None:
    # Ready: prechecks return None.
    ready = build_registry(
        _stub_caps(telegram_connected=True, telegram_pack=True, youtube_connected=True)
    )
    assert (
        ready.precheck("publish_telegram", {}) is None
    ), "telegram precheck should pass when connected + pack set"
    assert (
        ready.precheck("publish_youtube", {}) is None
    ), "youtube precheck should pass when connected"
    # Render tools have no precheck.
    assert ready.precheck("render_image", {}) is None, "render_image has no precheck"

    # Telegram not connected -> a guided handoff (no gate fires for this call).
    no_tg = build_registry(
        _stub_caps(
            telegram_connected=False, telegram_pack=False, youtube_connected=True
        )
    )
    msg = no_tg.precheck("publish_telegram", {})
    assert (
        msg is not None and "connect" in msg.lower()
    ), f"unconnected telegram precheck should hand off: {msg!r}"

    # Telegram connected but no pack -> a different handoff.
    no_pack = build_registry(
        _stub_caps(telegram_connected=True, telegram_pack=False, youtube_connected=True)
    )
    msg = no_pack.precheck("publish_telegram", {})
    assert (
        msg is not None and "pack" in msg.lower()
    ), f"no-pack telegram precheck should hand off: {msg!r}"

    # YouTube not connected -> a handoff.
    no_yt = build_registry(
        _stub_caps(telegram_connected=True, telegram_pack=True, youtube_connected=False)
    )
    msg = no_yt.precheck("publish_youtube", {})
    assert (
        msg is not None and "connect" in msg.lower()
    ), f"unconnected youtube precheck should hand off: {msg!r}"
    logger.info("H3 ok: publish prechecks hand off when unready, pass when ready")


_FAKE_TOKEN = "1234567890:AAfake_secret_tail_must_never_appear_anywhere"


def _check_credential_redaction() -> None:
    # H4 (security-critical, feature 020·19): the full token must NEVER appear in the handler's
    # LLM-facing msg, the resolved-card text, OR a persisted conversation round-trip — only the
    # masked prefix. mask_secret must hide the tail.
    masked = mask_secret(_FAKE_TOKEN)
    assert (
        _FAKE_TOKEN not in masked and "fake_secret_tail" not in masked
    ), f"mask_secret leaked the tail: {masked!r}"

    # The set_telegram_token handler (the credential shape) returns a redacted msg. Drive it with
    # a stub caps that ECHOES the bot_username only (no token), and assert no leak in the msg.
    caps = _stub_caps(
        telegram_connected=False, telegram_pack=False, youtube_connected=False
    )
    registry = build_registry(caps)
    ok, msg, payload = registry.execute("set_telegram_token", {}, _FAKE_TOKEN)
    _ = ok
    assert (
        _FAKE_TOKEN not in msg and "fake_secret_tail" not in msg
    ), f"set_telegram_token handler leaked the token into its msg: {msg!r}"
    assert payload is None or _FAKE_TOKEN not in str(
        payload
    ), "set_telegram_token leaked the token into its payload"

    # The resolved-card echo (answer_gate_credential path) + a persistence round-trip must carry
    # only the mask. Build a session, materialize a CREDENTIAL card, answer it with the token.
    client = OpenRouterLLMClient(get_api_key=lambda: "", get_model=lambda: "stub")
    session = CopilotSession(caps=caps, client=client, get_project_slug=lambda: "test")
    session.state = ChatState(
        messages=[
            Message(role="user", text="connect telegram"),
            Message(
                role="pending_action",
                text="Paste your Telegram bot token below.",
                gate_kind=GateKind.CREDENTIAL,
            ),
        ]
    )
    session.answer_gate_credential(
        _FAKE_TOKEN
    )  # resolves the open card with a redacted echo
    card = session.state.messages[1]
    assert (
        card.resolved and _FAKE_TOKEN not in card.text
    ), f"resolved credential card leaked the token: {card.text!r}"
    assert card.gate_input == "", "the typed-secret buffer was not cleared on answer"
    # A persistence round-trip never resurrects the token.
    store = ConversationStore.from_runtime(session.state, [])
    blob = store.model_dump_json()
    assert (
        _FAKE_TOKEN not in blob and "fake_secret_tail" not in blob
    ), "the token leaked into the persisted conversation"
    # gate_kind survives the round-trip (credential card stays a credential card).
    reloaded = ConversationStore.model_validate_json(blob).to_messages()
    assert reloaded[1].gate_kind is GateKind.CREDENTIAL, "gate_kind lost on round-trip"
    logger.info(
        "H4 ok: the token never leaks to msg / card / payload / persisted conversation"
    )


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
    app: App | None = None
    try:
        app = App(project_dir=_DEV_PROJECT, headless=True)
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
        _check_credential_redaction()
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
