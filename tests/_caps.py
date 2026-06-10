"""A fully-populated CopilotCapabilities for tests (feature 020).

The agent loop only exercises a handful of capabilities per test; the rest must still be provided
(the dataclass has no defaults). This factory supplies no-op defaults for ALL fields and lets a test
override only the ones it drives — so adding a new capability field doesn't break every test helper.
"""

from typing import Any

from shaderbox.copilot.capabilities import (
    CopilotCapabilities,
    DeleteNodeResult,
    PublishResult,
    RenderResult,
    SetUniformResult,
    SwitchNodeResult,
    TelegramConnectResult,
    TelegramOpResult,
)


def minimal_caps(**overrides: Any) -> CopilotCapabilities:
    defaults: dict[str, Any] = {
        "node_tree": lambda: [],
        "lib_catalog": lambda: [],
        "template_catalog": lambda: [],
        "read_shaders": lambda _ids: [],
        "grep": lambda _q: [],
        "read_lib": lambda _names: [],
        "read_working_set": lambda: [],
        "batch_begin": lambda: None,
        "apply_shader_edit": lambda _o, _n, _r, _t: None,
        "apply_line_edit": lambda _s, _e, _t, _tg: None,
        "set_uniform": lambda _n, _v, _node: SetUniformResult(ok=True),
        "create_node": lambda _n, _s, _t, _sw: ("node-new", []),
        "delete_node": lambda _n: DeleteNodeResult(ok=True),
        "switch_node": lambda _n: SwitchNodeResult(ok=True),
        "render_image": lambda _n, _w, _h: RenderResult(ok=True),
        "render_video": lambda _n, _s, _f, _w, _h: RenderResult(ok=True),
        "publish_telegram": lambda _e: PublishResult(ok=True),
        "publish_youtube": lambda _t, _d, _s: PublishResult(ok=True),
        "has_current_node": lambda: True,
        "telegram_connected": lambda: False,
        "youtube_connected": lambda: False,
        "telegram_has_default_pack": lambda: False,
        "set_telegram_token": lambda _s: TelegramConnectResult(ok=True),
        "telegram_connect": lambda: TelegramConnectResult(ok=True),
        "telegram_token_set": lambda: False,
        "list_telegram_packs": lambda: [],
        "select_telegram_pack": lambda _s: TelegramOpResult(ok=True),
        "create_telegram_pack": lambda _t: TelegramOpResult(ok=True),
        "delete_telegram_pack": lambda _s: TelegramOpResult(ok=True),
    }
    defaults.update(overrides)
    return CopilotCapabilities(**defaults)
