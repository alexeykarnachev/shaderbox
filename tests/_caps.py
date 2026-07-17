"""A fully-populated CopilotCapabilities fake for tests (feature 020).

CopilotCapabilities is a Protocol (production passes CopilotBackend); the fake is a frozen
dataclass of Callable fields — positional-only protocol methods are satisfied by Callable
instance attributes, so it conforms structurally. The factory supplies no-op defaults for
ALL fields and lets a test override only the ones it drives — so adding a new capability
field doesn't break every test helper.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from shaderbox.copilot.capabilities import (
    CompileErrorInfo,
    CopilotCapabilities,
    DeleteNodeResult,
    EditResult,
    ExampleEntry,
    GrepHit,
    LibCatalogEntry,
    LibFileResult,
    LibFunctionBody,
    MediaBindResult,
    NodeImportResult,
    NodeOpResult,
    NodeTreeEntry,
    PublishResult,
    RenderResult,
    ScriptView,
    ScriptWriteResult,
    SetUniformResult,
    ShaderView,
    SwitchNodeResult,
    TelegramConnectResult,
    TelegramOpResult,
    TelegramPackInfo,
    WorkingSetView,
)
from shaderbox.render_shape import RenderShape


@dataclass(frozen=True)
class _FakeCaps:
    node_tree: Callable[[], list[NodeTreeEntry]]
    lib_catalog: Callable[[], list[LibCatalogEntry]]
    example_catalog: Callable[[], list[ExampleEntry]]
    read_shaders: Callable[[list[str]], list[ShaderView]]
    grep: Callable[[str], list[GrepHit]]
    read_lib: Callable[[list[str]], list[LibFunctionBody]]
    read_working_set: Callable[[], list[WorkingSetView]]
    batch_begin: Callable[[], None]
    apply_shader_edit: Callable[[str, str, bool, str], EditResult]
    apply_full_rewrite: Callable[[str, str], EditResult]
    set_uniform: Callable[[str, object, str], SetUniformResult]
    read_script: Callable[[str], ScriptView]
    write_script: Callable[[str, str], ScriptWriteResult]
    apply_script_edit: Callable[[str, str, bool, str], ScriptWriteResult]
    create_node: Callable[
        [str, str, str, bool], tuple[str, list[CompileErrorInfo], str]
    ]
    delete_node: Callable[[str], DeleteNodeResult]
    switch_node: Callable[[str], SwitchNodeResult]
    rename_node: Callable[[str, str], NodeOpResult]
    set_canvas_size: Callable[[str, int, int], NodeOpResult]
    duplicate_node: Callable[[str, str, bool], tuple[str, list[CompileErrorInfo], str]]
    delete_lib_file: Callable[[str], LibFileResult]
    bind_media: Callable[[str, str], MediaBindResult]
    unbind_media: Callable[[str, str], MediaBindResult]
    import_node: Callable[[bool], NodeImportResult]
    render_image: Callable[[str, RenderShape], RenderResult]
    render_video: Callable[[str, float, int, RenderShape], RenderResult]
    probe_render: Callable[[str, float, str], str]
    publish_telegram: Callable[[str], PublishResult]
    publish_youtube: Callable[[str, str, RenderShape], PublishResult]
    has_current_node: Callable[[], bool]
    telegram_connected: Callable[[], bool]
    youtube_connected: Callable[[], bool]
    telegram_has_default_pack: Callable[[], bool]
    set_telegram_token: Callable[[str], TelegramConnectResult]
    telegram_connect: Callable[[], TelegramConnectResult]
    telegram_token_set: Callable[[], bool]
    list_telegram_packs: Callable[[], list[TelegramPackInfo]]
    select_telegram_pack: Callable[[str], TelegramOpResult]
    create_telegram_pack: Callable[[str], TelegramOpResult]
    delete_telegram_pack: Callable[[str], TelegramOpResult]


def minimal_caps(**overrides: Any) -> CopilotCapabilities:
    defaults: dict[str, Any] = {
        "node_tree": lambda: [],
        "lib_catalog": lambda: [],
        "example_catalog": lambda: [],
        "read_shaders": lambda _ids: [],
        "grep": lambda _q: [],
        "read_lib": lambda _names: [],
        "read_working_set": lambda: [],
        "batch_begin": lambda: None,
        "apply_shader_edit": lambda _o, _n, _r, _t: EditResult(matches=0, errors=[]),
        "apply_full_rewrite": lambda _t, _tg: EditResult(matches=1, errors=[]),
        "set_uniform": lambda _n, _v, _node: SetUniformResult(ok=True),
        "read_script": lambda _node: ScriptView("n0", "node", "", [], is_stub=True),
        "write_script": lambda _t, _node: ScriptWriteResult(ok=True),
        "apply_script_edit": lambda _o, _n, _r, _node: ScriptWriteResult(ok=True),
        "create_node": lambda _n, _s, _t, _sw: ("node-new", [], ""),
        "delete_node": lambda _n: DeleteNodeResult(ok=True),
        "switch_node": lambda _n: SwitchNodeResult(ok=True),
        "rename_node": lambda _n, _name: NodeOpResult(ok=True, name="renamed"),
        "set_canvas_size": lambda _n, _w, _h: NodeOpResult(
            ok=True, width=256, height=256
        ),
        "duplicate_node": lambda _n, _name, _sw: ("node-copy", [], ""),
        "delete_lib_file": lambda _p: LibFileResult(ok=True),
        "bind_media": lambda _n, _u: MediaBindResult(
            ok=True, basename="x.png", width=8, height=8
        ),
        "unbind_media": lambda _n, _u: MediaBindResult(ok=True),
        "import_node": lambda _sw: NodeImportResult(
            ok=True, node_id="node-imp", basename="x.glsl"
        ),
        "render_image": lambda _n, _shape: RenderResult(ok=True),
        "render_video": lambda _n, _s, _f, _shape: RenderResult(ok=True),
        "probe_render": lambda _n, _t, _lf="": "render@t=0.0s: ink 0% (probe)",
        "publish_telegram": lambda _e: PublishResult(ok=True),
        "publish_youtube": lambda _t, _d, _shape: PublishResult(ok=True),
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
    return _FakeCaps(**defaults)
