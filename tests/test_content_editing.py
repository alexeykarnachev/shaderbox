"""Content-addressed editing tools (feature 039).

Two layers, both GL-free:
- the REAL backend methods (apply_full_rewrite / apply_shader_edit / the persist seam),
  driven via the test_edit_messages stub idiom — production code, not test-file mirrors;
- the tool-layer behavior (edit_shader / write_shader) through a fake App via the loop,
  incl. whole-file mode, the removed-names fact, and the widened retry cap.
"""

import threading
import types
from pathlib import Path

from shaderbox.copilot.agent import AgentError, AgentToolCard, run_turn
from shaderbox.copilot.backend import CopilotBackend, _CopilotEditTarget
from shaderbox.copilot.capabilities import CompileErrorInfo, EditResult
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.gate import GateChannel
from shaderbox.copilot.llm.api import LLMDone, LLMStreamEvent, LLMTextDelta
from shaderbox.copilot.tools.registry import build_registry
from shaderbox.copilot.tools.shader import _applied_result
from shaderbox.shader_lib import parser
from tests.test_copilot_loop import _fake_caps, _fake_context, _FakeClient, _tool_call

# ---- parser.top_level_names (the removed-names extraction) ----


def test_top_level_names_functions_and_decls() -> None:
    src = "\n".join(
        [
            "uniform float u_speed = 0.4;",
            "uniform uint u_text[MAX_TEXT_LEN];",
            "const float SB_PI = 3.14159;",
            "float fbm(vec2 p) {",
            "    if (p.x > 0.0) {",
            "        return 1.0;",
            "    } else if (p.y > 0.0) {",
            "        return 2.0;",
            "    }",
            "    return 0.0;",
            "}",
            "void main() {",
            "    fs_color = vec4(fbm(vs_uv));",
            "}",
        ]
    )
    fns, decls = parser.top_level_names(src)
    assert fns == {"fbm", "main"}  # the nested `else if (...) {` is NOT a function
    assert decls == {"u_speed", "u_text", "SB_PI"}


def test_top_level_names_ignores_comments_and_nested_decls() -> None:
    src = "\n".join(
        [
            "/*",
            "uniform float u_ghost;",
            "void commented_out() {",
            "}",
            "*/",
            "void main() {",
            "    const float local = 1.0;",
            "}",
        ]
    )
    fns, decls = parser.top_level_names(src)
    assert fns == {"main"}
    assert decls == set()


def test_top_level_names_multiline_signature_residue() -> None:
    # Documented best-effort residue: a signature whose `{` sits on a later line is missed.
    src = (
        "float quarter_ellipse_dist(vec2 p,\n        vec2 radii)\n{\n    return 0.0;\n}"
    )
    fns, _decls = parser.top_level_names(src)
    assert fns == set()


# ---- the REAL apply_full_rewrite (stub idiom: production resolve/guard/note path) ----


def _rewrite_method(
    source: str, persist_result: EditResult | None = None
) -> tuple[types.SimpleNamespace, "types.FunctionType"]:
    tgt = _CopilotEditTarget(
        kind="node",
        source=source,
        ws_address="n1",
        label="node 'X' (n1)",
    )
    stub = types.SimpleNamespace(
        _copilot_resolve_target=lambda _target, *, allow_create: tgt,
        _batch_mutated=set(),
        _bridge=types.SimpleNamespace(run_on_main=lambda fn: fn()),
        _copilot_persist_target=lambda _tgt, _new, matches: (
            persist_result
            if persist_result is not None
            else EditResult(matches=matches, errors=[])
        ),
    )
    return stub, CopilotBackend.apply_full_rewrite.__get__(stub)


_SRC = "\n".join(
    [
        "uniform float u_glow = 0.4;",
        "uniform uint u_text[MAX_TEXT_LEN];",
        "float helper(float x) {",
        "    return x * 2.0;",
        "}",
        "void main() {",
        "    fs_color = vec4(helper(u_glow));",
        "}",
    ]
)


def test_rewrite_note_lists_removed_functions_and_declarations() -> None:
    _stub, rewrite = _rewrite_method(_SRC)
    res = rewrite("void main() {\n    fs_color = vec4(1.0);\n}", "")
    assert res.rewrite_note == (
        "note: this rewrite removed function(s): helper; "
        "declaration(s): u_glow, u_text"
    )


def test_rewrite_note_silent_when_nothing_removed() -> None:
    _stub, rewrite = _rewrite_method(_SRC)
    res = rewrite(_SRC + "\n// trailing comment", "")
    assert res.rewrite_note == ""


def test_rewrite_note_skipped_on_force_restore() -> None:
    restored = EditResult(matches=1, errors=[], restored_note="EDIT UNDONE")
    _stub, rewrite = _rewrite_method(_SRC, persist_result=restored)
    res = rewrite("void main() { }", "")
    assert res.restored_note == "EDIT UNDONE"
    assert res.rewrite_note == ""


def test_rewrite_note_skipped_on_unresolved_persist() -> None:
    failed = EditResult(
        matches=0, errors=[], unresolved=True, unresolved_reason="failed to write"
    )
    _stub, rewrite = _rewrite_method(_SRC, persist_result=failed)
    res = rewrite("void main() { }", "")
    assert res.unresolved and res.rewrite_note == ""


def test_rewrite_batch_guard_rejects_before_persist() -> None:
    calls: list[str] = []
    stub, rewrite = _rewrite_method(_SRC)
    inner = stub._copilot_persist_target
    stub._copilot_persist_target = lambda tgt, new, m: calls.append(new) or inner(
        tgt, new, m
    )
    stub._batch_mutated.add("n1")
    res = rewrite("void main() { }", "")
    assert res.unresolved
    assert "already edited earlier in this same step" in res.unresolved_reason
    assert res.target_label == "node 'X' (n1)"
    assert calls == []  # the reject never reaches the persist seam


def test_rewrite_note_suppressed_on_unbalanced_new_text() -> None:
    # Brace-broken text hides later definitions from the depth-0 scan — the note must
    # stay silent (the compile error / lib brace warning are the loud channel).
    _stub, rewrite = _rewrite_method(_SRC)
    res = rewrite("void main() {\n    fs_color = vec4(1.0);", "")
    assert res.rewrite_note == ""


def test_rewrite_note_not_fooled_by_restyled_signature() -> None:
    # Live false positive (cost re-measure run): new_text kept main() but in Allman
    # style; the per-line scan misses it — the note must NOT claim it removed.
    allman = "void main()\n{\n    fs_color = vec4(helper(u_glow));\n}"
    new_text = (
        "uniform float u_glow = 0.4;\n"
        "uniform uint u_text[MAX_TEXT_LEN];\n"
        "float helper(float x) {\n    return x * 2.0;\n}\n" + allman
    )
    _stub, rewrite = _rewrite_method(_SRC)
    res = rewrite(new_text, "")
    assert res.rewrite_note == ""


def test_rewrite_applies_with_compile_errors_and_still_notes() -> None:
    # The mutation is real even when it compiles broken — the fact still rides along.
    err = CompileErrorInfo(path="n.frag.glsl", line=1, message="boom")
    _stub, rewrite = _rewrite_method(
        _SRC, persist_result=EditResult(matches=1, errors=[err])
    )
    res = rewrite("void main() {\n}", "")
    assert res.errors and res.rewrite_note.startswith("note: this rewrite removed")


def test_rewrite_lib_create_path_invalidates_consumers() -> None:
    # The lib branch end-to-end on REAL persist: create flag, batch-mutated marking,
    # consumer invalidation — write_copilot_lib_file is the one stubbed write.
    written: dict[str, str] = {}
    invalidated: list[str] = []
    lib_path = types.SimpleNamespace(resolve=lambda: "LIB")
    tgt = _CopilotEditTarget(
        kind="lib",
        lib_path=lib_path,  # type: ignore[arg-type]
        source="",
        lib_create=True,
        ws_address="lib:new.glsl",
        label="lib:new.glsl",
    )
    stub = types.SimpleNamespace(
        _copilot_resolve_target=lambda _target, *, allow_create: tgt,
        _batch_mutated=set(),
        _bridge=types.SimpleNamespace(run_on_main=lambda fn: fn()),
        _capture_lib=lambda _addr, _src, _create: None,
        _get_shader_lib_files=lambda: types.SimpleNamespace(
            write_copilot_lib_file=lambda _p, text: written.update(text=text) or True
        ),
        invalidate_lib_consumers=lambda p: invalidated.append(p),
        _working_set_add=lambda _a: None,
        _oscillation_note=lambda _k, _p, _n: "",
    )
    stub._copilot_persist_target = CopilotBackend._copilot_persist_target.__get__(stub)
    rewrite = CopilotBackend.apply_full_rewrite.__get__(stub)
    res = rewrite("float SB_t(float x) { return x; }", "lib:new.glsl")
    assert "created" in res.lib_note
    assert written["text"] == "float SB_t(float x) { return x; }"
    assert invalidated == [lib_path]
    assert "lib:new.glsl" in stub._batch_mutated


# ---- the REAL persist seam: CRLF normalization ----


def test_persist_normalizes_crlf() -> None:
    captured: dict[str, str] = {}
    node = types.SimpleNamespace(
        compile_unit=types.SimpleNamespace(errors=[]), program=object()
    )
    tgt = _CopilotEditTarget(
        kind="node",
        node_id="n1",
        node=node,  # type: ignore[arg-type]
        source="old",
        ws_address="n1",
        label="node 'X' (n1)",
    )
    stub = types.SimpleNamespace(
        _capture_node=lambda _id: None,
        _copilot_persist_shader=lambda _id, _node, text: captured.update(text=text)
        or [],
        _working_set_add=lambda _a: None,
        _batch_mutated=set(),
        _broken_streak={},
        _last_clean={},
        _render_facts_for=lambda _n: "",
        _oscillation_note=lambda _k, _p, _n: "",
    )
    persist = CopilotBackend._copilot_persist_target.__get__(stub)
    res = persist(tgt, "a\r\nb\r\n}", 1)
    assert captured["text"] == "a\nb\n}"
    assert res.matches == 1


def test_persist_force_restores_after_streak_on_real_path() -> None:
    # The 033 unstick end-to-end on the REAL persist + _force_restore: N consecutive
    # broken edits put the file back at its last clean state and reset the streak.
    writes: list[str] = []
    node = types.SimpleNamespace(
        compile_unit=types.SimpleNamespace(errors=[]),
        program=object(),
        source=types.SimpleNamespace(path=Path("n.frag.glsl")),
    )
    tgt = _CopilotEditTarget(
        kind="node",
        node_id="n1",
        node=node,  # type: ignore[arg-type]
        source="void main() { }",
        ws_address="n1",
        label="node 'X' (n1)",
    )
    err = CompileErrorInfo(path="n.frag.glsl", line=1, message="boom")
    limit = COPILOT_CONFIG.auto_revert_after_failed_edits

    def persist_shader(_id: str, _node: object, text: str) -> list[CompileErrorInfo]:
        writes.append(text)
        # Every model edit compiles broken; the restore write itself is clean. Mirrors
        # production: the write updates the node's compile state (prev_clean reads it).
        errs = [] if text == "void main() { }" else [err]
        node.compile_unit.errors = errs
        return errs

    stub = types.SimpleNamespace(
        _capture_node=lambda _id: None,
        _copilot_persist_shader=persist_shader,
        _working_set_add=lambda _a: None,
        _batch_mutated=set(),
        _broken_streak={},
        _last_clean={},
        _render_facts_for=lambda _n: "",
        _oscillation_note=lambda _k, _p, _n: "",
    )
    stub._force_restore = CopilotBackend._force_restore.__get__(stub)
    persist = CopilotBackend._copilot_persist_target.__get__(stub)

    res = persist(tgt, "broken 1 {", 1)
    assert res.errors and not res.restored_note  # streak 1: a clean file just broke
    for i in range(2, limit):
        res = persist(tgt, f"broken {i} {{", 1)
        assert not res.restored_note
    res = persist(tgt, "broken last {", 1)
    assert "EDIT UNDONE" in res.restored_note
    assert f"{limit} consecutive edits" in res.restored_note
    assert writes[-1] == "void main() { }"  # the restore re-wrote the last clean state
    assert stub._broken_streak["n1"] == 0  # fresh budget after the restore


# ---- the REAL apply_shader_edit: deletion by quotation (the restored 038 contract) ----


def _edit_method(source: str) -> tuple[dict[str, str], "types.FunctionType"]:
    captured: dict[str, str] = {}
    tgt = _CopilotEditTarget(
        kind="node", source=source, ws_address="n1", label="node 'X' (n1)"
    )
    stub = types.SimpleNamespace(
        _copilot_resolve_target=lambda _target, *, allow_create: tgt,
        _bridge=types.SimpleNamespace(run_on_main=lambda fn: fn()),
        _copilot_persist_target=lambda _tgt, new, matches: captured.update(text=new)
        or EditResult(matches=matches, errors=[]),
    )
    return captured, CopilotBackend.apply_shader_edit.__get__(stub)


_DELETE_SRC = "\n".join(
    [
        "float helper(float x) {",
        "    // doubles x",
        "    return x * 2.0;",
        "}",
        "",
        "void main() {",
        "    fs_color = vec4(1.0);",
        "}",
    ]
)


def test_edit_shader_empty_new_str_deletes_quoted_block() -> None:
    captured, edit = _edit_method(_DELETE_SRC)
    old_str = "float helper(float x) {\n    // doubles x\n    return x * 2.0;\n}"
    res = edit(old_str, "", False, "")
    assert res.matches == 1
    assert "helper" not in captured["text"]
    assert "void main()" in captured["text"]


def test_edit_shader_delete_without_comment_refused() -> None:
    # The comment-loss guard: deleting the block while not reproducing its interior
    # comment would silently destroy author content — refused, nothing applied.
    captured, edit = _edit_method(_DELETE_SRC)
    res = edit("float helper(float x) {\n    return x * 2.0;\n}", "", False, "")
    assert res.comment_loss and res.matches == 0
    assert "text" not in captured


# ---- _applied_result: the rewrite-note trailer ----


def test_applied_result_appends_rewrite_note() -> None:
    ok, msg, _ = _applied_result(
        EditResult(
            matches=1,
            errors=[],
            target_label="node 'X' (n1)",
            rewrite_note="note: this rewrite removed function(s): fbm",
        )
    )
    assert ok is True
    assert msg.startswith("ok — compiled clean")
    assert msg.endswith("note: this rewrite removed function(s): fbm")


def test_applied_result_compile_errors_keep_note() -> None:
    err = CompileErrorInfo(path="n.frag.glsl", line=4, message="boom")
    ok, msg, _ = _applied_result(
        EditResult(matches=1, errors=[err], rewrite_note="note: x")
    )
    assert ok is True
    assert "compiled with errors" in msg and msg.endswith("note: x")


# ---- tool-layer behavior through the loop ----


def _run(scripts: list[list[LLMStreamEvent]], text: str = "vec3 p = u_pos;") -> list:
    caps = _fake_caps(edit_errors=[[]] * 20, text=text)
    return list(
        run_turn(
            _FakeClient(scripts),
            build_registry(caps),
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="edit it",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )


def test_write_shader_applies_through_loop() -> None:
    events = _run(
        [
            _tool_call("c1", "write_shader", '{"new_text": "// rewritten whole file"}'),
            [LLMTextDelta("done"), LLMDone("stop")],
        ]
    )
    card = next(e for e in events if isinstance(e, AgentToolCard))
    assert card.name == "write_shader"
    assert card.ok is True


def test_retry_cap_fires_on_spiraling_edit_shader() -> None:
    # An old_str that never matches must stop at max_edit_retries, not run to
    # max_iterations.
    fail = _tool_call(
        "cx",
        "edit_shader",
        '{"old_str": "zzz no such text", "new_str": "x"}',
    )
    scripts: list[list[LLMStreamEvent]] = [fail] * (COPILOT_CONFIG.max_iterations + 5)
    events = _run(scripts)
    failed = [e for e in events if isinstance(e, AgentToolCard) and not e.ok]
    assert len(failed) == COPILOT_CONFIG.max_edit_retries
    assert isinstance(events[-1], AgentError)
    assert len(failed) < COPILOT_CONFIG.max_iterations


def test_lib_write_warns_on_brace_imbalance() -> None:
    # A lib file has no standalone compile to scream — the persist seam appends a
    # deterministic brace warning instead.
    tgt = _CopilotEditTarget(
        kind="lib",
        lib_path=types.SimpleNamespace(resolve=lambda: "LIB"),  # type: ignore[arg-type]
        source="float SB_a(float x) {\n    return x;\n}\nfloat SB_b(float x) {\n    return x;\n}",
        ws_address="lib:t.glsl",
        label="lib:t.glsl",
    )
    stub = types.SimpleNamespace(
        _capture_lib=lambda _addr, _src, _create: None,
        _get_shader_lib_files=lambda: types.SimpleNamespace(
            write_copilot_lib_file=lambda _p, _t: True
        ),
        invalidate_lib_consumers=lambda _p: None,
        _working_set_add=lambda _a: None,
        _batch_mutated=set(),
        _oscillation_note=lambda _k, _p, _n: "",
    )
    persist = CopilotBackend._copilot_persist_target.__get__(stub)
    res = persist(tgt, "float SB_a(float x) {\n    return x;", 1)
    assert "warning: the written file has 1 '{' vs 0 '}'" in res.lib_note
    res = persist(tgt, "float SB_a(float x) {\n    return x;\n}", 1)
    assert "warning" not in res.lib_note


def test_persist_normalizes_lone_cr() -> None:
    captured: dict[str, str] = {}
    node = types.SimpleNamespace(
        compile_unit=types.SimpleNamespace(errors=[]), program=object()
    )
    tgt = _CopilotEditTarget(
        kind="node",
        node_id="n1",
        node=node,  # type: ignore[arg-type]
        source="old",
        ws_address="n1",
        label="node 'X' (n1)",
    )
    stub = types.SimpleNamespace(
        _capture_node=lambda _id: None,
        _copilot_persist_shader=lambda _id, _node, text: captured.update(text=text)
        or [],
        _working_set_add=lambda _a: None,
        _batch_mutated=set(),
        _broken_streak={},
        _last_clean={},
        _render_facts_for=lambda _n: "",
        _oscillation_note=lambda _k, _p, _n: "",
    )
    persist = CopilotBackend._copilot_persist_target.__get__(stub)
    persist(tgt, "a\rb\r\nc", 1)
    assert captured["text"] == "a\nb\nc"


def test_write_shader_arg_order_pinned_through_loop() -> None:
    # The handler maps args["new_text"]/args["target"] positionally into the
    # capability — a swap would pass pyright (both str); the round-trip pins it.
    caps = _fake_caps(edit_errors=[[]] * 3)
    events = list(
        run_turn(
            _FakeClient(
                [
                    _tool_call(
                        "c1", "write_shader", '{"new_text": "// sentinel-039"}'
                    ),
                    [LLMTextDelta("done"), LLMDone("stop")],
                ]
            ),
            build_registry(caps),
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text="rewrite",
            gate=GateChannel(),
            cancel=threading.Event(),
        )
    )
    card = next(e for e in events if isinstance(e, AgentToolCard))
    assert card.ok is True
    listing = caps.read_shaders([])[0].listing
    assert "// sentinel-039" in listing
