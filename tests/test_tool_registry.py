"""Invariant: every tool is fully carded at its single definition site (feature 029).

One test kills three drift classes: a missing label (the old _TOOL_VERBS gap), a missing
gate prompt on an always-gated tool (the old _GATE_PROMPTS gap), and the dogfood analyzer's
coverage list going stale on a tool add/remove. Plus the structural gate/credential
invariants and the publish precheck handoffs (all counts derived from the registry, never
literals).
"""

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import shaderbox
from scripts.dogfood.analyze import (
    _UNREACHABLE_IN_HARNESS,
    CANONICAL_TOOLS,
    REACHABLE_TOOLS,
)
from shaderbox.copilot.backend import CopilotBackend
from shaderbox.copilot.capabilities import NodeTreeEntry
from shaderbox.copilot.gate import GateKind
from shaderbox.copilot.tools.base import GatePolicy, ToolDefinition
from shaderbox.copilot.tools.registry import ToolRegistry, build_registry
from tests._caps import minimal_caps


def test_every_tool_fully_carded() -> None:
    registry: ToolRegistry = build_registry(minimal_caps())
    definitions: list[ToolDefinition] = registry.definitions()
    assert {d.name for d in definitions} == set(CANONICAL_TOOLS)
    for d in definitions:
        assert d.label_live and d.label_live != d.name
        assert d.label_done and d.label_done != d.name
        if d.gate_policy is GatePolicy.ALWAYS:
            assert d.gate_prompt is not None
        # The LLM-facing contract: hallucinated arg keys must be rejected, not swallowed.
        # Checks the emitted schema, so a stray non-ToolArgs args model can't sneak past.
        schema = d.args_model.model_json_schema()
        assert schema.get("additionalProperties") is False, d.name


def test_dead_tool_names_absent_from_copilot_sources() -> None:
    # 039 removed replace_lines/insert_after. A surviving mention anywhere in the
    # package — a tool description, a prompt bullet, a guard/nudge/hint string, even a
    # comment — silently steers the model into a nonexistent tool; this pins the
    # rewording forever. (scripts/ is excluded: the dogfood analyzer's HISTORICAL_TOOLS
    # vocabulary keeps the names to parse old transcripts.)
    root = Path(shaderbox.__file__).parent
    offenders = [
        str(p.relative_to(root))
        for p in sorted(root.rglob("*.py"))
        if "replace_lines" in p.read_text(encoding="utf-8")
        or "insert_after" in p.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_gate_and_credential_structural_invariants() -> None:
    registry = build_registry(minimal_caps())
    definitions = registry.definitions()

    # eager_specs() is exactly the eager-flagged definitions, and every spec emits a real
    # JSON schema (a pydantic Field/constraint typo would raise or yield an empty dict).
    eager = registry.eager_specs()
    assert {s.name for s in eager} == {d.name for d in definitions if d.eager}
    for spec in eager:
        assert isinstance(spec.parameters, dict) and spec.parameters, spec.name

    # Destructive/external tools are ALWAYS-gated.
    always = {d.name for d in definitions if d.gate_policy is GatePolicy.ALWAYS}
    assert {
        "render_image",
        "render_video",
        "publish_telegram",
        "publish_youtube",
        "delete_node",
    } <= always

    # set_telegram_token is THE credential-gated tool (derived, not a count literal).
    credential = [d for d in definitions if d.gate_kind is GateKind.CREDENTIAL]
    assert [d.name for d in credential] == ["set_telegram_token"]
    assert credential[0].secret_field == "telegram_bot_token"


def test_publish_prechecks_hand_off_until_ready() -> None:
    # Ready: prechecks return None (render tools have no precheck at all).
    ready = build_registry(
        minimal_caps(
            telegram_connected=lambda: True,
            telegram_has_default_pack=lambda: True,
            youtube_connected=lambda: True,
        )
    )
    assert ready.precheck("publish_telegram", {}) is None
    assert ready.precheck("publish_youtube", {}) is None
    assert ready.precheck("render_image", {}) is None

    # Not connected -> a guided handoff (no gate fires for this call).
    no_tg = build_registry(minimal_caps())
    msg = no_tg.precheck("publish_telegram", {})
    assert msg is not None and "connect" in msg.lower()

    # Connected but no pack -> a different handoff.
    no_pack = build_registry(minimal_caps(telegram_connected=lambda: True))
    msg = no_pack.precheck("publish_telegram", {})
    assert msg is not None and "pack" in msg.lower()

    no_yt = build_registry(minimal_caps())
    msg = no_yt.precheck("publish_youtube", {})
    assert msg is not None and "connect" in msg.lower()


def test_delete_gate_prompt_shows_node_name() -> None:
    # The confirm card asks with the node's NAME (resolved via the project map with
    # read_shader's prefix rule), short — no trash/recover tail (034 F01).
    registry = build_registry(
        minimal_caps(
            node_tree=lambda: [
                NodeTreeEntry(
                    node_id="a1b2", name="Blank", has_errors=False, is_current=True
                )
            ]
        )
    )
    d = registry.definition_for("delete_node")
    assert d is not None and d.gate_prompt is not None
    assert d.gate_prompt({"node": "a1b2"}) == "Delete node `Blank`?"
    # Full uuid matching the short id resolves too; an unknown id falls back raw.
    assert d.gate_prompt({"node": "a1b2c3d4-ffff"}) == "Delete node `Blank`?"
    assert d.gate_prompt({"node": "zzzz"}) == "Delete node `zzzz`?"
    assert d.gate_prompt({}) == "Delete node `?`?"


def test_delete_precheck_fails_fast_on_empty_or_unknown_target() -> None:
    # The precheck short-circuits BEFORE the always-gate so the user never confirms a
    # "Delete node `?`" that then errors in the handler.
    registry = build_registry(
        minimal_caps(
            node_tree=lambda: [
                NodeTreeEntry(
                    node_id="a1b2", name="Blank", has_errors=False, is_current=True
                )
            ]
        )
    )
    empty = registry.precheck("delete_node", {"node": ""})
    assert empty is not None and "empty" in empty
    unknown = registry.precheck("delete_node", {"node": "zzzz"})
    assert unknown is not None and "zzzz" in unknown
    # A resolvable target passes the precheck (None) so the gate proceeds.
    assert registry.precheck("delete_node", {"node": "a1b2"}) is None


# ---- lazy tool catalogue (feature 052 slice 0) ----


def test_lazy_tools_demoted_but_reachable() -> None:
    reg = build_registry(minimal_caps())
    eager = {s.name for s in reg.eager_specs()}
    assert "load_tools" in eager  # the meta-tool is always present
    for lazy in ("bind_media", "rename_node", "delete_lib_file", "set_telegram_token"):
        assert lazy not in eager  # demoted off the eager core
        assert reg.is_lazy(lazy)
    assert not reg.is_lazy("edit_shader")  # an eager tool is not lazy
    assert not reg.is_lazy("load_tools")  # the meta-tool isn't self-loadable
    assert "bind_media" in {s.name for s in reg.assemble_specs({"bind_media"})}


def test_assemble_specs_is_sorted_and_load_order_independent() -> None:
    reg = build_registry(minimal_caps())
    a = [s.name for s in reg.assemble_specs({"bind_media", "rename_node"})]
    b = [s.name for s in reg.assemble_specs({"rename_node", "bind_media"})]
    assert a == b == sorted(a)  # byte-stable tools= regardless of load order


def test_load_tools_catalog_lists_lazy_only() -> None:
    reg = build_registry(minimal_caps())
    d = reg.definition_for("load_tools")
    assert d is not None
    assert "bind_media:" in d.description and "rename_node:" in d.description
    assert "read_shader:" not in d.description  # eager tools are not in the catalogue


def test_dogfood_coverage_denominator_holds_every_tool_but_the_named_exclusions() -> (
    None
):
    # REACHABLE_TOOLS is the dogfood coverage DENOMINATOR. It must be derived from the registry
    # minus an explicitly-named exclusion set, never hand-listed: a hand-listed subset silently
    # shrinks the domain, so a newly added tool is neither used nor reported as a gap and the
    # metric reads green forever. Falsifier: drop a live tool from the denominator (or add one to
    # the exclusion set without a reason) and this goes red.
    registry: ToolRegistry = build_registry(minimal_caps())
    live: set[str] = {d.name for d in registry.definitions()}

    assert (
        live >= _UNREACHABLE_IN_HARNESS
    )  # no exclusion for a tool that no longer exists
    assert set(REACHABLE_TOOLS) == live - _UNREACHABLE_IN_HARNESS
    # Only the exporter-credential set may be excluded — it precheck-fails on the harness's
    # empty ExporterRegistry. Anything else must earn its exclusion here, visibly.
    for name in _UNREACHABLE_IN_HARNESS:
        assert "telegram" in name or "youtube" in name or name.startswith("publish_")


def test_delete_gate_and_backend_resolvers_accept_the_same_handles() -> None:
    # The gate NAMES the node; the backend re-resolves the raw handle and DELETES it. The two
    # ran different ambiguity rules (first-prefix-hit vs unique-prefix), so an ambiguous handle
    # opened a confirm dialog naming one node and then errored. Pin the two predicates to each
    # other rather than to examples — the defect class is "two checks that must agree, don't".
    ids = [
        "ab12cd00-1111-4111-8111-111111111111",
        "ab34ef00-2222-4222-8222-222222222222",
    ]
    names = {ids[0]: "Keeper", ids[1]: "Doomed"}
    holder = SimpleNamespace(_get_ui_nodes=lambda: dict.fromkeys(ids))
    short_ids = CopilotBackend._copilot_short_ids(holder)
    resolve_strict = partial(CopilotBackend._copilot_resolve_node_id, holder)

    registry = build_registry(
        minimal_caps(
            node_tree=lambda: [
                NodeTreeEntry(
                    node_id=short_ids[nid],
                    name=names[nid],
                    has_errors=False,
                    is_current=False,
                )
                for nid in ids
            ]
        )
    )
    definition = registry.definition_for("delete_node")
    assert definition is not None and definition.gate_prompt is not None

    handles = ["", "a", "ab", "ab1", "ab12", "ab12cd", ids[0], ids[1], "zz"]
    for handle in handles:
        strict = resolve_strict(handle)
        prompt = definition.gate_prompt({"node": handle})
        precheck = registry.precheck("delete_node", {"node": handle})
        if strict is None:
            # The backend would refuse this handle, so no confirm dialog may name a node:
            # the precheck has to fail fast first.
            assert precheck is not None, handle
            assert names[ids[0]] not in prompt, handle
            assert names[ids[1]] not in prompt, handle
        else:
            assert precheck is None, handle
            assert names[strict] in prompt, handle
