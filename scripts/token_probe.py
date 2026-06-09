"""Empirical probe of OpenRouter input-token accounting for the copilot's tool catalogue.

Answers, by MEASUREMENT (not reasoning), the questions raised about copilot token cost:

  Q1. WHAT does an input token get spent on? Empty request baseline, then the marginal
      cost of: the native `tools=` block as a whole, one tool, and the split between a
      tool's name / description / JSON-schema. (delta of usage.prompt_tokens per request)
  Q2. Are the tool specs re-sent (and re-billed) on EVERY step of a multi-iteration turn,
      as suspected? Simulate a 2-step turn (assistant tool_call -> tool result -> continue)
      and compare prompt_tokens with vs without tools on step 2.
  Q3. Does OpenRouter prompt-CACHING change the bill? Fire the SAME large request twice and
      read usage.prompt_tokens_details.cached_tokens (grok via OpenRouter).
  Q4. The proposed design: a COMPACT plaintext tool menu in the prompt (name + one-line
      desc) vs the full native `tools=` block — how many tokens each, and does a 2-stage
      "agent picks tools -> we attach only those natively" flow actually work end to end.

Run: OPENROUTER_API_KEY=… uv run python scripts/token_probe.py
Key from OPENROUTER_API_KEY (the shell env); model from OPENROUTER_MODEL or the in-tree
CopilotIntegration default.

This is a THROWAWAY experiment script (not wired into the app, not under make check). It
exists to inform the lazy-tool-catalogue design; delete once the design lands.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import MagicMock

from openai import OpenAI

from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.llm.api import LLMToolSpec
from shaderbox.copilot.prompt import _SYSTEM_PROMPT
from shaderbox.copilot.tools.registry import build_registry
from shaderbox.exporters.integrations import CopilotIntegration

_BASE_URL = "https://openrouter.ai/api/v1"


def _load_key_and_model() -> tuple[str, str]:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY")
    model = os.environ.get("OPENROUTER_MODEL", "") or CopilotIntegration().model
    return key, model


def _real_registry() -> Any:
    # The 21 real tool specs come straight from build_registry. Tool specs (name /
    # description / args_model schema) do NOT depend on capability behavior, only on the
    # static ToolDefinition fields -> a MagicMock satisfies every callable field.
    caps: CopilotCapabilities = MagicMock(spec=CopilotCapabilities)
    return build_registry(caps)


def _tool_to_wire(spec: LLMToolSpec) -> dict[str, Any]:
    # Mirror of openrouter.py::_tool_to_wire so the wire shape is identical to production.
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


@dataclass
class Probe:
    client: OpenAI
    model: str

    def prompt_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[int, int]:
        # Returns (prompt_tokens, cached_tokens). max_completion_tokens=1 keeps the OUTPUT
        # ~free; we only care about the INPUT accounting. usage.prompt_tokens is the billed
        # input count; cached_tokens (when present) is the prompt-cache-hit slice of it.
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": 1,
            "extra_body": {"reasoning": {"effort": "minimal"}, "usage": {"include": True}},
        }
        if tools:
            kwargs["tools"] = tools
        resp = self.client.chat.completions.create(**kwargs)
        usage = resp.usage
        prompt = usage.prompt_tokens if usage else -1
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        return prompt, cached


def _line(label: str, value: Any) -> None:
    print(f"  {label:<52} {value}")


def _hdr(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


# A tiny fixed user turn reused across probes so only the variable-under-test moves.
_USER = {"role": "user", "content": "make the shader red"}
_SYS = {"role": "system", "content": _SYSTEM_PROMPT}


def q1_anatomy(probe: Probe, registry: Any) -> None:
    _hdr("Q1 -- WHAT a prompt-token is spent on (marginal cost by delta)")
    specs: list[LLMToolSpec] = registry.eager_specs()
    wire_all = [_tool_to_wire(s) for s in specs]

    # Baseline: a minimal request, no system prompt, no tools.
    base_msgs = [_USER]
    base, _ = probe.prompt_tokens(base_msgs)
    _line("baseline (1 short user msg, no sys, no tools)", base)

    # System prompt alone.
    sys_only, _ = probe.prompt_tokens([_SYS, _USER])
    _line("+ full system prompt (prose)", f"{sys_only}  (+{sys_only - base})")

    # Full native tools block on top of the system prompt.
    with_tools, _ = probe.prompt_tokens([_SYS, _USER], tools=wire_all)
    _line(
        f"+ native tools=[{len(wire_all)} tools]",
        f"{with_tools}  (+{with_tools - sys_only} for the tools block)",
    )

    # Per-tool marginal: tools block with N vs N-1 (drop the largest tool).
    # Cheaper: measure 1 tool vs 0, and all vs all-but-one, to bracket per-tool cost.
    one, _ = probe.prompt_tokens([_USER], tools=wire_all[:1])
    _line("native tools=[1 tool] (no sys)", f"{one}  (+{one - base} for 1 tool incl. block overhead)")

    half = wire_all[: len(wire_all) // 2]
    half_tok, _ = probe.prompt_tokens([_USER], tools=half)
    full_tok, _ = probe.prompt_tokens([_USER], tools=wire_all)
    _line(f"native tools=[{len(half)} tools] (no sys)", half_tok)
    _line(f"native tools=[{len(wire_all)} tools] (no sys)", full_tok)
    if len(wire_all) - len(half):
        per = (full_tok - half_tok) / (len(wire_all) - len(half))
        _line("=> avg per-tool marginal (name+desc+schema)", f"{per:.1f} tok/tool")

    # Split one tool into name-only / +desc / +schema to see where the tool's tokens live.
    sample = specs[0]
    empty_schema: dict[str, Any] = {"type": "object", "properties": {}}
    name_only = [{"type": "function", "function": {"name": sample.name, "description": "", "parameters": empty_schema}}]
    name_desc = [{"type": "function", "function": {"name": sample.name, "description": sample.description, "parameters": empty_schema}}]
    name_full = [_tool_to_wire(sample)]
    n0, _ = probe.prompt_tokens([_USER], tools=name_only)
    n1, _ = probe.prompt_tokens([_USER], tools=name_desc)
    n2, _ = probe.prompt_tokens([_USER], tools=name_full)
    print(f"\n  anatomy of ONE tool ('{sample.name}'):")
    _line("name only (empty desc, empty schema)", f"{n0}  (+{n0 - base} over baseline = block+name)")
    _line("+ description", f"{n1}  (+{n1 - n0} for the description)")
    _line("+ full JSON schema (parameters)", f"{n2}  (+{n2 - n1} for the schema)")
    desc_chars = len(sample.description)
    schema_chars = len(json.dumps(sample.parameters))
    _line("   (description chars / json-schema chars)", f"{desc_chars} / {schema_chars}")


def q2_resent_every_step(probe: Probe, registry: Any) -> None:
    _hdr("Q2 -- are tool specs re-sent + re-billed on EVERY step of a turn?")
    specs: list[LLMToolSpec] = registry.eager_specs()
    wire_all = [_tool_to_wire(s) for s in specs]

    # Step 1: the opening request of a turn.
    step1_msgs = [_SYS, _USER]
    s1_notools, _ = probe.prompt_tokens(step1_msgs)
    s1_tools, _ = probe.prompt_tokens(step1_msgs, tools=wire_all)
    _line("step 1 prompt_tokens, NO tools", s1_notools)
    _line("step 1 prompt_tokens, WITH tools", f"{s1_tools}  (+{s1_tools - s1_notools})")

    # Step 2: assistant made a tool_call, we appended its result, and continue the SAME turn.
    # The native tools= block is sent AGAIN (production does: agent.py re-passes specs each
    # iteration). If the block is re-billed, step-2-with-tools should exceed step-2-no-tools
    # by ~the same delta as step 1.
    assistant_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": specs[0].name, "arguments": "{}"},
            }
        ],
    }
    tool_result = {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "ok -- read_shader returned; source is in the working set",
    }
    step2_msgs = [_SYS, _USER, assistant_call, tool_result]
    s2_notools, _ = probe.prompt_tokens(step2_msgs)
    s2_tools, _ = probe.prompt_tokens(step2_msgs, tools=wire_all)
    _line("step 2 prompt_tokens, NO tools", s2_notools)
    _line("step 2 prompt_tokens, WITH tools", f"{s2_tools}  (+{s2_tools - s2_notools})")

    print()
    d1 = s1_tools - s1_notools
    d2 = s2_tools - s2_notools
    _line("tools-block delta on step 1", d1)
    _line("tools-block delta on step 2", d2)
    if abs(d1 - d2) <= max(3, int(0.05 * d1)):
        print("  => VERDICT: the tools block is re-billed in FULL on every step (deltas match).")
    else:
        print(f"  => deltas differ ({d1} vs {d2}) -- inspect (caching or accounting nuance).")


def q3_caching(probe: Probe, registry: Any) -> None:
    _hdr("Q3 -- does OpenRouter prompt-CACHING reduce the bill on a repeated prefix?")
    specs: list[LLMToolSpec] = registry.eager_specs()
    wire_all = [_tool_to_wire(s) for s in specs]
    msgs = [_SYS, _USER]
    # Fire twice back-to-back; grok via OpenRouter may report cached_tokens on the 2nd.
    p1, c1 = probe.prompt_tokens(msgs, tools=wire_all)
    p2, c2 = probe.prompt_tokens(msgs, tools=wire_all)
    _line("call 1: prompt_tokens / cached_tokens", f"{p1} / {c1}")
    _line("call 2: prompt_tokens / cached_tokens", f"{p2} / {c2}")
    if c2 > 0:
        print(f"  => caching IS active: {c2}/{p2} tokens hit cache on the repeat.")
        print("     (NOTE: cached input is cheaper but usually NOT free -- still billed, at a discount.)")
    else:
        print("  => NO cached_tokens reported for this model/route: the full prefix is billed every call.")


def q4_compact_menu(probe: Probe, registry: Any) -> None:
    _hdr("Q4 -- compact plaintext tool MENU vs the native tools= block")
    specs: list[LLMToolSpec] = registry.eager_specs()
    wire_all = [_tool_to_wire(s) for s in specs]

    # The full native block cost (over the SAME system+user baseline).
    base, _ = probe.prompt_tokens([_SYS, _USER])
    full_native, _ = probe.prompt_tokens([_SYS, _USER], tools=wire_all)
    _line("native tools= block (all 21)", f"+{full_native - base} tok")

    # A compact plaintext menu: one line per tool, name + description, as a system message.
    # This is the "agent sees a cheap menu, then asks for the few it needs" idea.
    menu_lines = [f"- {s.name}: {s.description.splitlines()[0] if s.description else ''}" for s in specs]
    menu = "AVAILABLE TOOLS (ask to load the ones you need):\n" + "\n".join(menu_lines)
    menu_msgs = [_SYS, {"role": "system", "content": menu}, _USER]
    with_menu, _ = probe.prompt_tokens(menu_msgs)
    _line("compact plaintext menu (all 21, name+1-line desc)", f"+{with_menu - base} tok")

    # The eager-core-only native block (shader tools only): what a typical edit turn would
    # actually need if the long tail loaded lazily.
    shader_specs = [s for s in specs if s.name in {
        "read_shader", "edit_shader", "replace_lines", "insert_after",
        "set_uniform", "create_node", "grep", "read_lib", "delete_node", "switch_node",
    }]
    shader_wire = [_tool_to_wire(s) for s in shader_specs]
    with_shader, _ = probe.prompt_tokens([_SYS, _USER], tools=shader_wire)
    _line(f"native tools= block (shader-core only, {len(shader_wire)})", f"+{with_shader - base} tok")

    print()
    print("  Interpretation:")
    print(f"    full native (21):        +{full_native - base} tok / request, EVERY step")
    print(f"    shader-core native (10): +{with_shader - base} tok / request")
    print(f"    compact menu (21):       +{with_menu - base} tok (a cheap always-on index)")
    print("    => lazy design: ship shader-core natively + a 1-line menu of the long tail;")
    print("       attach a long-tail tool natively only once the agent asks for it.")


def q4b_two_stage_works(probe: Probe, registry: Any) -> None:
    _hdr("Q4b -- does a 2-stage 'pick tools from a menu' flow actually WORK?")
    # Give the model ONLY a meta-tool `load_tools(names)` + a plaintext menu, ask it to
    # publish to telegram, and check it picks the right tools to load.
    specs: list[LLMToolSpec] = registry.eager_specs()
    menu_lines = [f"- {s.name}: {(s.description.splitlines()[0] if s.description else '')[:90]}" for s in specs]
    menu = (
        "You can load tools on demand. Below is the MENU of available tools. To use any, "
        "first call load_tools(names=[...]) with the ones you need; they then become callable.\n\n"
        + "\n".join(menu_lines)
    )
    load_tools_spec = {
        "type": "function",
        "function": {
            "name": "load_tools",
            "description": "Load the named tools so you can call them this turn.",
            "parameters": {
                "type": "object",
                "properties": {"names": {"type": "array", "items": {"type": "string"}}},
                "required": ["names"],
            },
        },
    }
    msgs = [
        {"role": "system", "content": menu},
        {"role": "user", "content": "publish the current shader to my telegram sticker pack"},
    ]
    resp = probe.client.chat.completions.create(
        model=probe.model,
        messages=msgs,  # type: ignore[arg-type]  # dict literals vs the SDK's TypedDicts (throwaway probe)
        tools=[load_tools_spec],  # type: ignore[list-item]
        max_completion_tokens=200,
        extra_body={"reasoning": {"effort": "minimal"}},
    )
    choice = resp.choices[0]
    calls = choice.message.tool_calls or []
    if not calls:
        _line("model response", "NO tool call -- replied in text:")
        print("    " + (choice.message.content or "")[:300])
        return
    fns = [cast(Any, c).function for c in calls]
    for fn in fns:
        _line(f"model called {fn.name}", fn.arguments)
    # Did it pick the publish/telegram tools?
    try:
        args = json.loads(fns[0].arguments)
        picked = args.get("names", [])
        good = any("telegram" in p or "publish" in p for p in picked)
        print(f"\n  => picked {picked}")
        print(f"  => {'CORRECT' if good else 'MISSED'}: it {'did' if good else 'did NOT'} request a publish/telegram tool.")
    except (json.JSONDecodeError, AttributeError) as exc:
        print(f"  (couldn't parse names: {exc})")


def main() -> None:
    key, model = _load_key_and_model()
    print(f"model = {model}")
    client = OpenAI(base_url=_BASE_URL, api_key=key)
    probe = Probe(client=client, model=model)
    registry = _real_registry()
    print(f"registry: {len(registry.eager_specs())} eager tool specs")

    q1_anatomy(probe, registry)
    q2_resent_every_step(probe, registry)
    q3_caching(probe, registry)
    q4_compact_menu(probe, registry)
    q4b_two_stage_works(probe, registry)
    print("\n(done)")


if __name__ == "__main__":
    main()
