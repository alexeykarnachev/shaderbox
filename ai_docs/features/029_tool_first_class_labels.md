# 029 — Tool labels as first-class tool properties (deferred)

> **STATUS: PENDING — deferred during the 028 UI/UX wave.** Filed to capture the problem; not yet
> designed/locked. Pick up after 028 ships.

## Problem

A tool's human-readable label currently lives in a parallel `dict[str, str]` keyed by tool-name string
(`copilot/state.py::_TOOL_VERBS` / `tool_label()`), separate from the tool entity itself. The tool is
ALREADY a first-class object — `ToolDefinition` (frozen dataclass: `name`, `description`, `args_model`,
`handler` callable, `mutating`, `needs_gl`, `category`, gate policy, …) built once per tool in a
`*_tools(caps)` builder and resolved through the single `ToolRegistry._by_name`. So the label is the one
property that ISN'T on the entity, which means it drifts: add a tool → its label is silently missing
(falls back to the raw `create_node` id) until someone remembers the side dict. (`ToolRegistry.status_for`
is the same smell — a stub that returns the raw name instead of a real human phrase.)

This is exactly the coupling the maintainer rejected: there must be ONE entity holding name + description
+ callable + label + gate policy + schema, defined in one place, with no parallel maps.

## Direction (not locked)

- Add the label to `ToolDefinition` (the one definition site per tool), resolve it through the registry
  (`ToolRegistry.label_for(name)` / fold into the existing `status_for`), delete `_TOOL_VERBS`/`tool_label`
  from `state.py`. Every `ToolDefinition(...)` in `tools/{shader,publish,telegram,youtube}.py` gets a label.
- Open question: ONE label field (present-participle, e.g. "Editing shader" — serves both the live status
  line AND the snippet hover) vs TWO tenses (present for the live line, past for the finished-step hover).
  Lean one field unless the hover tense grates in practice.
- Possibly the broader "tools as first-class entities" cleanup the maintainer gestured at — assess whether
  `ToolDefinition` already suffices (it largely does) or wants a tighter constructor/decorator shape.

## Files (when picked up)
- `copilot/tools/base.py` (`ToolDefinition` field), `copilot/tools/registry.py` (resolver), the 4
  `tools/*.py` builders (fill labels), `copilot/state.py` (delete the dict), `copilot/session.py` +
  `widgets/copilot_chat.py` (call the registry resolver).
