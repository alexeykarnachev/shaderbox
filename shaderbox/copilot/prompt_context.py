from dataclasses import dataclass

from shaderbox.copilot.capabilities import (
    CopilotCapabilities,
    LibCatalogEntry,
    NodeTreeEntry,
    TemplateEntry,
)

# Per-turn app-state snapshot, GL-FREE so it builds off-main. Rendered to text here so
# prompt.py stays a pure assembler. Current shader source is NOT here — it enters via the
# read_shader tool result.

# Conventions always in the prompt (never a tool). Keep terse — this is steering, not a manual.
_CONVENTIONS = """\
- Fragment shader: `#version 460 core`; read `in vec2 vs_uv` ([0,1], NO gl_FragCoord); write
  `out vec4 fs_color`; no `precision` qualifier (desktop GL).
- Library: `SB_` prefix, call by name (auto-resolves, no #include). Layered on SIGNED distance:
  `SB_sd_*` sources (negative inside) -> `SB_op_*` SDF transforms -> renderers
  (`SB_fill`/`SB_fill_aa`/`SB_glow`) -> 0..1 mask. Compose: source -> ops -> render.
- Uniforms: `u_` prefix. `u_time`/`u_aspect`/`u_resolution` are engine-driven (read, never set)
  but MUST still be declared — an undeclared one fails the compile (nothing is auto-injected).
- Keep helpers small/single-purpose so they factor into the library."""


@dataclass(frozen=True)
class CopilotContext:
    node_tree: str  # rendered project-map block (name/id/has_errors/is_current)
    lib_catalog: str  # rendered lib-catalogue block (name/signature/doc)
    template_catalog: (
        str  # rendered template-library block (name/template: handle/description)
    )
    conventions: str


def _render_node_tree(entries: list[NodeTreeEntry]) -> str:
    if not entries:
        return "(no shaders yet)"
    rows: list[str] = []
    for e in entries:
        marks: list[str] = []
        if e.is_current:
            marks.append("current")
        if e.has_errors:
            marks.append("HAS ERRORS")
        suffix = f"  [{', '.join(marks)}]" if marks else ""
        rows.append(f"- {e.name} (id: {e.node_id}){suffix}")
    return "\n".join(rows)


def _render_lib_catalog(entries: list[LibCatalogEntry]) -> str:
    if not entries:
        return "(library is empty)"
    rows: list[str] = []
    for e in sorted(entries, key=lambda x: x.name):
        doc = f" — {e.doc.strip()}" if e.doc.strip() else ""
        rows.append(f"- {e.signature}  ({e.lib_address}){doc}")
    return "\n".join(rows)


def _render_template_catalog(entries: list[TemplateEntry]) -> str:
    # name + the `template:` handle + one-line description.
    if not entries:
        return "(no templates)"
    rows: list[str] = []
    for e in entries:
        desc = f" — {e.description.strip()}" if e.description.strip() else ""
        rows.append(f"- {e.name} ({e.template_id}){desc}")
    return "\n".join(rows)


def build_context(caps: CopilotCapabilities) -> CopilotContext:
    return CopilotContext(
        node_tree=_render_node_tree(caps.node_tree()),
        lib_catalog=_render_lib_catalog(caps.lib_catalog()),
        template_catalog=_render_template_catalog(caps.template_catalog()),
        conventions=_CONVENTIONS,
    )
