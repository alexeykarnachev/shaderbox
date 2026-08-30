from dataclasses import dataclass

from shaderbox.copilot.capabilities import (
    CopilotCapabilities,
    DocumentTreeEntry,
    ExampleEntry,
    LibCatalogEntry,
)
from shaderbox.scripting.api_doc import script_api_summary

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
- The canvas is NOT square in general: `vs_uv` is [0,1] on BOTH axes, so anything that must keep
  true proportions (a circle staying round, a square staying square, even spacing) needs
  aspect-corrected coordinates — e.g. center then `uv.x *= u_aspect` — before the shape math.
  After that correction x spans +-u_aspect/2 (y stays +-0.5): DERIVE layout positions and sizes
  from that live range, never from fixed constants that assume a wide (or square) canvas.
- `vs_uv.y` grows UPWARD: y=0 is the BOTTOM of the screen, y=1 the TOP. The user's spatial words
  are SCREEN words — "top row" / "upper left" mean HIGH y — so map row/placement indices through
  that inversion explicitly (top row = the highest-y band).
- TEXT content: a caption is `uniform uint u_text[64];` fed by set_uniform -- NEVER a const array in
  source (a dynamically indexed const array is demoted to per-thread local memory on NVIDIA, ~100x
  slower).
- Keep helpers small/single-purpose so they factor into the library."""


@dataclass(frozen=True)
class CopilotContext:
    document_tree: str  # rendered project-map block (name/id/has_errors/is_current)
    lib_catalog: str  # rendered lib-catalogue block (name/signature/doc)
    example_catalog: (
        str  # rendered example-library block (name/example: handle/description)
    )
    script_api: str  # generated SCRIPT API block (the Python side of a document script)
    conventions: str


def _render_document_tree(entries: list[DocumentTreeEntry]) -> str:
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
        rows.append(f"- {e.name} (id: {e.document_id}){suffix}")
    return "\n".join(rows)


def _render_lib_catalog(entries: list[LibCatalogEntry]) -> str:
    if not entries:
        return "(library is empty)"
    rows: list[str] = []
    for e in sorted(entries, key=lambda x: x.name):
        doc = f" — {e.doc.strip()}" if e.doc.strip() else ""
        rows.append(f"- {e.signature}  ({e.lib_address}){doc}")
    return "\n".join(rows)


def _render_example_catalog(entries: list[ExampleEntry]) -> str:
    # name + the `example:` handle + one-line description.
    if not entries:
        return "(no examples)"
    rows: list[str] = []
    for e in entries:
        desc = f" — {e.description.strip()}" if e.description.strip() else ""
        rows.append(f"- {e.name} ({e.example_id}){desc}")
    return "\n".join(rows)


def build_context(caps: CopilotCapabilities) -> CopilotContext:
    return CopilotContext(
        document_tree=_render_document_tree(caps.document_tree()),
        lib_catalog=_render_lib_catalog(caps.lib_catalog()),
        example_catalog=_render_example_catalog(caps.example_catalog()),
        script_api=script_api_summary(),
        conventions=_CONVENTIONS,
    )
