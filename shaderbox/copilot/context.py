from dataclasses import dataclass

from shaderbox.copilot.capabilities import CopilotCapabilities

# Per-turn app-state snapshot, built from GL-free reads via the capability seam. The
# ACTUAL fields (current node source, uniforms, compile errors, lib functions) are the
# later capability/prompt brainstorm (§0 #8) — this is the scaffold shape + the
# build-from-capabilities seam.


@dataclass(frozen=True)
class CopilotContext:
    current_node_id: str
    # node source / uniforms / compile errors / lib catalog land here in a later wave.


def build_context(caps: CopilotCapabilities) -> CopilotContext:
    return CopilotContext(current_node_id=caps.current_node_id())
