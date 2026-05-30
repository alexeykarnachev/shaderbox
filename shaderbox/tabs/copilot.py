from shaderbox.app import App
from shaderbox.ui_primitives import caption_text, unconnected_gate

# The Copilot tab (feature 020) — a 4th tab beside Node/Render/Share. SCAFFOLD shell:
# it reads app.copilot.state and renders the not-configured gate. The transcript /
# input / streaming / tool-call UI is the later capability + UX brainstorm (§0 #3/#8);
# this wave stands the tab up so the wiring + nav (Ctrl+4) are in place.


def draw(app: App) -> None:
    if not app.integrations_store.copilot.openrouter_key:
        unconnected_gate(
            not_connected_msg="Copilot is not set up.",
            hint="Add your OpenRouter API key in Settings to enable the copilot.",
            action_label="Open Settings",
            on_action=app.open_settings,
        )
        return

    caption_text("Copilot chat — coming in a later wave.")
