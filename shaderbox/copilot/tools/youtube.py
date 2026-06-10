from typing import Any

from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.gate import GateKind
from shaderbox.copilot.tools.base import EmptyArgs, GatePolicy, ToolDefinition

# The YouTube connect tool (feature 020·21). set_youtube_credentials is CONFIG-gated: the chat renders
# the exporter's EXISTING draw_config_ui() inline (paste client_secret + Connect, the same widgets as
# Settings) + a Cancel button. The gate blocks until the panel reaches connected (auto-resolved) or the
# user cancels; the handler then only reports the outcome. Mirrors the Telegram connect shape, but the
# inline panel IS the multi-step flow, so no separate async-connect tail tool is needed.


_SET_YT_DESC = (
    "THIS is how YOU connect YouTube — call it whenever the user wants to connect / set up YouTube. "
    "It opens the YouTube setup panel INLINE in the chat (a short instruction + a field to paste the "
    "client_secret JSON + a Connect button that opens the browser sign-in), plus a Cancel button. Do "
    "NOT just tell the user to go to Settings — you have this panel. If they cancel, you'll be told."
)


def youtube_tools(caps: CopilotCapabilities) -> list[ToolDefinition]:
    def set_youtube_credentials(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        # Runs only AFTER the CONFIG gate resolved approved=True — i.e. the inline panel reached
        # connected. A decline routes around execute (the gate's decline handoff), so reaching here
        # means connected. Confirm defensively.
        _ = args
        if caps.youtube_connected():
            return True, "YouTube is connected — you can publish now.", None
        return (
            False,
            "YouTube still isn't connected. Ask the user to finish the setup panel, or to "
            "connect in Settings -> Integrations -> YouTube.",
            None,
        )

    def already_connected_precheck(args: dict[str, Any]) -> str | None:
        # If YouTube is already connected, skip the panel entirely (don't pop a setup gate for a
        # connection that exists) — exactly like the publish precheck handoffs (feature 020·18).
        _ = args
        if caps.youtube_connected():
            return "YouTube is already connected — no setup needed; you can publish."
        return None

    return [
        ToolDefinition(
            name="set_youtube_credentials",
            label_live="Setting YouTube credentials",
            label_done="Set YouTube credentials",
            description=_SET_YT_DESC,
            args_model=EmptyArgs,
            handler=set_youtube_credentials,
            mutating=True,
            needs_gl=False,
            category="youtube",
            eager=True,
            gate_policy=GatePolicy.ALWAYS,
            gate_prompt=lambda a: (
                "Set up YouTube below: paste your client_secret JSON, then press "
                "Connect (a browser sign-in opens). Or Cancel."
            ),
            gate_kind=GateKind.CONFIG,
            secret_field="youtube",  # names the exporter whose draw_config_ui the card renders
            precheck=already_connected_precheck,
        ),
    ]
