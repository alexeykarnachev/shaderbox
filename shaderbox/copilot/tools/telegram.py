from typing import Any

from pydantic import BaseModel, Field

from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.gate import GateKind
from shaderbox.copilot.tools.base import GatePolicy, ToolDefinition, mask_secret

# The Telegram connect + pack-CRUD tool surface (feature 020·19). set_telegram_token is the one
# CREDENTIAL-gated tool: the chat shows a masked input, the secret arrives as the handler's 2nd
# arg (never in args/trace/history), the handler sets it via caps + auto-links, and returns a
# REDACTED confirmation. The pack tools are normal CONFIRM-gated mutations (list is ungated); all
# precheck on connection so an unconnected op never pops a confirm.


class _EmptyArgs(BaseModel):
    model_config = {"extra": "forbid"}


class _SetNameArgs(BaseModel):
    set_name: str = Field(description="the pack's set_name (from list_telegram_packs)")
    model_config = {"extra": "forbid"}


class _CreatePackArgs(BaseModel):
    title: str = Field(description="a human title for the new sticker pack")
    model_config = {"extra": "forbid"}


_SET_TOKEN_DESC = (
    "Connect Telegram: opens a SECURE input in the chat for the user to paste their bot token "
    "(from @BotFather) — you never see the token itself. After it's set, I try to link the "
    "account. If the user hasn't messaged the bot yet, tell them to open the bot and press Start, "
    "then call telegram_connect."
)
_CONNECT_DESC = (
    "Finish linking Telegram after the user has pressed Start / sent a message to their bot. Use "
    "this when set_telegram_token said the token is set but the account isn't linked yet."
)
_LIST_DESC = "List the user's saved Telegram sticker packs (the active one is marked)."
_SELECT_DESC = "Make a saved Telegram sticker pack the active one (for publishing)."
_CREATE_DESC = (
    "Register a new Telegram sticker pack and make it active. The pack becomes real on Telegram "
    "when the first sticker is published to it."
)
_DELETE_DESC = (
    "Delete a Telegram sticker pack — removes it from Telegram entirely (irreversible)."
)


def telegram_tools(caps: CopilotCapabilities) -> list[ToolDefinition]:
    def set_telegram_token(
        args: dict[str, Any], secret: str
    ) -> tuple[bool, str, dict | None]:
        # `secret` is the gate's typed token (out-of-band). caps.set_telegram_token sets it +
        # auto-links; the returned msg is REDACTED (the only thing reaching LLM history).
        _ = args
        if not secret.strip():
            return False, "error: no token entered", None
        result = caps.set_telegram_token(secret)
        red = mask_secret(secret)
        if result.ok:
            return True, f"token set ({red}) and linked to @{result.bot_username}.", None
        if result.needs_start:
            return (
                True,
                f"token set ({red}). Tell the user to open their bot in Telegram, press "
                "Start (or send any message), then ask me to connect.",
                None,
            )
        return False, f"token set ({red}), but linking failed: {result.error}", None

    def telegram_connect(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        _ = args
        result = caps.telegram_connect()
        if result.ok:
            return True, f"linked to @{result.bot_username}.", None
        if result.needs_start:
            return (
                True,
                "still no message from the user to the bot — tell them to open the bot and "
                "press Start, then ask me to connect again.",
                None,
            )
        return False, f"error: {result.error}", None

    def list_telegram_packs(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        _ = args
        packs = caps.list_telegram_packs()
        if not packs:
            return True, "no sticker packs yet — create one to get started.", {"packs": 0}
        rows = "\n".join(
            f"- {p.title} ({p.set_name}){' [active]' if p.is_default else ''}"
            for p in packs
        )
        return True, f"your sticker packs:\n{rows}", {"packs": len(packs)}

    def select_telegram_pack(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        result = caps.select_telegram_pack(args["set_name"])
        if not result.ok:
            return False, f"error: {result.error}", None
        return True, f"active pack is now '{result.set_name}'.", None

    def create_telegram_pack(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        result = caps.create_telegram_pack(args["title"])
        if not result.ok:
            return False, f"error: {result.error}", None
        return (
            True,
            f"created pack '{args['title']}' ({result.set_name}) and made it active. "
            "Publish a shader to it to actually create it on Telegram.",
            None,
        )

    def delete_telegram_pack(args: dict[str, Any]) -> tuple[bool, str, dict | None]:
        result = caps.delete_telegram_pack(args["set_name"])
        if not result.ok:
            return False, f"error: {result.error}", None
        return True, f"deleted pack '{args['set_name']}' from Telegram.", None

    def connected_precheck(args: dict[str, Any]) -> str | None:
        _ = args
        if not caps.telegram_connected():
            return (
                "Telegram isn't connected. Use set_telegram_token to connect first, then try "
                "again."
            )
        return None

    return [
        ToolDefinition(
            name="set_telegram_token",
            description=_SET_TOKEN_DESC,
            args_model=_EmptyArgs,
            handler=set_telegram_token,
            mutating=True,
            needs_gl=False,
            category="telegram",
            eager=True,
            gate_policy=GatePolicy.ALWAYS,
            gate_kind=GateKind.CREDENTIAL,
            secret_field="telegram_bot_token",
        ),
        ToolDefinition(
            name="telegram_connect",
            description=_CONNECT_DESC,
            args_model=_EmptyArgs,
            handler=telegram_connect,
            mutating=True,
            needs_gl=False,
            category="telegram",
            eager=True,
            gate_policy=GatePolicy.NONE,
        ),
        ToolDefinition(
            name="list_telegram_packs",
            description=_LIST_DESC,
            args_model=_EmptyArgs,
            handler=list_telegram_packs,
            mutating=False,
            needs_gl=False,
            category="telegram",
            eager=True,
            gate_policy=GatePolicy.NONE,
            precheck=connected_precheck,
        ),
        ToolDefinition(
            name="select_telegram_pack",
            description=_SELECT_DESC,
            args_model=_SetNameArgs,
            handler=select_telegram_pack,
            mutating=True,
            needs_gl=False,
            category="telegram",
            eager=True,
            gate_policy=GatePolicy.ALWAYS,
            precheck=connected_precheck,
        ),
        ToolDefinition(
            name="create_telegram_pack",
            description=_CREATE_DESC,
            args_model=_CreatePackArgs,
            handler=create_telegram_pack,
            mutating=True,
            needs_gl=False,
            category="telegram",
            eager=True,
            gate_policy=GatePolicy.ALWAYS,
            precheck=connected_precheck,
        ),
        ToolDefinition(
            name="delete_telegram_pack",
            description=_DELETE_DESC,
            args_model=_SetNameArgs,
            handler=delete_telegram_pack,
            mutating=True,
            needs_gl=False,
            category="telegram",
            eager=True,
            gate_policy=GatePolicy.ALWAYS,
            precheck=connected_precheck,
        ),
    ]
