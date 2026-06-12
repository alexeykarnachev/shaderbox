"""The copilot chat focus-grab guard (no GL, no imgui).

A background window that calls set_next_window_focus() every frame while a popup is open
dismisses that popup — imgui reads the focus-steal as a close (/imgui-ui §8). The cold-start
gate (no key) exposed it: the focus-pending latch is set on open but its usual consumer (the
transcript's set_keyboard_focus_here) never runs without a key, so the chat re-grabbed focus
every frame and the Settings modal flickered open/closed, dimming everything with no content.
"""

from shaderbox.widgets.copilot_chat import should_grab_chat_focus


def test_grabs_focus_when_pending_and_no_popup() -> None:
    assert should_grab_chat_focus(
        focus_pending=True, in_flight=False, any_popup_open=False
    )


def test_grabs_focus_mid_turn_when_no_popup() -> None:
    assert should_grab_chat_focus(
        focus_pending=False, in_flight=True, any_popup_open=False
    )


def test_yields_focus_while_popup_open_even_if_pending() -> None:
    # The regression: a popup is open AND the latch is set -> must NOT grab (it would dismiss
    # the popup). Both drivers of the grab are suppressed while a popup owns focus.
    assert not should_grab_chat_focus(
        focus_pending=True, in_flight=False, any_popup_open=True
    )
    assert not should_grab_chat_focus(
        focus_pending=False, in_flight=True, any_popup_open=True
    )


def test_no_grab_when_idle() -> None:
    assert not should_grab_chat_focus(
        focus_pending=False, in_flight=False, any_popup_open=False
    )
