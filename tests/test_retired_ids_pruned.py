"""Retired ids do not outlive the thing they named.

Both of these dicts are keyed by an id that a future release can remove. Nothing pruned
them, so a removed exporter's settings block and a retired command's rebinding sat in the
user's state file indefinitely — the tracked sandbox state still carried a block for an
exporter deleted commits earlier.
"""

from shaderbox.commands import COMMAND_SPECS, CommandId


def test_effective_bindings_drop_a_retired_command(app) -> None:
    app.app_state.key_bindings["a_command_that_no_longer_exists"] = 1234
    live_id = COMMAND_SPECS[0].id
    app.app_state.key_bindings[live_id.value] = 5678

    app._merge_effective_bindings()

    assert "a_command_that_no_longer_exists" not in app.app_state.key_bindings
    assert app.app_state.key_bindings[live_id.value] == 5678
    assert app.effective_bindings[live_id] == 5678


def test_saved_exporter_settings_hold_only_live_exporters(app) -> None:
    app.app_state.exporter_settings["a_retired_exporter"] = {"stale": True}

    app.save()

    assert set(app.app_state.exporter_settings) == set(app.exporter_registry.ids())
    assert "a_retired_exporter" not in app.app_state.exporter_settings


def test_a_live_rebinding_survives_the_prune(app) -> None:
    # The other side of the bound: pruning must not eat real user rebindings.
    target: CommandId = COMMAND_SPECS[1].id
    app.app_state.key_bindings[target.value] = 4242
    app._merge_effective_bindings()
    assert app.effective_bindings[target] == 4242
