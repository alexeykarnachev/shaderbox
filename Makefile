.PHONY: check smoke

# Lint + format + typecheck. Run before declaring anything done.
# Delegates to pre-commit (ruff fix, ruff format, pyright) — the single source of
# truth for the check config is `.pre-commit-config.yaml`. Both ruff and pyright
# block on failure.
check:
	uv run pre-commit run --all-files

# Headless smoke test — runs ~200 frames of update_and_draw against projects/dev/
# in an invisible glfw window. Catches import errors, callback dispatch failures,
# popup state-machine crashes, released-texture binding errors. Doesn't catch
# visual bugs. Useful after any refactor in ui.py / app.py / widgets/ / popups/ /
# tabs/ / hotkeys.py before declaring done.
smoke:
	uv run python scripts/smoke.py
