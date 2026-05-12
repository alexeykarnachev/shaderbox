.PHONY: check

# Lint + format + typecheck. Run before declaring anything done.
# Delegates to pre-commit (ruff fix, ruff format, pyright) — the single source of
# truth for the check config is `.pre-commit-config.yaml`. Note: pyright is
# non-blocking for now (pre-existing type debt in ui.py) but its findings print.
check:
	uv run pre-commit run --all-files
