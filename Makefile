.PHONY: run run-bundle check test smoke release
.ONESHELL:

# Run the app from source (the dev / personal-use path).
run:
	uv run python ./shaderbox/ui.py

# Verify the BUILT bundle the way a NEW user would: build fresh, unzip, run the
# launcher with a throwaway data dir — a true fresh first-run (starter node seeded,
# no existing projects). Rebuilds via --allow-dirty so it tests CURRENT source (incl.
# uncommitted work), never a stale dist/. SHADERBOX_DATA_DIR redirects the app's
# project/state/log store away from the real one, so this never touches your projects.
run-bundle:
	@set -e
	./build.sh --allow-dirty
	rm -rf /tmp/shaderbox-run-bundle /tmp/shaderbox-run-bundle-data
	unzip -q dist/shaderbox-linux.zip -d /tmp/shaderbox-run-bundle
	cd /tmp/shaderbox-run-bundle/shaderbox-build-linux && SHADERBOX_DATA_DIR=/tmp/shaderbox-run-bundle-data ./run.sh

# Lint + format + typecheck. Run before declaring anything done.
# Delegates to pre-commit (ruff fix, ruff format, pyright) — the single source of
# truth for the check config is `.pre-commit-config.yaml`. Both ruff and pyright
# block on failure.
check:
	uv run pre-commit run --all-files

# Unit tests. Pure logic (resolve_dims) + GL-backed render glue (render_for,
# render_media) against a headless standalone moderngl context; the GL module
# skips if no GL driver is available rather than failing.
test:
	uv run pytest tests/ -q

# Headless smoke test — runs ~200 frames of update_and_draw against projects/dev/
# in an invisible glfw window. Catches import errors, callback dispatch failures,
# popup state-machine crashes, released-texture binding errors. Doesn't catch
# visual bugs. Useful after any refactor in ui.py / app.py / widgets/ / popups/ /
# tabs/ / hotkeys.py before declaring done.
smoke:
	uv run python scripts/smoke.py

# Cut a release: bump pyproject version, commit, tag. Does NOT build or push
# (./build.sh then ./upload-itch.sh stay separate). Semver bump policy lives in
# conventions.md ## Design decisions. Usage: make release VERSION=x.y.z
release:
	@set -e
	if [ -z "$(VERSION)" ]; then echo "usage: make release VERSION=x.y.z"; exit 1; fi
	echo "$(VERSION)" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$$' || { echo "VERSION must be semver core x.y.z"; exit 1; }
	test -z "$$(git status --porcelain)" || { echo "working tree dirty; commit or stash first"; exit 1; }
	git rev-parse -q --verify "refs/tags/v$(VERSION)" >/dev/null && { echo "tag v$(VERSION) already exists"; exit 1; } || true
	sed -i 's/^version = ".*"/version = "$(VERSION)"/' pyproject.toml
	$(MAKE) check || { echo "check failed; rolling back version edit"; git checkout -- pyproject.toml; exit 1; }
	git commit -aqm "release: v$(VERSION)"
	git tag "v$(VERSION)"
	echo "tagged v$(VERSION). next: ./build.sh && ./upload-itch.sh"
