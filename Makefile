.PHONY: run run-bundle check test smoke gates release
.ONESHELL:

# Run the app from source (the dev / personal-use path).
run:
	uv run python ./shaderbox/ui.py

# Build the bundle from the current source (uncommitted work included), unzip it and run
# the launcher as a first-time user would. SHADERBOX_DATA_DIR points the app's projects,
# state and logs at a throwaway directory.
ifeq ($(OS),Windows_NT)
BUNDLE_PLATFORM := windows
BUNDLE_LAUNCH := cmd //c run.bat
else
BUNDLE_PLATFORM := linux
BUNDLE_LAUNCH := ./run.sh
endif

run-bundle:
	@set -e
	./build.sh --allow-dirty
	root="$$(pwd)/dist/run-bundle"
	rm -rf "$$root"
	unzip -q dist/shaderbox-$(BUNDLE_PLATFORM).zip -d "$$root"
	cd "$$root/shaderbox-build-$(BUNDLE_PLATFORM)" && SHADERBOX_DATA_DIR="$$root/data" $(BUNDLE_LAUNCH)

# Lint, format and typecheck through pre-commit (`.pre-commit-config.yaml`).
# The files are tracked plus untracked-not-ignored (`--all-files` would mean tracked only), so a
# new module is checked before it is staged; NUL-separated, so any file name reaches the hooks
# whole. The existence filter drops a tracked file deleted from the working tree.
check:
	git ls-files -z --cached --others --exclude-standard \
		| xargs -0 sh -c 'for f; do [ -e "$$f" ] && printf "%s\0" "$$f"; done' _ \
		| xargs -0 uv run pre-commit run --files

# Unit tests. The MESA overrides are required: the shaders are #version 460, and compiling
# 460 on a bare llvmpipe 4.5 context segfaults Mesa. GLCONTEXT_LINUX_LIBGL avoids the
# libGL.so dev-symlink dlopen failure on boxes without libgl-dev.
# Each xdist worker is its own process with its own glfw window and GL context, which is
# what makes this safe where `pytest-forked` is not: a forked child inheriting the parent's
# X11 socket kills the connection. `--max-worker-restart=0` makes a dead worker a failure
# instead of a run that waits on it forever; the per-test deadline in pytest.ini turns a
# hang into that death. PYTHONDONTWRITEBYTECODE keeps __pycache__ from going stale against a
# source edited and restored in place.
test:
	env MESA_GL_VERSION_OVERRIDE=4.6 MESA_GLSL_VERSION_OVERRIDE=460 \
		GLCONTEXT_LINUX_LIBGL=libGL.so.1 PYTHONDONTWRITEBYTECODE=1 \
		uv run pytest tests/ -n 8 --dist loadgroup \
		--max-worker-restart=0

# Headless smoke test: ~200 frames of update_and_draw against a throwaway project in an
# invisible glfw window. Catches import errors, callback dispatch failures, popup
# state-machine crashes and released-texture binding errors; not visual bugs.
smoke:
	uv run python scripts/smoke.py

# check -> test -> smoke, cheapest first, stopping at the first failure, one exit code for
# the lot. Nothing here is piped: each step redirects to a log and its status is read before
# anything touches that log (make's shell is dash, which has no `set -o pipefail`).
#
# pre-commit exits non-zero whenever a hook modified a file, which pyright's env bootstrap
# does on a first run, so a failing `check` that changed nothing is run once more.
#
# Three outcomes, not two. `scripts/smoke.py` exits 0 when it skips for want of a
# display, which is the right contract for its direct callers but makes a skip
# indistinguishable from a pass to a composing caller -- and a display-less box hits
# that branch every run. SHADERBOX_SMOKE_SKIP_EXIT lifts the skip onto its own code so
# the summary can say "skipped" instead of quietly scoring it as "passed". That code has
# to survive the call, and a sub-make does not preserve it -- make reports its own 2 for
# any failed child -- so this one gate runs `scripts/smoke.py` directly instead of via
# `$(MAKE) smoke`. The `smoke` target stays the entry point for everyone else.
gates:
	@set -e
	log=$${TMPDIR:-/tmp}/shaderbox-gates.log
	rm -f "$$log"
	status=0
	smoke_outcome=passed
	echo "== gates: check =="
	rc=0
	before=$$(git diff HEAD 2>/dev/null | md5sum; git status --porcelain 2>/dev/null | sort)
	$(MAKE) --no-print-directory check >"$$log" 2>&1 || rc=$$?
	if [ $$rc -ne 0 ]; then
		# Retried only when the hooks changed nothing: a retry after `ruff --fix` or
		# `ruff-format` rewrote a file would pass over an unreviewed repair.
		after=$$(git diff HEAD 2>/dev/null | md5sum; git status --porcelain 2>/dev/null | sort)
		if [ "$$after" != "$$before" ]; then
			echo "== gates: the hooks REWROTE files -- review and stage them, then re-run =="
		else
			echo "== gates: check exited $$rc leaving the tree unchanged; re-running =="
			rc=0
			$(MAKE) --no-print-directory check >"$$log" 2>&1 || rc=$$?
		fi
	fi
	if [ $$rc -ne 0 ]; then
		# The log, on every failing path.
		cat "$$log"
		echo "== gates: FAILED at check (exit $$rc); test and smoke not run =="
		status=$$rc
	else
		echo "== gates: check passed =="
		echo "== gates: test =="
		rc=0
		$(MAKE) --no-print-directory test >"$$log" 2>&1 || rc=$$?
		if [ $$rc -ne 0 ]; then
			cat "$$log"
			echo "== gates: FAILED at test (exit $$rc); smoke not run =="
			status=$$rc
		else
			echo "== gates: test passed =="
			echo "== gates: smoke =="
			rc=0
			SHADERBOX_SMOKE_SKIP_EXIT=87 uv run python scripts/smoke.py >"$$log" 2>&1 || rc=$$?
			if [ $$rc -eq 87 ]; then
				cat "$$log"
				echo "== gates: smoke SKIPPED (no GPU window on this box) =="
				smoke_outcome=skipped
			elif [ $$rc -ne 0 ]; then
				cat "$$log"
				echo "== gates: FAILED at smoke (exit $$rc) =="
				status=$$rc
			else
				echo "== gates: smoke passed =="
			fi
		fi
	fi
	if [ ! -t 1 ]; then
		echo "== gates: stdout is not a terminal. If you piped this, \$$? is the PIPE's"
		echo "==        exit code, not the gate's -- use PIPESTATUS, or redirect to a"
		echo "==        file and read \$$? before anything else runs. =="
	fi
	if [ $$status -ne 0 ]; then
		echo "== gates: RED (exit $$status) =="
	elif [ "$$smoke_outcome" = skipped ]; then
		echo "== gates: GREEN, smoke SKIPPED -- check passed, test passed, smoke did not run =="
	else
		echo "== gates: GREEN -- check passed, test passed, smoke passed =="
	fi
	exit $$status

# Cut a release: bump the pyproject version, commit, tag. Does not build or push
# (./build.sh then ./upload-itch.sh stay separate). Usage: make release VERSION=x.y.z
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
