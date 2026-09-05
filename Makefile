.PHONY: run run-bundle check test smoke gates release
.ONESHELL:

# Run the app from source (the dev / personal-use path).
run:
	uv run python ./shaderbox/ui.py

# Verify the BUILT bundle the way a NEW user would: build fresh, unzip, run the
# launcher with a throwaway data dir — a true fresh first-run (starter document seeded,
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
# render_media) against a headless standalone moderngl context; app-fixture
# modules skip without a display. The MESA overrides are LOAD-BEARING: the
# shaders are #version 460 and compiling 460 on a bare llvmpipe 4.5 context
# SEGFAULTS Mesa (the "GL segfault modules" class; same overrides the dogfood
# harness sets). GLCONTEXT_LINUX_LIBGL avoids the libGL.so dev-symlink dlopen
# failure on boxes without libgl-dev.
# `-n 8`: the suite is fixture-bound, not assertion-bound -- 136 tests take the `app`
# fixture and each pays ~90ms building a real App (a window, a GL context, the shipped
# example documents). Serial that is ~53s; 8 workers is ~15s. Measured 4/8/12/24 workers:
# the curve flattens after 8, so more workers buy ~0.5s and cost more memory. Each worker
# is its OWN PROCESS with its own glfw window and GL context -- xdist does not fork after
# fixture setup, which is what makes this safe where `pytest-forked` is not (a forked child
# inheriting the parent's X11 socket kills the connection; see test_revert_executor.py).
test:
	env MESA_GL_VERSION_OVERRIDE=4.6 MESA_GLSL_VERSION_OVERRIDE=460 \
		GLCONTEXT_LINUX_LIBGL=libGL.so.1 uv run pytest tests/ -n 8

# Headless smoke test — runs ~200 frames of update_and_draw against a THROWAWAY tmp project
# in an invisible glfw window. Catches import errors, callback dispatch failures,
# popup state-machine crashes, released-texture binding errors. Doesn't catch
# visual bugs. Useful after any refactor in ui.py / app.py / widgets/ / popups/ /
# tabs/ / hotkeys.py before declaring done.
smoke:
	uv run python scripts/smoke.py

# The composed gate: check -> test -> smoke, cheapest first, stopping at the first
# failure, one exit code for the lot. This target exists because a prose rule did not
# hold: the repo twice announced a green gate that was red, once by reading a tool's
# output instead of its status and once by capturing a pipe's exit code. So nothing
# here is piped -- each gate redirects to a log and its status is read before anything
# touches that log (make's shell is dash; `set -o pipefail` is not available anyway).
#
# `check` runs twice before it is believed. pre-commit exits non-zero whenever a hook
# MODIFIED a file, which ruff's formatter and pyright's env bootstrap both legitimately
# do on a first run -- so a first-run non-zero says "something changed", not "something
# is wrong". A second run over the now-settled tree is the one that carries a verdict.
# This is the rule `dev_flow.md ### make check` states in prose; here it is executable.
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
		# A first-run failure is only benign when the hooks changed NOTHING -- pyright's env
		# bootstrap is the case that earns the retry. `ruff --fix` and `ruff-format` are
		# file-rewriting hooks, so retrying unconditionally turned every auto-fixable
		# violation into a GREEN gate with an uncommitted repair in the tree, and this repo
		# shipped exactly that.
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
		# The log, on EVERY failing path -- a red gate with no diagnostic is worse than no
		# gate, and deduplicating the verdict message once cost exactly that on the retry path.
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
