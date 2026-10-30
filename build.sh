#!/bin/bash
set -euo pipefail

# Build ShaderBox source distributions for itch.io.
#
# The bundle is an allowlist: only the files listed here ship, and a check at the
# end aborts the build if anything else slips into a staged tree.

echo "Building ShaderBox distributions for itch.io..."
echo "================================================"

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# --allow-dirty skips the clean-tree guard; the test gate always runs.
ALLOW_DIRTY=0
[ "${1:-}" = "--allow-dirty" ] && ALLOW_DIRTY=1

if [ "$ALLOW_DIRTY" -eq 0 ] && [ -n "$(git status --porcelain)" ]; then
    echo "✗ working tree is dirty — commit, stash, or pass --allow-dirty:"
    git status --short
    exit 1
fi

# Gate: never bundle code that fails lint/typecheck or the headless smoke test.
make check
make smoke

rm -rf dist
# Staged inside dist/ rather than /tmp: the zipper is a native Python, which on
# Windows cannot read Git Bash's /tmp mount, and a relative path reads the same
# under both.
STAGE_ROOT=dist/stage
mkdir -p "$STAGE_ROOT"

# Root files that ship in every bundle (allowlist).
ROOT_FILES=(pyproject.toml uv.lock .python-version)
[ -f LICENSE ] && ROOT_FILES+=(LICENSE)

# The vendored native libraries, relative to shaderbox/resources/, without the
# platform suffix (.so on Linux, .dll on Windows).
NATIVE_LIBS=(editor/libeditor graph_canvas/libgraph_canvas)

# Forbidden patterns — if any match in a staged tree, the build aborts.
FORBIDDEN_NAMES=(CLAUDE.md Makefile .pre-commit-config.yaml itch-config BUILDING.md)
FORBIDDEN_PATHS=(ai_docs .claude .github __pycache__ .git .venv .ruff_cache .mypy_cache)

stage_common() {
    # $1 = staging dir, $2 = platform label. Copies the allowlisted payload,
    # stripping bytecode.
    local stage="$1" platform="$2"
    cp -r shaderbox "$stage/"
    # The native libraries ship one build per platform side by side; each stage
    # keeps its own and drops the other's.
    local lib
    for lib in "${NATIVE_LIBS[@]}"; do
        if [ "$platform" = "windows" ]; then
            rm -f "$stage/shaderbox/resources/$lib.so"
        else
            rm -f "$stage/shaderbox/resources/$lib.dll"
        fi
    done
    # Strip Python bytecode (stale .pyc for deleted modules must not ship).
    find "$stage/shaderbox" -type d -name __pycache__ -prune -exec rm -rf {} +
    find "$stage/shaderbox" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

    local f
    for f in "${ROOT_FILES[@]}"; do
        cp "$f" "$stage/"
    done

    # The user-facing install guide.
    cp scripts/README.md "$stage/"
}

verify_clean() {
    # $1 = staging dir, $2 = platform label. Aborts the build on any forbidden file.
    local stage="$1" platform="$2"
    local hit name path
    local own foreign lib
    if [ "$platform" = "windows" ]; then own=dll; foreign=so; else own=so; foreign=dll; fi
    hit="$(find "$stage" -name "*.$foreign" -print -quit)"
    [ -n "$hit" ] && { echo "✗ FORBIDDEN .$foreign in the $platform bundle: $hit"; exit 1; }
    for lib in "${NATIVE_LIBS[@]}"; do
        [ -f "$stage/shaderbox/resources/$lib.$own" ] || {
            echo "✗ MISSING $lib.$own in the $platform bundle: the app cannot start without it"
            exit 1
        }
    done
    for name in "${FORBIDDEN_NAMES[@]}"; do
        hit="$(find "$stage" -name "$name" -print -quit)"
        [ -n "$hit" ] && { echo "✗ FORBIDDEN file in bundle: $hit"; exit 1; }
    done
    for path in "${FORBIDDEN_PATHS[@]}"; do
        hit="$(find "$stage" -name "$path" -print -quit)"
        [ -n "$hit" ] && { echo "✗ FORBIDDEN path in bundle: $hit"; exit 1; }
    done
    echo "✓ Bundle verified clean: $(basename "$stage")"
}

build_platform() {
    # $1 = platform label (windows|linux), $2 = launcher script in scripts/.
    local platform="$1" launcher="$2"
    local stage="$STAGE_ROOT/shaderbox-build-$platform"

    echo "Building $platform distribution..."
    rm -rf "$stage"
    mkdir -p "$stage"

    stage_common "$stage" "$platform"
    cp "scripts/$launcher" "$stage/"
    [ "$launcher" = "run.sh" ] && chmod +x "$stage/run.sh"

    verify_clean "$stage" "$platform"
}

build_platform windows run.bat
build_platform linux run.sh

# Archive. Build-dir basenames stay stable (shaderbox-build-*) so the extracted
# top-level folder matches what the install guide expects.
for platform in windows linux; do
    uv run python scripts/zip_bundle.py "$STAGE_ROOT/shaderbox-build-$platform" "dist/shaderbox-$platform.zip"
    echo "✓ ${platform^} distribution created: dist/shaderbox-$platform.zip"
done

rm -rf "$STAGE_ROOT"

echo
echo "Build complete! Distribution files in 'dist/'."
echo "Next: ./upload-itch.sh (needs butler + an itch-config file)."
