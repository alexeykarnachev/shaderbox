#!/bin/bash
set -euo pipefail

# Build ShaderBox source distributions for itch.io.
#
# The bundle is an ALLOWLIST: only the files listed here ship. Coding-agent and
# dev-flow files (CLAUDE.md, ai_docs/, .claude/, Makefile, .pre-commit-config.yaml,
# conventions/roadmap, etc.) are never staged. A verification gate at the end aborts
# the build if any forbidden pattern slips into a staged tree — so the "no dev files
# in the bundle" guarantee is asserted, not merely assumed.

echo "Building ShaderBox distributions for itch.io..."
echo "================================================"

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

rm -rf dist
mkdir -p dist

# Root files that ship in every bundle (allowlist).
ROOT_FILES=(pyproject.toml uv.lock .python-version)
[ -f LICENSE ] && ROOT_FILES+=(LICENSE)

# Forbidden patterns — if any match in a staged tree, the build aborts.
FORBIDDEN_NAMES=(CLAUDE.md Makefile .pre-commit-config.yaml itch-config)
FORBIDDEN_PATHS=(ai_docs .claude __pycache__ .git .venv .ruff_cache .mypy_cache)

stage_common() {
    # $1 = staging dir. Copies the allowlisted payload, stripping bytecode.
    local stage="$1"
    cp -r shaderbox "$stage/"
    # Strip Python bytecode (stale .pyc for deleted modules must not ship).
    find "$stage/shaderbox" -type d -name __pycache__ -prune -exec rm -rf {} +
    find "$stage/shaderbox" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

    local f
    for f in "${ROOT_FILES[@]}"; do
        cp "$f" "$stage/"
    done

    # The single user-facing install guide (NOT the repo README, which has
    # itch-only screenshot links that would 404 in the bundle).
    cp scripts/README.md "$stage/"
}

verify_clean() {
    # $1 = staging dir. Aborts the build on any forbidden file.
    local stage="$1"
    local hit name path
    for name in "${FORBIDDEN_NAMES[@]}"; do
        hit="$(find "$stage" -name "$name" -print -quit)"
        [ -n "$hit" ] && { echo "✗ FORBIDDEN file in bundle: $hit"; exit 1; }
    done
    for path in "${FORBIDDEN_PATHS[@]}"; do
        hit="$(find "$stage" -name "$path" -print -quit)"
        [ -n "$hit" ] && { echo "✗ FORBIDDEN path in bundle: $hit"; exit 1; }
    done
    echo "✓ Bundle verified clean (no agent/dev/bytecode files): $(basename "$stage")"
}

build_platform() {
    # $1 = platform label (windows|linux), $2 = launcher script in scripts/.
    local platform="$1" launcher="$2"
    local stage="/tmp/shaderbox-build-$platform"

    echo "Building $platform distribution..."
    rm -rf "$stage"
    mkdir -p "$stage"

    stage_common "$stage"
    cp "scripts/$launcher" "$stage/"
    [ "$launcher" = "run.sh" ] && chmod +x "$stage/run.sh"

    verify_clean "$stage"
}

build_platform windows run.bat
build_platform linux run.sh

# Archive. Build-dir basenames stay stable (shaderbox-build-*) so the extracted
# top-level folder matches what the install guide expects.
( cd /tmp && zip -rq shaderbox-windows.zip shaderbox-build-windows/ && mv shaderbox-windows.zip "$ROOT/dist/" )
echo "✓ Windows distribution created: dist/shaderbox-windows.zip"

( cd /tmp && tar -czf shaderbox-linux.tar.gz shaderbox-build-linux/ && mv shaderbox-linux.tar.gz "$ROOT/dist/" )
echo "✓ Linux distribution created: dist/shaderbox-linux.tar.gz"

rm -rf /tmp/shaderbox-build-windows /tmp/shaderbox-build-linux

echo
echo "Build complete! Distribution files in 'dist/'."
echo "Next: ./upload-itch.sh (needs butler + an itch-config file)."
