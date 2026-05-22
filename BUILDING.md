# Building & releasing ShaderBox

For maintainers. The shipped bundle is a **source distribution** (the `shaderbox/` package +
`uv.lock`; the user's machine runs `uv sync` + `uv run` via the launcher on first launch), not a
frozen binary. This file is never shipped in the bundle.

## Cut a release

```
make release VERSION=x.y.z   # bumps pyproject.toml, commits, tags v<x.y.z>
git push && git push --tags  # commits + tag (plain `git push` does NOT push tags)
./build.sh                   # gated: runs make check + make smoke, refuses a dirty tree
yes | ./upload-itch.sh       # butler push to windows/linux channels (needs itch-config + butler login)
```
Then sync the store page (`ai_docs/dev_flow.md ### Sync the itch.io page` — agent-driven Playwright,
done last so the page describes what's downloadable).

`make release` does NOT build or push — those stay separate so a tag can be cut without an
immediate upload. Semver bump policy is in `ai_docs/conventions.md ## Design decisions`.

`build.sh --allow-dirty` skips the clean-tree guard (the check/smoke gate still runs).
`upload-itch.sh` has an interactive y/N confirm — `yes |` pipes past it for an unattended run.

## Building on / for Windows

The default `build.sh` produces **both** platform archives **from Linux** — the bundle is pure
Python source, so no Windows toolchain is needed to *build* it. What can't be proven on Linux is
that the app *runs* on Windows (real GL context, glfw, ffmpeg path, file dialogs).

CI (`.github/workflows/ci.yml`) runs `uv sync` on `windows-latest` as a hard gate (catches a
dependency that won't install on Windows). The headless smoke test there is soft — GitHub's Windows
runners may lack a usable GL context — so **runtime is verified by hand on a real Windows box**:

1. Clone the repo.
2. `uv sync`
3. Run `scripts\run.bat` (or `uv run python ./shaderbox/ui.py`).
4. Exercise the manual checklist: the starter "UV Mango" shader renders; edit + Ctrl+S hot-reloads;
   a uniform slider drives the image; New (Ctrl+N) creates a node from a template; export to image
   and video produces files; "Open dir" opens the node folder in Explorer.
