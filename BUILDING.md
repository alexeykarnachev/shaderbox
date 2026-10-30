# Building & releasing ShaderBox

For maintainers. The shipped bundle is a **source distribution** (the `shaderbox/` package +
`uv.lock`; the user's machine runs `uv sync` + `uv run` via the launcher on first launch), not a
frozen binary. This file is never shipped in the bundle.

## Cut a release

The full release procedure (sanitize → commit/push `dev` → promote to `master` → auto-pick the semver
bump → `make release` → gated `build.sh` → butler upload → page sync → land clean on `dev`) is the
**`/ship` skill** (`.claude/skills/ship/SKILL.md`) — the single canonical home for the flow. Invoke it
with "ship" / "release" / "publish" / `/ship`.

The individual tools it drives: `make release VERSION=x.y.z` (bumps `pyproject.toml`, commits, tags —
does NOT build/push), `./build.sh` (gated: `make check` + `make smoke`, refuses a dirty tree;
`--allow-dirty` skips only the dirty-tree guard), `yes | ./upload-itch.sh` (butler push to both
channels; needs `itch-config` + `butler login`). Semver bump policy: `ai_docs/conventions.md ##
Design decisions`.

## Building on / for Windows

`build.sh` produces **both** platform archives on either host: the bundle is Python source plus
the two vendored native libraries, and both platform builds of each are committed side by side
(`libeditor.{so,dll}`, `libgraph_canvas.{so,dll}`). Each archive keeps its own and the build aborts
if one is missing. The zipper is `scripts/zip_bundle.py`, so no `zip` binary is needed.

On Windows everything runs from **Git Bash** with GNU `make` on `PATH` (`winget install
ezwinports.make`): `uv sync`, `make gates`, `./build.sh`, `make run-bundle`. A bare `uv sync`
installs every extra through the default `app` dependency group.

The `.dll`s can only be built on Windows (Odin does not cross-link). With Odin on `PATH`, in the
library's repo at the vendored sha:

```
odin build ffi -build-mode:shared -out:libeditor.dll          # editor repo
odin build ffi -build-mode:shared -out:libgraph_canvas.dll    # graph_canvas repo
```

No `-no-entry-point` on Windows: it drops the C runtime and the link fails on `memset`/`memcpy`.
`tests/test_vendored_binaries.py` fails on either host when a `.dll` exports a different API from
its `.so`.

CI runs `uv sync` and the headless smoke test on `windows-latest`; the smoke step is soft because
the hosted runner may have no GL context. On a real Windows box, check the unzipped bundle by hand:
`make run-bundle` builds, unzips and launches it with a throwaway data dir. Checklist: the starter
"UV Mango" shader renders; edit + Ctrl+S hot-reloads; a uniform slider drives the image; New
document (Ctrl+Shift+N) creates one from a template; export to image and video produces files;
"Open folder" opens the document's folder in Explorer.
