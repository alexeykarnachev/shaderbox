# Building & releasing ShaderBox

For maintainers. The shipped bundle is a **source distribution** (the `shaderbox/` package +
`uv.lock`; the user's machine runs `uv sync` + `uv run` via the launcher on first launch), not a
frozen binary. This file is never shipped in the bundle.

## Cut a release

1. Commit and push `dev`, then promote it to `master`.
2. `make release VERSION=x.y.z` bumps `pyproject.toml`, commits and tags. It does not build or push.
3. `./build.sh` runs `make check` and `make smoke`, refuses a dirty tree (`--allow-dirty` skips only
   that guard) and writes both archives to `dist/`.
4. `yes | ./upload-itch.sh` pushes both channels with butler; it needs `itch-config` and
   `butler login`.

## Building on / for Windows

`build.sh` produces **both** platform archives on either host: the bundle is Python source plus
the two vendored native libraries, and both platform builds of each are committed side by side
(`libeditor.{so,dll}`, `libgraph_canvas.{so,dll}`). Each archive keeps its own and the build aborts
if one is missing. The zipper is `scripts/zip_bundle.py`, so no `zip` binary is needed.
Files the repo keeps only for tests and vendoring (`DEV_ONLY_RESOURCES` in `build.sh`) are
left out, and the launchers install with `uv sync --no-dev`, so a user gets no test or lint tools.

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
