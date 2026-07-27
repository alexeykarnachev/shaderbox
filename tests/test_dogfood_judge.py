"""Tests for the dogfood judge primitives + the package's no-side-effect import (feature 057).

Every primitive is exercised on a SYNTHETIC image whose answer is known by construction, so a
number drifting is a test failure rather than a report the human silently misreads.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

import scripts.dogfood
from scripts.dogfood.judge import (
    bright_centroid,
    color_mask_centroid,
    column_runs,
    farthest_bright_angle,
    grid_cell,
    load_rgb,
    region_diff,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _blank(w: int = 100, h: int = 100) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.int16)


def test_load_rgb_reads_shape_and_dtype(tmp_path: Path) -> None:
    path = tmp_path / "img.png"
    PILImage.new("RGBA", (7, 5), (10, 20, 30, 255)).save(path)
    im = load_rgb(path)
    assert im.shape == (5, 7, 3)
    assert im.dtype == np.int16
    assert tuple(int(v) for v in im[0, 0]) == (10, 20, 30)


def test_grid_cell_splits_evenly_and_is_a_view() -> None:
    im = _blank(90, 60)
    im[0:20, 30:60] = 255  # row 0, col 1 of a 3x3 grid
    cell = grid_cell(im, 0, 1, 3, 3)
    assert cell.shape == (20, 30, 3)
    assert int(cell.min()) == 255
    assert int(grid_cell(im, 0, 0, 3, 3).max()) == 0


def test_region_diff_is_mean_abs_per_channel() -> None:
    a = _blank(10, 10)
    b = _blank(10, 10)
    b[:, :, 0] = 30  # one channel of three shifted by 30 -> mean 10
    assert region_diff(a, b) == 10.0
    assert region_diff(a, a) == 0.0


def test_bright_centroid_finds_the_blob_and_returns_none_when_dark() -> None:
    im = _blank()
    im[40:51, 20:31] = 200
    centroid = bright_centroid(im)
    assert centroid is not None
    assert centroid == (25.0, 45.0)
    # A saturated RED blob is bright too — the max-channel rule, not luminance.
    red = _blank()
    red[0:5, 0:5, 0] = 255
    assert bright_centroid(red) == (2.0, 2.0)
    assert bright_centroid(_blank()) is None


def test_color_mask_centroid_selects_only_the_named_color() -> None:
    im = _blank()
    im[10:21, 10:21] = (0, 0, 255)  # blue block, centroid (15, 15)
    im[60:71, 60:71] = (255, 0, 0)  # red block, must be ignored
    assert color_mask_centroid(im, (0, 0, 200), (60, 60, 255)) == (15.0, 15.0)
    assert color_mask_centroid(im, (0, 200, 0), (60, 255, 60)) is None


def test_column_runs_counts_separated_blobs() -> None:
    im = _blank()
    im[40:60, 10:21] = 255
    im[40:60, 50:61] = 255
    assert column_runs(im) == [(10, 20), (50, 60)]
    assert column_runs(_blank()) == []


def test_farthest_bright_angle_reads_the_screen_direction() -> None:
    # y grows DOWNWARD, so a blob below-right of center is a POSITIVE angle near +45.
    im = _blank(101, 101)
    im[90:96, 90:96] = 255
    angle = farthest_bright_angle(im)
    assert angle is not None and abs(angle - 45.0) < 2.0
    up_right = _blank(101, 101)
    up_right[5:11, 90:96] = 255
    other = farthest_bright_angle(up_right)
    assert other is not None and abs(other + 45.0) < 2.0
    assert farthest_bright_angle(_blank()) is None


def _sanitized_env(data_dir: Path | None) -> dict[str, str]:
    # Never let a child process see the live OpenRouter key: the harness's env block writes it into
    # whatever SHADERBOX_DATA_DIR points at, and a test must not leave creds anywhere.
    env = dict(os.environ)
    env.pop("OPENROUTER_API_KEY", None)
    if data_dir is not None:
        data_dir.mkdir(parents=True, exist_ok=True)
        env["SHADERBOX_DATA_DIR"] = str(data_dir)
    return env


def _run_child(code: str, env: dict[str, str]) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"child failed ({proc.returncode}):\n{proc.stderr}"
    return proc.stdout.strip()


def test_importing_analyze_and_judge_has_no_gl_side_effects() -> None:
    # The package init used to import `harness` EAGERLY, so ANY `scripts.dogfood.*` import ran its
    # module-top env block: a mkdtemp'd runs/data-* dir holding a live OpenRouter key, plus
    # glfw/moderngl. The lazy __getattr__ kills that; this pins it (D4).
    runs = _REPO_ROOT / "scripts" / "dogfood" / "runs"
    before = {p.name for p in runs.glob("data-*")} if runs.is_dir() else set()
    loaded = _run_child(
        "import sys\n"
        "import scripts.dogfood.analyze\n"
        "import scripts.dogfood.judge\n"
        "print(','.join(m for m in ('glfw', 'moderngl') if m in sys.modules))",
        _sanitized_env(None),
    )
    assert loaded == ""
    after = {p.name for p in runs.glob("data-*")} if runs.is_dir() else set()
    assert after == before


def test_public_harness_import_still_resolves_through_the_lazy_hook(
    tmp_path: Path,
) -> None:
    # `from scripts.dogfood import DogfoodHarness` is the documented entry point; the lazy hook must
    # keep it working verbatim. Out-of-process (it DOES pull in GL) with SHADERBOX_DATA_DIR pointed
    # at tmp_path so the harness's env block adopts it instead of mkdtemp'ing under runs/.
    out = _run_child(
        "from scripts.dogfood import DogfoodHarness\nprint(DogfoodHarness.__name__)",
        _sanitized_env(tmp_path / "data"),
    )
    assert out == "DogfoodHarness"


def test_lazy_hook_rejects_an_unknown_name() -> None:
    with pytest.raises(AttributeError):
        _ = scripts.dogfood.NotAThing
