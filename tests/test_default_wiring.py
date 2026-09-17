"""An input uniform's NAME is its default wire, resolved at render time (069 W-D, D9; 072).

A sampler called `u_<pass>` reads that pass without anyone choosing it; `u_prev` reads the
pass's own previous frame; an explicit `NoSource` means BLACK and survives a reload, because the
rule must not undo a decision the user made. The resolution is one pure function (`wired_pass`)
and one method built on it (`Document.effective_wiring`), and every consumer reads the same
wiring -- so "the renderer draws it" and "the strip says it is live" cannot disagree.

The two GL tests that build a document do it on a COPY of a shipped example, never in place.
"""

import shutil
from pathlib import Path
from typing import Any

import moderngl
import numpy as np
import pytest
from imgui_bundle import imgui

from shaderbox.media import texture_to_rgba8
from shaderbox.pass_graph import (
    AutoSource,
    NoSource,
    PassSource,
    plan_for_output,
    plan_passes,
    wired_pass,
)
from shaderbox.paths import shader_lib_root
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.ui_models import UIUniform, load_document_from_dir
from shaderbox.widgets import uniform as uniform_widget

_EXAMPLES = (
    Path(__file__).resolve().parent.parent / "shaderbox/resources/document_examples"
)
_RC = _EXAMPLES / "77a84d27-2e5b-406d-8011-ee1cb1a9587c"
_BLOOM = (
    Path(__file__).parent / "fixtures" / "bloom_chain"
)  # a test fixture, not shipped
_MEDIA_INPUT = _EXAMPLES / "73ea2431-13f6-41e4-b923-04d846b678b0"

_TRAIL = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_prev;
out vec4 fs_color;
void main() { fs_color = texture(u_prev, vs_uv) * 0.9 + vec4(0.1); }
"""

_EDGE = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_df;
out vec4 fs_color;
void main() { fs_color = vec4(texture(u_df, vs_uv).rgb, 1.0); }
"""


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    context = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(shader_lib_root()))
    return context


# ---------------------------------------------------------------- the pure resolution


def test_the_planner_orders_an_auto_edge() -> None:
    # The planner must SEE the auto edges or it cannot order the draw, and it cannot detect a
    # cycle a name default creates. `wired_pass` is where a name becomes an edge.
    names = {"a", "b"}

    def reads(consumer: str, samplers: list[str]) -> dict[str, str]:
        wired = {u: wired_pass(AutoSource(), u, consumer, names) for u in samplers}
        return {u: p for u, p in wired.items() if p is not None}

    wiring = {"a": reads("a", []), "b": reads("b", ["u_a"])}
    order, errors = plan_for_output(wiring, "b")
    assert order == ["a", "b"]
    assert errors == []
    assert plan_passes(wiring)[0].reads["b"] == {"a"}

    # A name default that closes a loop is a real cycle and is reported as one.
    looped = {"a": reads("a", ["u_b"]), "b": reads("b", ["u_a"])}
    cycle_errors = plan_passes(looped)[1]
    assert {e.pass_name for e in cycle_errors} == {"a", "b"}
    assert any("cycle" in e.message for e in cycle_errors)


# ---------------------------------------------------------------- the render path


def _rc_with_edge(tmp_path: Path) -> Any:
    """A copy of the Radiance Cascades example plus an `edge` pass declaring `u_df`.

    The pass file is written directly and `document.json` is left alone: no row decides
    `u_df`, so only the name rule can fill it.
    """
    document_dir = tmp_path / "document"
    shutil.copytree(_RC, document_dir)
    (document_dir / "passes" / "edge.frag.glsl").write_text(_EDGE, encoding="utf-8")
    return load_document_from_dir(document_dir)


def _render_until_online(document: Any, frames: int) -> None:
    # One never-rendered pass per frame, the way the live loop's sweep does it.
    for frame in range(frames):
        document.begin_frame(frame)
        document.render(u_time=frame / 30.0)
        pending = next(
            (
                name
                for name, render_pass in document.passes.items()
                if not render_pass.first_render_done
            ),
            None,
        )
        if pending is not None:
            document.render(target=pending)


def test_u_df_beside_df_renders_without_the_gear(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    ui_document = _rc_with_edge(tmp_path)
    document = ui_document.document
    document.graph = document.graph.with_output("edge")
    assert "u_df" not in document.passes["edge"].uniform_values

    _render_until_online(document, len(document.passes) + 4)

    assert document.effective_wiring()["edge"] == {"u_df": "df"}
    # The PICTURE, not merely a non-black one: `edge` passes `df`'s texel straight through, so
    # the two canvases must agree. Asserting brightness alone cannot tell the distance field from
    # the seeded default PHOTO an unbound sampler used to fall through to -- both are non-black.
    drawn = np.asarray(texture_to_rgba8(document.passes["edge"].canvas.texture))[
        :, :, :3
    ]
    produced = np.asarray(texture_to_rgba8(document.passes["df"].canvas.texture))[
        :, :, :3
    ]
    assert drawn.shape == produced.shape
    assert int(np.abs(drawn.astype(int) - produced.astype(int)).max()) <= 1, (
        "`edge` did not render what `df` produced"
    )
    assert int(drawn.max()) > 0, "the distance field itself came out black"
    # The rule stores nothing: the sampler's value is still undecided.
    assert isinstance(document.passes["edge"].uniform_values["u_df"], AutoSource)


def test_an_unresolved_sampler_renders_black(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # 065 D3: an input the graph does not fill reads BLACK. Left unbound the sampler falls
    # through to its own seeded default photo, which is what makes the gear's `auto: none` and
    # the copilot's `reads BLACK` false about the same sampler in the same frame.
    document_dir = tmp_path / "document"
    shutil.copytree(_RC, document_dir)
    (document_dir / "passes" / "edge.frag.glsl").write_text(
        _EDGE.replace("u_df", "u_nosuchpass"), encoding="utf-8"
    )
    document = load_document_from_dir(document_dir).document
    document.graph = document.graph.with_output("edge")

    # Frame-indexed, and frame 0 is the one that matters: the seed reads the pass's program to
    # learn what it declares, so a seed built before the compile is empty and lets the very
    # first frame through to the photo. Asserting only the settled state cannot see that.
    for frame in range(len(document.passes) + 4):
        document.begin_frame(frame)
        document.render(u_time=frame / 30.0)
        drawn = np.asarray(texture_to_rgba8(document.passes["edge"].canvas.texture))[
            :, :, :3
        ]
        assert int(drawn.max()) == 0, (
            f"frame {frame}: an unresolved sampler rendered the seeded default image"
        )

    assert document.effective_wiring()["edge"] == {}


def test_a_user_bound_texture_survives_the_black_seed(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # `Pass.render` resolves `inputs.get(name, uniform_values.get(name))`, so the seed SHADOWS
    # what the user bound -- and the effective graph makes no edge for a user-bound sampler by
    # design, so nothing would overwrite it. The shipped Media Input example is the document
    # whose whole subject is that case, and it must render its PNG and video, not black.
    document_dir = tmp_path / "document"
    shutil.copytree(_MEDIA_INPUT, document_dir)
    document = load_document_from_dir(document_dir).document
    _render_until_online(document, len(document.passes) + 4)

    drawn = np.asarray(texture_to_rgba8(document.render_pass.canvas.texture))[:, :, :3]
    assert int(drawn.max()) > 0, "the user's bound media was replaced by the black seed"
    # A real picture, not one bright texel: the seed's failure mode is uniform black, and a
    # color count separates the bound image from any flat fill.
    assert len(np.unique(drawn.reshape(-1, 3), axis=0)) > 1000, (
        "the output is a flat fill rather than the bound image"
    )


def test_an_explicit_none_stays_black_across_a_reload(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    ui_document = _rc_with_edge(tmp_path)
    document = ui_document.document
    document.graph = document.graph.with_output("edge")
    document.passes["edge"].uniform_values["u_df"] = NoSource()

    ui_document.save(tmp_path, dir_name="document")
    reloaded = load_document_from_dir(tmp_path / "document").document
    # Through disk on purpose: an in-memory assertion would pass under a save that wrote no row
    # for the decision and a load that seeded it undecided.
    assert isinstance(reloaded.passes["edge"].uniform_values["u_df"], NoSource)
    assert reloaded.effective_wiring()["edge"] == {}

    _render_until_online(reloaded, len(reloaded.passes) + 4)
    image = texture_to_rgba8(reloaded.passes["edge"].canvas.texture)
    assert int(np.asarray(image)[:, :, :3].max()) == 0, (
        "an explicit none was re-wired by the name rule"
    )


def test_an_uncompiled_pass_contributes_no_auto_edge_and_compiles_nothing(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    document_dir = tmp_path / "document"
    shutil.copytree(_BLOOM, document_dir)
    document = load_document_from_dir(document_dir).document
    assert all(p.program is None for p in document.passes.values())

    # The fixture stores no row (every edge is the name rule's), and the name rule waits for a
    # program: nothing reads anything yet.
    assert document.effective_wiring() == {name: {} for name in document.passes}
    # (b): asking the program is not asking `get_active_uniforms()`, which would compile the
    # whole document on frame one and invert 066 D1.
    assert all(p.program is None for p in document.passes.values())

    # An explicit row is visible BEFORE the compile (072 D3): a wire is never lost to lazy
    # compilation, only the name rule waits.
    document.passes["composite"].uniform_values["u_extra"] = PassSource("scene")
    assert document.effective_wiring()["composite"] == {"u_extra": "scene"}
    assert all(p.program is None for p in document.passes.values())
    del document.passes["composite"].uniform_values["u_extra"]

    _render_until_online(document, len(document.passes) + 2)
    online = document.effective_wiring()
    assert online["composite"]["u_blur"] == "blur"


# ---------------------------------------------------------------- the panel row


def _combo_capture(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, list[str]]]:
    # The capture flattens the combo's rows back to the (selected index, labels) reading the
    # states below are asserted on.
    seen: list[tuple[int, list[str]]] = []
    real = uniform_widget.grouped_combo

    def spy(id_: str, current: Any, groups: Any, width: float) -> int | None:
        if id_.startswith("##source_"):
            labels = [label for _, rows in groups for label, _ in rows]
            seen.append((labels.index(current[0]), labels))
        return real(id_, current, groups, width)

    monkeypatch.setattr(uniform_widget, "grouped_combo", spy)
    return seen


def _frame(body: Any) -> None:
    imgui.new_frame()
    imgui.begin("rig")
    body()
    imgui.end()
    imgui.end_frame()


def test_the_row_shows_three_distinct_states(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The sampler's row on the uniforms panel is where its source is chosen (072 D7). 079 D6
    # flattened the list to `none` / the passes / `file...`: `AutoSource` is still the VALUE a
    # fresh sampler holds, but the control shows what it RESOLVES to, so the row reads as the
    # pass it reads rather than as a rule the user has to decode.
    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    app.session.rename_pass(document_id, next(iter(document.passes)), "df")
    app.session.add_pass(document_id, "edge")
    document.passes["edge"].release_program(_EDGE)
    document.passes["edge"].compile()
    document.passes["edge"].seed_uniform_values()
    # The panel shows the active shader tab's pass.
    app.ensure_shader_tab(document_id, "edge")
    ui_uniform = next(
        UIUniform.from_uniform(u)
        for u in document.passes["edge"].get_active_uniforms()
        if u.name == "u_df"
    )
    seen = _combo_capture(monkeypatch)

    def draw() -> None:
        uniform_widget.draw_ui_uniform(app, ui_uniform)

    # Undecided: shown as the pass the name rule wires it to, with no rule row in the list.
    _frame(draw)
    assert seen[-1] == (1, ["none", "df", "edge", "file..."])

    # An explicit none selects `none`.
    app.session.set_sampler_source(document_id, "edge", "u_df", NoSource())
    _frame(draw)
    assert seen[-1][0] == 0

    # An explicit pass selects that pass.
    app.session.set_sampler_source(document_id, "edge", "u_df", PassSource("edge"))
    _frame(draw)
    assert seen[-1][0] == seen[-1][1].index("edge")

    # Undecided and resolving to nothing reads as `none`, with `df` renamed away.
    app.session.set_sampler_source(document_id, "edge", "u_df", AutoSource())
    app.session.rename_pass(document_id, "df", "field")
    _frame(draw)
    assert seen[-1] == (0, ["none", "edge", "field", "file..."])


# ---------------------------------------------------------------- the shipped examples


@pytest.mark.parametrize("example", [_RC, _BLOOM], ids=["radiance_cascades", "bloom"])
def test_every_multi_pass_example_compiles_every_pass(
    gl: moderngl.Context, tmp_path: Path, example: Path
) -> None:
    # A declaration renamed without its `texture()` read links nowhere -- and nothing else in the
    # suite sees that on Bloom, since the first-render stamps are written on ATTEMPT.
    document_dir = tmp_path / example.name
    shutil.copytree(example, document_dir)
    document = load_document_from_dir(document_dir).document
    _render_until_online(document, len(document.passes) + 2)
    broken = {
        name: render_pass.compile_unit.errors
        for name, render_pass in document.passes.items()
        if render_pass.compile_unit.errors
    }
    assert broken == {}, f"{example.name}: passes failed to compile: {broken}"


def test_a_u_prev_pass_has_feedback_without_a_stored_row(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # The plan of the EFFECTIVE wiring must see the auto edge too: a pass declaring `u_prev` and
    # nothing else reads its own previous frame, and the renderer allocates its history from
    # that plan. Bloom's `trail` is the only feedback in the fixture; it is rewired to a fresh
    # `u_prev`-only pass so the name rule is the only thing that can wire it.
    document_dir = tmp_path / "document"
    shutil.copytree(_BLOOM, document_dir)
    (document_dir / "passes" / "trail.frag.glsl").write_text(_TRAIL, encoding="utf-8")
    document = load_document_from_dir(document_dir).document
    assert "u_prev" not in document.passes["trail"].uniform_values
    assert not plan_passes(document.effective_wiring())[0].feedback, (
        "nothing has compiled yet, so no auto edge exists"
    )

    _render_until_online(document, len(document.passes) + 4)
    assert isinstance(document.passes["trail"].uniform_values["u_prev"], AutoSource)
    assert plan_passes(document.effective_wiring())[0].feedback == {"trail"}
