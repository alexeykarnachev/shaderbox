"""An input uniform's NAME is its default wire, resolved at render time (069 W-D, D9).

A sampler called `u_<pass>` reads that pass without anyone opening the gear; `u_prev` reads the
pass's own previous frame; an explicit `""` in `graph.json` means BLACK and survives a reload,
because the rule must not undo a decision the user made. The resolution is one pure function and
one method built on it, and every consumer reads the same effective graph -- so "the renderer
draws it" and "the strip says it is live" cannot disagree.

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
    PassEntry,
    PassGraph,
    effective_inputs,
    plan_for_output,
    plan_passes,
)
from shaderbox.paths import shader_lib_root
from shaderbox.popups import pass_settings
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.ui_models import load_document_from_dir

_EXAMPLES = (
    Path(__file__).resolve().parent.parent / "shaderbox/resources/document_examples"
)
_RC = _EXAMPLES / "77a84d27-2e5b-406d-8011-ee1cb1a9587c"
_BLOOM = _EXAMPLES / "1c4f8a20-7b6e-4d31-9a55-2f0e6b8c31d4"
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
    ctx = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(shader_lib_root()))
    return ctx


# ---------------------------------------------------------------- the pure resolution


def test_effective_inputs_over_every_state() -> None:
    # The nine cells of (absent, "", explicit) x (the named pass exists or not) x (media-bound),
    # plus `u_prev`. A media-bound sampler never auto-wires, and an explicit answer -- a name or
    # an explicit none -- always beats both the name rule and the bound exclusion.
    cases: list[tuple[dict[str, str], set[str], list[str], dict[str, str]]] = [
        ({}, {"df", "jfa"}, [], {"u_df": "df"}),
        ({}, {"jfa"}, [], {}),
        ({}, {"df", "jfa"}, ["u_df"], {}),
        ({"u_df": ""}, {"df", "jfa"}, [], {}),
        ({"u_df": ""}, {"jfa"}, [], {}),
        ({"u_df": ""}, {"df", "jfa"}, ["u_df"], {}),
        ({"u_df": "jfa"}, {"df", "jfa"}, [], {"u_df": "jfa"}),
        ({"u_df": "jfa"}, {"jfa"}, [], {"u_df": "jfa"}),
        ({"u_df": "jfa"}, {"df", "jfa"}, ["u_df"], {"u_df": "jfa"}),
    ]
    for stored, passes, bound, expected in cases:
        got = effective_inputs(
            PassEntry(inputs=stored), ["u_df"], passes, "edge", bound
        )
        assert got == expected, (stored, sorted(passes), bound)

    # The feedback exception wins over a sibling that happens to be called `prev`: D9 writes
    # `u_prev` down as reading yourself, so that is the branch a user can predict.
    assert effective_inputs(
        PassEntry(), ["u_prev"], {"cascade", "prev"}, "cascade"
    ) == {"u_prev": "cascade"}

    # No `u_` prefix, no auto edge -- D9's rule is about `u_<pass>` names.
    assert effective_inputs(PassEntry(), ["df"], {"df"}, "edge") == {}


def test_the_planner_orders_an_auto_edge() -> None:
    # The planner must SEE the auto edges or it cannot order the draw, and it cannot detect a
    # cycle a name default creates.
    raw = PassGraph(output="b", passes={"a": PassEntry(), "b": PassEntry()})
    resolved = raw.with_passes(
        {
            "a": raw.passes["a"],
            "b": raw.passes["b"].model_copy(
                update={
                    "inputs": effective_inputs(
                        raw.passes["b"], ["u_a"], set(raw.passes), "b"
                    )
                }
            ),
        }
    )
    order, errors = plan_for_output(resolved, "b")
    assert order == ["a", "b"]
    assert errors == []
    assert plan_passes(resolved)[0].reads["b"] == {"a"}

    # A name default that closes a loop is a real cycle and is reported as one.
    looped = raw.with_passes(
        {
            name: entry.model_copy(
                update={
                    "inputs": effective_inputs(
                        entry,
                        ["u_b"] if name == "a" else ["u_a"],
                        set(raw.passes),
                        name,
                    )
                }
            )
            for name, entry in raw.passes.items()
        }
    )
    cycle_errors = plan_passes(looped)[1]
    assert {e.pass_name for e in cycle_errors} == {"a", "b"}
    assert any("cycle" in e.message for e in cycle_errors)


# ---------------------------------------------------------------- the render path


def _rc_with_edge(tmp_path: Path) -> Any:
    """A copy of the Radiance Cascades example plus an `edge` pass declaring `u_df`.

    The pass file is written directly and `graph.json` is left alone: nothing wires `u_df`, so
    only the name rule can fill it.
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
    assert document.graph.passes["edge"].inputs == {}

    _render_until_online(document, len(document.passes) + 4)

    assert document.effective_graph().passes["edge"].inputs == {"u_df": "df"}
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
    # The rule stores nothing: `graph.json` still holds only what the user decided.
    assert document.graph.passes["edge"].inputs == {}


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

    assert document.effective_graph().passes["edge"].inputs == {}


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


def test_a_stored_empty_string_stays_black_across_a_reload(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    ui_document = _rc_with_edge(tmp_path)
    document = ui_document.document
    document.graph = document.graph.with_output("edge").with_input("edge", "u_df", "")
    assert document.graph.passes["edge"].inputs == {"u_df": ""}

    ui_document.save(tmp_path, dir_name="document")
    reloaded = load_document_from_dir(tmp_path / "document").document
    # Through disk on purpose: an in-memory assertion would pass under a `with_input` that
    # stored `""` and a `model_dump` that dropped it.
    assert reloaded.graph.passes["edge"].inputs == {"u_df": ""}
    assert reloaded.effective_graph().passes["edge"].inputs == {}

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

    resolved = document.effective_graph()
    assert {n: e.inputs for n, e in resolved.passes.items()} == {
        n: e.inputs for n, e in document.graph.passes.items()
    }
    # (b): asking the program is not asking `get_active_uniforms()`, which would compile the
    # whole document on frame one and invert 066 D1.
    assert all(p.program is None for p in document.passes.values())

    _render_until_online(document, len(document.passes) + 2)
    online = document.effective_graph()
    assert online.passes["composite"].inputs["u_blur"] == "blur"


# ---------------------------------------------------------------- the gear


def _combo_capture(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, list[str]]]:
    seen: list[tuple[int, list[str]]] = []
    real = imgui.combo

    def spy(label: str, current: int, items: list[str], *a: Any, **kw: Any) -> Any:
        if label.startswith("##wire_"):
            seen.append((current, list(items)))
        return real(label, current, items, *a, **kw)

    monkeypatch.setattr(pass_settings.imgui, "combo", spy)
    return seen


def _frame(body: Any) -> None:
    imgui.new_frame()
    imgui.begin("rig")
    body()
    imgui.end()
    imgui.end_frame()


def test_the_gear_shows_three_distinct_states(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    app.session.rename_pass(document_id, next(iter(document.passes)), "df")
    app.session.add_pass(document_id, "edge")
    document.passes["edge"].release_program(_EDGE)
    document.passes["edge"].compile()
    app.open_pass_settings("edge")
    seen = _combo_capture(monkeypatch)

    # Absent key: the name rule's answer, selected.
    _frame(
        lambda: pass_settings._draw_inputs(
            app, document_id, "edge", document.passes["edge"]
        )
    )
    assert seen[-1] == (0, ["auto: df", "(none)", "df", "edge"])

    # An explicit none is its OWN item, not the same index an absent key falls to.
    app.session.wire_pass_input(document_id, "edge", "u_df", "")
    _frame(
        lambda: pass_settings._draw_inputs(
            app, document_id, "edge", document.passes["edge"]
        )
    )
    assert seen[-1][0] == 1

    # An explicit name selects that pass.
    app.session.wire_pass_input(document_id, "edge", "u_df", "edge")
    _frame(
        lambda: pass_settings._draw_inputs(
            app, document_id, "edge", document.passes["edge"]
        )
    )
    assert seen[-1][0] == seen[-1][1].index("edge")

    # `auto: none` when the name names no pass: back to undecided, with `df` renamed away.
    app.session.unwire_pass_input(document_id, "edge", "u_df")
    app.session.rename_pass(document_id, "df", "field")
    _frame(
        lambda: pass_settings._draw_inputs(
            app, document_id, "edge", document.passes["edge"]
        )
    )
    assert seen[-1] == (0, ["auto: none", "(none)", "edge", "field"])


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


def test_a_u_prev_pass_has_feedback_without_a_stored_edge(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # The plan of the EFFECTIVE graph must see the auto edge too: a pass declaring `u_prev` and
    # nothing else reads its own previous frame, and the renderer allocates its history from
    # that plan.
    # Bloom's `trail` is the only feedback in the example and its edge is EXPLICIT, so it is
    # rewired to a fresh `u_prev`-only pass that stores nothing.
    document_dir = tmp_path / "document"
    shutil.copytree(_BLOOM, document_dir)
    (document_dir / "passes" / "trail.frag.glsl").write_text(_TRAIL, encoding="utf-8")
    document = load_document_from_dir(document_dir).document
    document.graph = document.graph.without_input("trail", "u_prev").without_input(
        "trail", "u_scene"
    )
    assert document.graph.passes["trail"].inputs == {}
    assert not plan_passes(document.effective_graph())[0].feedback, (
        "nothing has compiled yet, so no auto edge exists"
    )

    _render_until_online(document, len(document.passes) + 4)
    assert document.graph.passes["trail"].inputs == {}
    assert plan_passes(document.effective_graph())[0].feedback == {"trail"}
