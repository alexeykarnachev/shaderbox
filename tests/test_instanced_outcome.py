"""Who decides what a pass's draw did, and where it lands (102 D4, D4a, D4b, D6, I5).

`InstancedOutcome` has three producers (`instanced_outcome.py`'s own docstring): `Pass.render`
/ `_upload_instances` for `drew` / `empty` / `fullscreen` / `refused`; `Pass.compile` for
`compile_failed` / `stale_program`; the script engine for `no_fields`, which is NOT this flow's
to test (`scripting/engine.py` belongs to another flow). Each gate here is scoped to exactly
the producer this flow owns, per the spec's own warning against a gate on `render`'s return
claiming all eight states -- that gate could see at most five and would narrow its own domain.
"""

import dataclasses
import subprocess
from pathlib import Path

import moderngl
import numpy as np
import pytest

from shaderbox.core import Pass
from shaderbox.document import Document
from shaderbox.instanced_outcome import InstancedOutcome, InstancedState

_ENTITY_FRAGMENT = """#version 460 core
in vec2 vs_quad;
flat in vec2  pos;
flat in float radius;
out vec4 frag_color;
void main(){
    if (length(vs_quad) > 1.0) discard;
    frag_color = vec4(1.0);
}
"""

_FULLSCREEN_FRAGMENT = """#version 460 core
in vec2 vs_uv;
out vec4 frag_color;
void main(){ frag_color = vec4(1.0); }
"""

_BROKEN_FRAGMENT = """#version 460 core
out vec4 frag_color;
void main(){ frag_color = nonsense_symbol; }
"""


def _entity_pass(gl: moderngl.Context) -> Pass:
    render_pass = Pass(gl=gl, canvas_size=(16, 16))
    render_pass.release_program(_ENTITY_FRAGMENT)
    return render_pass


def _population(n: int) -> dict[str, np.ndarray]:
    return {
        "pos": np.zeros((n, 2), dtype="f4"),
        "radius": np.full(n, 0.2, dtype="f4"),
    }


# --- `Pass.render` / `_upload_instances`: drew, empty, fullscreen, refused -----------------


def test_a_populated_draw_reports_drew_with_the_count() -> None:
    """Break: stop setting `last_outcome` on the draw path and this reads `not_compiled`
    (the constructor default) forever, which is exactly I2/I3's silence returning."""
    render_pass = _entity_pass(moderngl.create_standalone_context())
    render_pass.compile()
    render_pass.render(u_time=0.0, instances=_population(3))
    assert render_pass.last_outcome.state == "drew"
    assert render_pass.last_outcome.count == 3


def test_zero_entities_reports_empty_not_fullscreen() -> None:
    """I3: zero entities clears the canvas today with no signal. A population that IS
    present but has zero rows must report `empty`, never `fullscreen` -- the two look the
    same on an unconditional black clear but mean opposite things about the script."""
    render_pass = _entity_pass(moderngl.create_standalone_context())
    render_pass.compile()
    render_pass.render(u_time=0.0, instances=_population(0))
    assert render_pass.last_outcome.state == "empty"
    assert render_pass.last_outcome.count == 0


def test_no_population_reports_fullscreen() -> None:
    """I2: an instanced pass whose script sent nothing this frame draws fullscreen with no
    signal today. `pending_instances` defaults to `{}`, which is exactly this case."""
    render_pass = _entity_pass(moderngl.create_standalone_context())
    render_pass.compile()
    render_pass.render(u_time=0.0, instances={})
    assert render_pass.last_outcome.state == "fullscreen"
    assert render_pass.last_outcome.count is None


def test_empty_and_fullscreen_are_the_same_picture_but_different_states() -> None:
    """The pair the spec names as worth the most care: both leave the canvas cleared with
    nothing drawn (an unconditional black clear precedes both paths), so pixels alone
    cannot tell them apart -- the outcome TYPE is the only thing that does, which is the
    entire reason 102 D4 exists rather than a render-time log line."""
    gl = moderngl.create_standalone_context()
    empty_pass = _entity_pass(gl)
    empty_pass.compile()
    empty_pass.render(u_time=0.0, instances=_population(0))

    fullscreen_pass = _entity_pass(gl)
    fullscreen_pass.compile()
    fullscreen_pass.render(u_time=0.0, instances={})

    empty_pixels = np.frombuffer(
        empty_pass.canvas.fbo.read(components=4, dtype="f2"), dtype="f2"
    )
    fullscreen_pixels = np.frombuffer(
        fullscreen_pass.canvas.fbo.read(components=4, dtype="f2"), dtype="f2"
    )
    assert float(empty_pixels.max()) == 0.0, "an empty population must draw nothing"
    assert float(fullscreen_pixels.max()) > 0.0, "no population must draw the quad"
    assert empty_pass.last_outcome.state != fullscreen_pass.last_outcome.state
    assert empty_pass.last_outcome.describe() != fullscreen_pass.last_outcome.describe()


def test_a_validation_failure_reports_refused_with_the_reason() -> None:
    """A population that does not match what the shader declares. Break: drop the `detail=`
    kwarg and this still reports `refused` but with an empty reason -- the assertion on
    `detail` catches that, not just the state."""
    render_pass = _entity_pass(moderngl.create_standalone_context())
    render_pass.compile()
    render_pass.render(
        u_time=0.0,
        instances={
            "pos": np.zeros((4, 2), dtype="f4"),
            "radius": np.zeros(4, dtype="f8"),  # wrong dtype
        },
    )
    assert render_pass.last_outcome.state == "refused"
    assert "radius" in render_pass.last_outcome.detail


def test_the_engine_refused_sentinel_also_reports_refused() -> None:
    """The OTHER refused path: `Pass._upload_instances` receiving the engine's
    `REFUSED_POPULATION` sentinel directly, distinct from the pass's own validation
    failure above. Both land on the same STATE (the research's I2/I3 separation is about
    (b) vs (e), not about the two refusal causes), each with its own `detail`."""
    from shaderbox.scripting.keys import REFUSED_POPULATION

    render_pass = _entity_pass(moderngl.create_standalone_context())
    render_pass.compile()
    render_pass.render(u_time=0.0, instances=REFUSED_POPULATION)
    assert render_pass.last_outcome.state == "refused"
    assert render_pass.last_outcome.detail


# --- `Pass.compile`: compile_failed, stale_program (I5) -------------------------------------


def test_a_first_ever_compile_failure_reports_compile_failed() -> None:
    """No program ever existed, so there is nothing stale to draw -- `compile_failed`, not
    `stale_program`. Break: swap the ternary in `_fail_compile` and this reads
    `stale_program` on a pass that has never once compiled."""
    render_pass = Pass(gl=moderngl.create_standalone_context(), canvas_size=(16, 16))
    render_pass.release_program(_BROKEN_FRAGMENT)
    render_pass.compile()
    assert render_pass.program is None
    assert render_pass.last_outcome.state == "compile_failed"


def test_a_failed_recompile_reports_stale_program_and_still_draws_the_old_one() -> None:
    """I5: a failed recompile keeps the old fields and the old program (`compile()` never
    touches `self.program` until it has a NEW one to swap in), so the pass draws the
    PREVIOUS shader under the new error. Every production caller invalidates first
    (`release_program`), which nulls `self.program` before the retry -- so replicating the
    research's measurement means mutating `.source` directly, the same way `watch.py`'s
    lib-reload path and a future caller could, and calling `compile()` again in place.
    Pins both halves: the STATE, and that the stale picture really does keep rendering --
    which is the decision this wave takes on the stale-FIELDS half: the old program (and
    therefore its fields) is left exactly as it was, only the OUTCOME becomes visible.
    Break: after a recompile failure, either report `compile_failed` (losing the
    distinction from a first-ever failure) or drop the still-valid program (which would
    turn the canvas black instead of stale) and this fails."""
    gl = moderngl.create_standalone_context()
    render_pass = Pass(gl=gl, canvas_size=(16, 16))
    render_pass.release_program(_FULLSCREEN_FRAGMENT)
    render_pass.compile()
    assert render_pass.program is not None, "the premise: a good compile happened first"
    render_pass.render(u_time=0.0)
    good_pixels = np.frombuffer(
        render_pass.canvas.fbo.read(components=4, dtype="f2"), dtype="f2"
    )
    assert float(good_pixels.max()) > 0.0, "the valid program actually drew"

    render_pass.source = dataclasses.replace(render_pass.source, text=_BROKEN_FRAGMENT)
    render_pass.compile()
    assert render_pass.program is not None, "the OLD program must survive the failure"
    assert render_pass.last_outcome.state == "stale_program"

    render_pass.canvas.fbo.use()
    gl.clear()
    render_pass.render(u_time=0.0)
    stale_pixels = np.frombuffer(
        render_pass.canvas.fbo.read(components=4, dtype="f2"), dtype="f2"
    )
    assert float(stale_pixels.max()) > 0.0, "the stale program must still draw"
    assert render_pass.last_outcome.state == "stale_program", (
        "render() must not let the still-successful draw overwrite compile's verdict"
    )


def test_a_pass_that_never_rendered_reports_not_compiled() -> None:
    """Frame one of every document: nothing has attempted a compile yet. Break: change the
    constructor default and every fresh document reports something other than the one
    healthy state that means 'nothing has happened yet'."""
    render_pass = Pass(gl=moderngl.create_standalone_context(), canvas_size=(16, 16))
    assert render_pass.last_outcome.state == "not_compiled"


# --- D6: `_instances_error` has no reader after deletion -------------------------------------


def test_instances_error_has_no_reader_left_in_the_repo() -> None:
    """Grep-based per the spec's own gate recipe (D6). Scoped to the attribute-access
    SPELLING (`self._instances_error` / `.​_instances_error`) rather than the bare word, so
    `instanced_outcome.py`'s own docstring -- which legitimately narrates I4's history as
    the reason this type exists -- does not read as a false positive. Break: reintroduce
    `self._instances_error = ...` or a read of it anywhere under `shaderbox/` and this
    fails."""
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["grep", "-rn", r"\._instances_error", str(repo_root / "shaderbox")],
        capture_output=True,
        text=True,
    )
    assert result.stdout == "", f"a reader survived deletion: {result.stdout}"


# --- D4a / D4b: `Document.instanced_outcomes` and per-iteration aggregation -----------------


def _document(
    tmp_path: Path, gl_ctx: moderngl.Context, iterations: int = 1
) -> Document:
    (tmp_path / "passes").mkdir()
    (tmp_path / "passes" / "main.frag.glsl").write_text(_FULLSCREEN_FRAGMENT)
    (tmp_path / "graph.json").write_text(
        '{"version": 2, "output": "main", "passes": '
        f'{{"main": {{"iterations": {iterations}}}}}}}'
    )
    (tmp_path / "document.json").write_text('{"uniforms": {}}')
    document, _ = Document.load_from_dir(tmp_path, gl=gl_ctx, canvas_size=(16, 16))
    return document


def test_instanced_outcomes_carries_the_real_pass_name(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    """`Pass` itself does not know its name (it is keyed by the Document's dict), so
    `last_outcome.pass_name` is empty at the source -- `Document` must rewrap it. Break:
    store `render_pass.last_outcome` directly into the dict and this reads ''."""
    document = _document(tmp_path, gl_ctx)
    document.render()
    assert "main" in document.instanced_outcomes
    assert document.instanced_outcomes["main"].pass_name == "main"


def test_an_iterated_pass_reports_only_its_last_iteration(
    gl_ctx: moderngl.Context,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """102 D4b: this wave's decision is LAST WINS -- every iteration draws into the same
    canvas (the chain advances by swapping BETWEEN iterations, not per-frame), so the last
    iteration is what the canvas a reader would see actually shows. Proven here by forcing
    each of three iterations to a DIFFERENT, distinguishable state and asserting only the
    third survives into `Document.instanced_outcomes` -- a fixture that reported iteration
    0 or 'all of them' would fail this, not just one that silently passed through."""
    document = _document(tmp_path, gl_ctx, iterations=3)
    seen_states: list[InstancedState] = ["drew", "empty", "fullscreen"]
    calls: list[int] = []
    original_render = Pass.render

    def _fake_render(self: Pass, **kwargs: object) -> None:
        original_render(self, **kwargs)
        index = len(calls)
        calls.append(index)
        state = seen_states[index]
        self.last_outcome = InstancedOutcome(
            "", state, count=None if state == "fullscreen" else index
        )

    monkeypatch.setattr(Pass, "render", _fake_render)
    document.render()

    assert len(calls) == 3, "the premise: all three iterations actually ran"
    outcome = document.instanced_outcomes["main"]
    assert outcome.state == "fullscreen", (
        f"expected the LAST iteration's state to win, got {outcome.state}"
    )
