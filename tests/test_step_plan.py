"""Evaluation order: topological sort WITH memoization, and ping-pong as a non-edge.

Both prior designs in this repo's history got exactly one half. The deleted DAG had correct
pull-recursion order but no memoization, so a diamond re-rendered its shared ancestor once
per consumer. freska had per-node targets but iterated an unordered_map under its own
`// TODO: this is incorrect!`, lagging an N-step chain by N-1 frames nondeterministically.
"""

from pathlib import Path

from shaderbox.step_spec import find_steps, plan_steps

_PATH = Path("node.frag.glsl")


def _plan(source: str):
    parsed = find_steps(source, _PATH)
    assert parsed.errors == [], parsed.errors
    plan, errors = plan_steps(source, parsed.steps, _PATH)
    if not errors:
        # THE memoization invariant, asserted on every plan this module builds rather
        # than in one test: a step appears in the order exactly once, so it draws once
        # per frame. A duplicate re-renders a shared ancestor per consuming path -- the
        # defect the deleted DAG shipped, and one that reads as "slow" rather than
        # "wrong" because the picture stays correct.
        assert len(plan.order) == len(set(plan.order)), plan.order
    return plan, errors


def test_a_linear_chain_orders_producers_first() -> None:
    src = (
        "#version 330\n"
        "uniform sampler2D u_step_a;\n"
        "uniform sampler2D u_step_b;\n"
        "void step_a(out vec4 o) { o = vec4(1.0); }\n"
        "void step_b(out vec4 o) { o = texture(u_step_a, vec2(0.0)); }\n"
        "void main() { gl_FragColor = texture(u_step_b, vec2(0.0)); }\n"
    )
    plan, errors = _plan(src)
    assert errors == []
    assert plan.order == ["a", "b"]
    assert plan.reads["b"] == {"a"}
    assert plan.final_reads == {"b"}


def test_declaration_order_does_not_decide_evaluation_order() -> None:
    # The consumer is declared FIRST; the producer must still run first.
    src = (
        "#version 330\n"
        "uniform sampler2D u_step_b;\n"
        "uniform sampler2D u_step_a;\n"
        "void step_b(out vec4 o) { o = texture(u_step_a, vec2(0.0)); }\n"
        "void step_a(out vec4 o) { o = vec4(1.0); }\n"
        "void main() {}\n"
    )
    plan, errors = _plan(src)
    assert errors == []
    assert plan.order == ["a", "b"]


def test_a_diamond_renders_the_shared_ancestor_once() -> None:
    # The memoization half. `base` feeds both `left` and `right`; it must appear ONCE.
    src = (
        "#version 330\n"
        "uniform sampler2D u_step_base;\n"
        "uniform sampler2D u_step_left;\n"
        "uniform sampler2D u_step_right;\n"
        "uniform sampler2D u_step_merge;\n"
        "void step_base(out vec4 o) { o = vec4(1.0); }\n"
        "void step_left(out vec4 o) { o = texture(u_step_base, vec2(0.0)); }\n"
        "void step_right(out vec4 o) { o = texture(u_step_base, vec2(0.0)); }\n"
        "void step_merge(out vec4 o) { o = texture(u_step_left, vec2(0.0)) + texture(u_step_right, vec2(0.0)); }\n"
        "void main() {}\n"
    )
    plan, errors = _plan(src)
    assert errors == []
    assert plan.order.count("base") == 1
    assert plan.order.index("base") < plan.order.index("left")
    assert plan.order.index("base") < plan.order.index("right")
    assert plan.order.index("left") < plan.order.index("merge")
    assert plan.order.index("right") < plan.order.index("merge")
    assert plan.reads["merge"] == {"left", "right"}


def test_a_step_reading_itself_is_ping_pong_not_a_cycle() -> None:
    # The trap: a self-edge must be EXCLUDED from the cycle check (it reads the previous
    # frame, so it constrains nothing) while staying in the model as ping-pong.
    src = (
        "#version 330\n"
        "uniform sampler2D u_step_trail;\n"
        "void step_trail(out vec4 o) { o = texture(u_step_trail, vec2(0.0)) * 0.9; }\n"
        "void main() {}\n"
    )
    plan, errors = _plan(src)
    assert errors == []
    assert plan.order == ["trail"]
    assert plan.self_reads == {"trail"}
    assert plan.reads["trail"] == set()


def test_a_self_read_still_orders_its_other_inputs() -> None:
    src = (
        "#version 330\n"
        "uniform sampler2D u_step_scene;\n"
        "uniform sampler2D u_step_trail;\n"
        "void step_scene(out vec4 o) { o = vec4(1.0); }\n"
        "void step_trail(out vec4 o) { o = max(texture(u_step_scene, vec2(0.0)), texture(u_step_trail, vec2(0.0))); }\n"
        "void main() {}\n"
    )
    plan, errors = _plan(src)
    assert errors == []
    assert plan.order == ["scene", "trail"]
    assert plan.self_reads == {"trail"}
    assert plan.reads["trail"] == {"scene"}


def test_a_two_step_cycle_is_reported_and_does_not_hang() -> None:
    src = (
        "#version 330\n"
        "uniform sampler2D u_step_a;\n"
        "uniform sampler2D u_step_b;\n"
        "void step_a(out vec4 o) { o = texture(u_step_b, vec2(0.0)); }\n"
        "void step_b(out vec4 o) { o = texture(u_step_a, vec2(0.0)); }\n"
        "void main() {}\n"
    )
    _plan_result, errors = _plan(src)
    assert errors
    assert "cycle" in errors[0].message


def test_a_three_step_cycle_is_reported() -> None:
    src = (
        "#version 330\n"
        "uniform sampler2D u_step_a;\n"
        "uniform sampler2D u_step_b;\n"
        "uniform sampler2D u_step_c;\n"
        "void step_a(out vec4 o) { o = texture(u_step_c, vec2(0.0)); }\n"
        "void step_b(out vec4 o) { o = texture(u_step_a, vec2(0.0)); }\n"
        "void step_c(out vec4 o) { o = texture(u_step_b, vec2(0.0)); }\n"
        "void main() {}\n"
    )
    _plan_result, errors = _plan(src)
    assert errors
    assert "cycle" in errors[0].message


def test_the_eight_level_cascade_orders_coarse_to_fine() -> None:
    # The anchor scenario's shape: every level reads the scene AND the level below it.
    decls = ["uniform sampler2D u_step_scene;"]
    bodies = ["void step_scene(out vec4 o) { o = vec4(1.0); }"]
    for i in range(7, -1, -1):
        decls.append(f"uniform sampler2D u_step_c{i};")
        below = f" + texture(u_step_c{i + 1}, vec2(0.0))" if i < 7 else ""
        bodies.append(
            f"void step_c{i}(out vec4 o) {{ o = texture(u_step_scene, vec2(0.0)){below}; }}"
        )
    src = (
        "#version 330\n"
        + "\n".join(decls)
        + "\n"
        + "\n".join(bodies)
        + "\nvoid main() { gl_FragColor = texture(u_step_c0, vec2(0.0)); }\n"
    )
    plan, errors = _plan(src)
    assert errors == []
    assert plan.order[0] == "scene"
    # Coarsest (c7) merges upward into the finest (c0), so c7 precedes c0.
    for i in range(7, 0, -1):
        assert plan.order.index(f"c{i}") < plan.order.index(f"c{i - 1}")
    assert plan.final_reads == {"c0"}
    assert len(plan.order) == 9


def test_a_read_through_a_helper_is_still_an_edge() -> None:
    # Over-detection is the safe failure: a missed read would order a step before its
    # input. The match is deliberately generous, so a mention anywhere in the body counts.
    src = (
        "#version 330\n"
        "uniform sampler2D u_step_a;\n"
        "uniform sampler2D u_step_b;\n"
        "void step_a(out vec4 o) { o = vec4(1.0); }\n"
        "vec4 helper(sampler2D s) { return texture(s, vec2(0.0)); }\n"
        "void step_b(out vec4 o) { o = helper(u_step_a); }\n"
        "void main() {}\n"
    )
    plan, errors = _plan(src)
    assert errors == []
    assert plan.reads["b"] == {"a"}
    assert plan.order == ["a", "b"]


def test_a_step_nobody_reads_still_runs() -> None:
    # An orphan output is half-built authoring, not an error: it must still evaluate so
    # its thumbnail is live while the user wires it up.
    src = (
        "#version 330\n"
        "uniform sampler2D u_step_a;\n"
        "void step_a(out vec4 o) { o = vec4(1.0); }\n"
        "void main() {}\n"
    )
    plan, errors = _plan(src)
    assert errors == []
    assert plan.order == ["a"]
    assert plan.final_reads == set()


def test_the_text_scan_misses_indirect_reads_and_the_driver_does_not() -> None:
    """Why the edge set comes from GL, not from scanning the body text.

    A read through a `#define` is invisible to a text scan, and a read it does catch it
    catches by coincidence -- the name happening to appear -- not by analysis. A MISSED
    edge orders a step before its input and renders a frame of lag per hop -- the exact freska bug
    this scheduler exists to avoid, and it is invisible because the picture still looks
    plausible. The driver has already done exact dataflow analysis; ask it.
    """
    src = (
        "#version 330\n"
        "#define MY_SRC u_step_a\n"
        "uniform sampler2D u_step_a;\n"
        "uniform sampler2D u_step_b;\n"
        "uniform sampler2D u_step_c;\n"
        "vec4 helper(sampler2D s) { return texture(s, vec2(0.0)); }\n"
        "void step_a(out vec4 o) { o = vec4(1.0); }\n"
        "void step_b(out vec4 o) { o = texture(MY_SRC, vec2(0.0)); }\n"
        "void step_c(out vec4 o) { o = helper(u_step_b); }\n"
        "void main() { gl_FragColor = texture(u_step_c, vec2(0.0)); }\n"
    )
    parsed = find_steps(src, _PATH)
    assert parsed.errors == []

    # Text-scan fallback: the `#define` hop is invisible to it. (It happens to catch
    # `helper(u_step_b)` because the name appears literally -- which is exactly why a text
    # scan is untrustworthy: it is right by coincidence, not by analysis.)
    text_plan, _ = plan_steps(src, parsed.steps, _PATH)
    assert text_plan.reads["b"] == set()  # MISSED: read via `#define MY_SRC u_step_a`

    # What the driver reports for the same shader (measured on a real context).
    active = {
        "a": set(),
        "b": {"u_step_a"},
        "c": {"u_step_b"},
        "": {"u_step_c"},
    }
    gl_plan, errors = plan_steps(src, parsed.steps, _PATH, active_by_step=active)
    assert errors == []
    assert gl_plan.reads["b"] == {"a"}
    assert gl_plan.reads["c"] == {"b"}
    assert gl_plan.order == ["a", "b", "c"]
    assert gl_plan.final_reads == {"c"}


def test_the_driver_edge_set_still_treats_a_self_read_as_ping_pong() -> None:
    src = (
        "#version 330\n"
        "uniform sampler2D u_step_scene;\n"
        "uniform sampler2D u_step_trail;\n"
        "void step_scene(out vec4 o) { o = vec4(1.0); }\n"
        "void step_trail(out vec4 o) { o = vec4(0.0); }\n"
        "void main() {}\n"
    )
    parsed = find_steps(src, _PATH)
    plan, errors = plan_steps(
        src,
        parsed.steps,
        _PATH,
        active_by_step={
            "scene": set(),
            "trail": {"u_step_scene", "u_step_trail"},
            "": set(),
        },
    )
    assert errors == []
    assert plan.self_reads == {"trail"}
    assert plan.reads["trail"] == {"scene"}
    assert plan.order == ["scene", "trail"]
