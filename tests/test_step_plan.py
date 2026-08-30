"""Evaluation order: topological sort WITH memoization, and ping-pong as a non-edge.

Both prior designs in this repo's history got exactly one half. The deleted DAG had correct
pull-recursion order but no memoization, so a diamond re-rendered its shared ancestor once
per consumer. freska had per-node targets but iterated an unordered_map under its own
`// TODO: this is incorrect!`, lagging an N-step chain by N-1 frames nondeterministically.
"""

from pathlib import Path

from shaderbox.step_spec import parse_steps, plan_steps

_PATH = Path("node.frag.glsl")


def _plan(source: str):
    parsed = parse_steps(source, _PATH)
    assert parsed.errors == [], parsed.errors
    return plan_steps(source, parsed.steps, _PATH)


def test_a_linear_chain_orders_producers_first() -> None:
    src = (
        "#version 330\n"
        "uniform sampler2D u_a;  // step\n"
        "uniform sampler2D u_b;  // step\n"
        "void step_a(out vec4 o) { o = vec4(1.0); }\n"
        "void step_b(out vec4 o) { o = texture(u_a, vec2(0.0)); }\n"
        "void main() { gl_FragColor = texture(u_b, vec2(0.0)); }\n"
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
        "uniform sampler2D u_b;  // step\n"
        "uniform sampler2D u_a;  // step\n"
        "void step_b(out vec4 o) { o = texture(u_a, vec2(0.0)); }\n"
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
        "uniform sampler2D u_base;   // step\n"
        "uniform sampler2D u_left;   // step\n"
        "uniform sampler2D u_right;  // step\n"
        "uniform sampler2D u_merge;  // step\n"
        "void step_base(out vec4 o) { o = vec4(1.0); }\n"
        "void step_left(out vec4 o) { o = texture(u_base, vec2(0.0)); }\n"
        "void step_right(out vec4 o) { o = texture(u_base, vec2(0.0)); }\n"
        "void step_merge(out vec4 o) { o = texture(u_left, vec2(0.0)) + texture(u_right, vec2(0.0)); }\n"
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
        "uniform sampler2D u_trail;  // step\n"
        "void step_trail(out vec4 o) { o = texture(u_trail, vec2(0.0)) * 0.9; }\n"
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
        "uniform sampler2D u_scene;  // step\n"
        "uniform sampler2D u_trail;  // step\n"
        "void step_scene(out vec4 o) { o = vec4(1.0); }\n"
        "void step_trail(out vec4 o) { o = max(texture(u_scene, vec2(0.0)), texture(u_trail, vec2(0.0))); }\n"
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
        "uniform sampler2D u_a;  // step\n"
        "uniform sampler2D u_b;  // step\n"
        "void step_a(out vec4 o) { o = texture(u_b, vec2(0.0)); }\n"
        "void step_b(out vec4 o) { o = texture(u_a, vec2(0.0)); }\n"
        "void main() {}\n"
    )
    _plan_result, errors = _plan(src)
    assert errors
    assert "cycle" in errors[0].message


def test_a_three_step_cycle_is_reported() -> None:
    src = (
        "#version 330\n"
        "uniform sampler2D u_a;  // step\n"
        "uniform sampler2D u_b;  // step\n"
        "uniform sampler2D u_c;  // step\n"
        "void step_a(out vec4 o) { o = texture(u_c, vec2(0.0)); }\n"
        "void step_b(out vec4 o) { o = texture(u_a, vec2(0.0)); }\n"
        "void step_c(out vec4 o) { o = texture(u_b, vec2(0.0)); }\n"
        "void main() {}\n"
    )
    _plan_result, errors = _plan(src)
    assert errors
    assert "cycle" in errors[0].message


def test_the_eight_level_cascade_orders_coarse_to_fine() -> None:
    # The anchor scenario's shape: every level reads the scene AND the level below it.
    decls = ["uniform sampler2D u_scene;  // step"]
    bodies = ["void step_scene(out vec4 o) { o = vec4(1.0); }"]
    for i in range(7, -1, -1):
        decls.append(f"uniform sampler2D u_c{i};  // step, scale: {2.0 ** -i}, f2")
        below = f" + texture(u_c{i + 1}, vec2(0.0))" if i < 7 else ""
        bodies.append(
            f"void step_c{i}(out vec4 o) {{ o = texture(u_scene, vec2(0.0)){below}; }}"
        )
    src = (
        "#version 330\n"
        + "\n".join(decls)
        + "\n"
        + "\n".join(bodies)
        + "\nvoid main() { gl_FragColor = texture(u_c0, vec2(0.0)); }\n"
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
        "uniform sampler2D u_a;  // step\n"
        "uniform sampler2D u_b;  // step\n"
        "void step_a(out vec4 o) { o = vec4(1.0); }\n"
        "vec4 helper(sampler2D s) { return texture(s, vec2(0.0)); }\n"
        "void step_b(out vec4 o) { o = helper(u_a); }\n"
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
        "uniform sampler2D u_a;  // step\n"
        "void step_a(out vec4 o) { o = vec4(1.0); }\n"
        "void main() {}\n"
    )
    plan, errors = _plan(src)
    assert errors == []
    assert plan.order == ["a"]
    assert plan.final_reads == set()
