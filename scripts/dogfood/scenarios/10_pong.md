# Echelon-2 scenario 10 — Self-playing pong (script state machine)

**Capability cluster:** a real game state machine in `script.py` — continuous motion (ball) +
reactive control (two AI paddles) + discrete events (wall/paddle bounces, scoring) + persistent
score state — driving a multi-uniform shader. This is the state-logic step past 05's single
integrator.

## Opening message (verbatim)

> Make a self-playing pong game: a ball bouncing around the field, a left and a right paddle that
> each move on their own to intercept it, and a score for each side shown as a row of small dots
> at the top of the field. The paddles should have limited speed so they sometimes miss and points
> actually get scored. All game logic lives in the python script; the shader only draws the state
> it is given.

## Ground truth / checklist (pixel-judged; raw script values only for rates/counters)

- [ ] field, ball, two side paddles, top dot rows all visible
- [ ] the ball MOVES continuously and stays inside the field at all times
- [ ] ball reflects off top/bottom walls (y reverses near the edges)
- [ ] each paddle tracks the ball vertically (paddle-y correlates with ball-y over time) with
      visible lag (limited speed)
- [ ] a miss resets the ball and INCREMENTS the corresponding dot row (dot count is monotonic,
      never decreasing; verified over a 30-60s replay)
- [ ] logic in `script.py` (script tools fired; shader consumes uniforms only)

## Drive

Fresh seeded project. Correction budget ≤2 (symptoms only). Judge motion/events from
`render_strip` + an mp4 replay (export replays the script from a clean init — the same run every
time); count dots per frame with `judge.py` primitives. Report: dialogue + 15-20s mp4 + six-axis
verdicts (CODE: state-machine structure — states/transitions readable, no copy-paste per paddle).
