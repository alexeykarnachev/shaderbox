<!-- THE dogfood report shape. Every round produces this file, in this order, with these headings.

WHY IT IS RIGID: five reports were audited and they used three incompatible heading taxonomies;
the bottom line sat in a different place in every one, and one report had none in the first 15
lines at all. The maintainer reads these weeks apart and should never have to re-learn the shape.
So: keep the headings verbatim, keep the order, and keep a section with "none" in it rather than
deleting it -- an absent section and an empty one mean different things.

WRITTEN FOR A HUMAN who has not read any spec. No internal codenames ("D6", "F2", "wave 3"): say
what the thing IS, and put an id in parentheses after the plain-English statement if it helps
someone find the spec. One idea per paragraph. Numbers carry their meaning ("4% cache reuse
against 78-85% either side", not "2.9% vs 62.7%").

Every {{AUTO:...}} is filled by analyze.py from the run's logs -- never hand-summed. The
{{HUMAN:...}} slots are yours, but a NUMBER in one still comes from a script you ran against
`dogfood/runs/<experiment>/events.jsonl`, never from memory. Delete these comments when you fill
it in. -->

# Dogfood round: {{HUMAN:one_line_title}}

**What this was.** {{HUMAN:what_was_tested}} <!-- The mission in 2-3 sentences: the ask, the
models, how many turns, and what the round was meant to ESTABLISH. -->

**Models:** {{AUTO:model}} — {{HUMAN:why_these_models}}

**Commit:** `{{HUMAN:sha}}` · **Station:** `dogfood/runs/{{HUMAN:experiment_id}}/` · **Run:** {{AUTO:run_label}} ({{AUTO:date}})

{{HUMAN:caveats}} <!-- Anything that limits the comparison: fewer turns than the round being
compared against, a model swapped, a changed ask. Omit the line if there are none. -->

---

## The headline

{{HUMAN:headline}}

<!-- The bottom line, in the first 20 lines of the document, ALWAYS under this heading. One bold
sentence, then at most a short paragraph. A reader who stops here must have the result. -->

---

## Results

| Model | Outcome | Cost | Turns | Requests | Requests per tool call | Hidden reasoning |
|---|---|---|---|---|---|---|
{{HUMAN:results_table}}

**Outcome vocabulary** (`dogfood/report/log.py::OUTCOMES`, enforced at `end_attempt`) — an outcome
says what the MODEL reached, never that the driver stopped driving:
`built` goal met · `partial` real progress, goal not reached · `regressed` ended worse than it
started · `blocked` an engine defect stopped it · `abandoned` the model gave up · `smoke` infra check.

<!-- These columns are FIXED. Do not rename them or swap in a different metric: a previous round
called a raw count "requests" and the next called a ratio "requests per tool call", so the same
label meant two different quantities and neither could be compared. Add a column only by adding it
here for every future round too. -->

---

## The animation

{{HUMAN:video_embed}}

<!-- REQUIRED whenever the mission produced something that moves, which is nearly always. A still
cannot show motion, and motion is half of what these missions ask for. Produce it with
`drive.py --mp4 4` on the final turn (or `h.render_video_mp4(seconds=4, fps=20, size=320)`), which
puts it on the attempt page automatically. Link the file here and say what it shows. If the mission
had no moving output, write "not applicable -- <why>" and keep the heading. -->

---

## Per model

<!-- One `###` subsection per model, in the results-table order. Each one: what it built, what it
cost, what went right, what went wrong. Lead with the outcome, then the mechanism. -->

{{HUMAN:per_model}}

---

## What this found in the engine

{{HUMAN:engine_findings}}

<!-- Defects and behaviours of ShaderBox itself, NOT of the models. For each: what happens, the
evidence, and whether it is fixed or open. A finding with no fix says so plainly and says why
(a wrong fix reverted, one instance not enough to act on). Write "none this round" if there were
none -- that is a real result. -->

---

## Predictions from the previous round

{{HUMAN:predictions}}

<!-- Only when a previous round left measurable claims. State each as **Held.** or **Did not
hold.** with the number that settles it. A refuted prediction is the most valuable line in a
report: never soften it, never bury it under a confirmed one. Write "none outstanding" if the
previous round left no claims. -->

---

## Tool coverage

{{AUTO:tool_coverage_table}}

**Cold:** {{AUTO:cold_tools}} — {{HUMAN:cold_tool_reading}}

<!-- Thin coverage is a first-class finding. For each cold tool say which it was: the scenario
never pressured it, or a move aimed at it and the model dodged (the second is a finding about the
model). -->

---

## What a next round would test

{{HUMAN:next_round}}

<!-- Open questions this round raised and could not settle. Not a backlog: only things a NEXT RUN
would answer. Code-scoped work goes to a commit or a TODO at the code site, never parked here. -->
