# 020 · 14 — Slice 2: line-anchored editing tools + apply-feedback

Step 2 of the maintainer's 2-step sequence (the lexer was step 1, `13_glsl_lexer.md`). Slice 1 gave the
agent ONE mutating tool, `edit_shader` (substring match → replace → recompile). Two friction points
remain that line-anchored tools remove:

1. **Quoting overhead.** To insert a line, the model must quote a unique neighboring substring as
   `old_str` and echo it back in `new_str` — reproducing a region it doesn't intend to change just to
   anchor the insert. `get_current_shader` already shows the model **1-based line numbers** (the
   `cat -n` listing); a line-anchored tool lets it edit by those numbers directly, quoting nothing.
2. **No "what changed" signal.** A clean `edit_shader` returns `"ok — compiled clean"` — the model
   can't see WHAT the file now looks like around the edit without re-reading the whole shader. A
   concise apply-feedback (the changed line range + a few context lines) closes the write→observe loop
   without a full re-read.

The lexer (step 1) already fixed the *whitespace-reproduction* half of friction-1 for `edit_shader`;
line-anchored tools remove the *quoting* half and are the cheaper primitive for inserts/replacements
the model can address by line number. `edit_shader` stays (the token matcher is its matching layer);
the new tools are additive, not a replacement.

---

## Goal

Two new eager, current-node, mutating tools + apply-feedback on every mutating edit:

1. **`replace_lines(start_line, end_line, new_text)`** — replace the INCLUSIVE 1-based line range
   `[start_line, end_line]` with `new_text` (which may be any number of lines, including zero → a
   deletion), then recompile. Line numbers are the ones `get_current_shader` displayed.
2. **`insert_after(line, new_text)`** — insert `new_text` as new line(s) AFTER 1-based `line`
   (`line=0` → insert at the very top, before line 1), then recompile. Pure insertion; existing lines
   shift down, none are replaced.
3. **Apply-feedback on `edit_shader` + `replace_lines` + `insert_after`.** On a clean or
   compiled-with-errors apply, the tool result includes a concise "what changed" excerpt — the final
   1-based line range that the edit produced plus a few lines of surrounding context — so the model
   sees the post-edit shape without re-reading. (`edit_shader` gains this too: its match resolves to a
   byte span → a line range.)

### Locked invariants

- **L1 — Line numbers are 1-based and match `get_current_shader`'s listing exactly.** The listing is
  `_number_lines(text)`, which is `text.split("\n")` enumerated from 1. So the model's line model IS
  `src.split("\n")`: a file ending in `\n` shows a trailing EMPTY line as line `N` (e.g. `"a\nb\n"` →
  3 lines: `a`, `b`, ``). The handler MUST split with `split("\n")`, **never `splitlines()`**
  (`splitlines()` drops the trailing empty + treats `\r`/`\f` as breaks → diverges from the listing).
  The model must `get_current_shader` first (same rule as `edit_shader`, §16.2).
- **L2 — All editing is defined on the line LIST, not byte offsets — round-trip byte-exact by
  construction.** Let `lines = src.split("\n")` (length `N`). Every edit produces a new list, and the
  new source is `"\n".join(new_lines)`. Because `split("\n")` and `"\n".join(...)` are exact inverses,
  an edit that reproduces the same lines is byte-identical — no seam-newline arithmetic, no doubled or
  dropped `\n`. Concretely, with `new_text` split into its own lines `repl = new_text.split("\n")`:
  - **`replace_lines(s, e, new_text)`** (1-based inclusive, `1 <= s <= e <= N`): `new_lines =
    lines[:s-1] + repl + lines[e:]`. An empty `new_text` gives `repl == [""]`; to DELETE the range
    cleanly (no leftover blank line) the tool passes `repl = []` when `new_text == ""` (see D2).
  - **`insert_after(k, new_text)`** (`0 <= k <= N`): `new_lines = lines[:k] + repl + lines[k:]`.
    `k=0` inserts before line 1; `k=N` appends after the last line. No special-casing of the endpoints
    — slicing `lines[:k]`/`lines[k:]` handles `k=0` and `k=N` uniformly, and the `"\n".join` supplies
    every seam newline. (This is why the model is list-based, NOT byte-offset-based — the endpoint
    seam bugs the byte-offset formula hits do not exist here.)
- **L3 — Out-of-range fails loud, mutates nothing.** `replace_lines`: `s < 1`, `e > N`, or `s > e` →
  error string naming the valid range (`"the shader has N lines (1..N)"`), NO mutation.
  `insert_after`: `k < 0` or `k > N` → same. The `s > e` check lives in the `replace_lines` TOOL
  handler (D2) — the App capability never receives an inverted range, so there is no insert-vs-error
  ambiguity. Same fail-soft posture as `edit_shader`'s 0/ambiguous match.
- **L4 — The apply is on the AUTHORITATIVE source, one bridge round-trip, like `apply_shader_edit`.**
  Split + edit the list + join + recompile + persist + editor-refresh happen together on the main
  thread against `node.source.text` via `node.release_program(new_text)` → `compile()` →
  `write_text` → `sync_editor_from_disk` (the EXACT sequence `_copilot_apply_shader_edit::_on_main`
  uses — `release_program` is what swaps in the new text, not a bare assignment). No worker-side
  re-read, no staleness window (mirrors §16.3).

---

## Out of scope (each with a trigger)

- **A semantic-editing suite (`rename_symbol`, `outline`, `add_uniform`, `extract_function`).**
  Deliberately deferred — the maintainer kept Slice 2 tight. **Trigger:** a copilot trace shows the
  model struggling to do a rename/refactor with the line + substring tools (e.g. it issues N
  `replace_lines` to rename one identifier across a file). Build the semantic tool then, on the lexer.
- **A diff/patch-format tool (unified-diff apply).** A bigger primitive; the line + substring + insert
  trio covers the common edits. **Trigger:** the model repeatedly wants to apply a multi-hunk change
  and the per-tool round-trips get expensive (visible in a trace as many sequential edits in one turn).
- **Multi-file / lib-file editing.** Slice 2 is current-node-only, like Slice 1. **Trigger:** the
  lib-file editing deferrals in `todo.md` (cross-file uniform jump, export-from-selection) get picked
  up — line tools generalize to lib files at that point.
- **A `delete_lines` tool.** Folded INTO `replace_lines` (an empty `new_text` deletes the range) — a
  separate tool would be a redundant surface. **Trigger:** a trace shows the model confused by
  "replace with nothing" enough that an explicit `delete_lines` reads clearer.
- **Undo/redo of agent edits.** Out of scope across the whole copilot feature (the editor + disk are
  the source of truth; the user's own Ctrl+Z covers the editor). **Trigger:** users ask to revert a
  copilot edit without manual undo.

---

## Design decisions
*(numbered, lock-in only)*

1. **Two new tools live in `tools/shader.py` alongside `edit_shader` (the current-node shader group).**
   They are the same category (`"shader"`), eager, `mutating=True`, `needs_gl=True`,
   `gate_policy=NONE` (single reversible edits, like `edit_shader`). `shader_tools(caps)` returns the
   expanded list; `build_registry` is unchanged (it already splats `shader_tools`).

2. **One new capability `apply_line_edit(start_line, end_line, new_text) -> EditResult`, but it is
   ONLY ever called with a valid `start_line <= end_line` range — NO inverted-range insert encoding.**
   The capability's contract: replace the 1-based inclusive line range `[start_line, end_line]` (over
   the `split("\n")` list, L2) with `new_text`'s lines. Insert is NOT an inverted range; it is expressed
   in the `insert_after` TOOL handler as a range that selects ZERO existing lines via a sentinel the
   capability understands — concretely, `apply_line_edit` treats `end_line == start_line - 1` as the
   ONE legal "empty selection at position `start_line`" (pure insert: `new_lines = lines[:start_line-1]
   + repl + lines[start_line-1:]`). `insert_after(k, t)` → `apply_line_edit(k+1, k, t)`. This is the
   single empty-selection case; ANY OTHER `start > end` (i.e. `end < start - 1`) is invalid and the
   capability errors. Crucially, the `replace_lines` TOOL handler validates `start <= end` BEFORE
   calling the capability (L3), so a user `replace_lines(5, 3)` errors in the tool layer and never
   reaches the capability as a spurious insert — resolving the D2/L2 ambiguity at the right altitude.
   The App handler mirrors `_copilot_apply_shader_edit::_on_main` (L4 sequence). `replace_all`-style
   empty deletes (`new_text == ""`) pass `repl = []` (L2) so the range vanishes with no blank residue.

3. **`EditResult` gains the apply-feedback fields (leaf, cycle-free).** Add
   `changed_excerpt: str = ""` (the post-edit "what changed" context block) and
   `changed_range: tuple[int, int] | None = None` (the 1-based line range the edit produced). Populated
   on a successful single-region apply. Empty/None on a non-apply (0/ambiguous/out-of-range) — those
   return the existing error strings unchanged. The tool layer appends the excerpt to the success
   message when present.

4. **The "what changed" excerpt is computed App-side from the NEW text.** After the edit, the handler
   knows the changed lines' span in the NEW source (for line edits: `[start_line, start_line +
   len(repl) - 1]` against the new list; for `edit_shader`: the applied byte span mapped to new-source
   line numbers — D5). It renders that range plus `config.edit_feedback_context` (default 2) lines on
   each side, line-numbered against the NEW source (reusing `_number_lines`'s `split("\n")`+enumerate
   style over a sub-range). A new `CopilotConfig` field `edit_feedback_context: int = 2`. Pure-text
   render of the already-computed new source — no extra GL, no extra round-trip.

5. **`edit_shader` feedback: ONLY for a single applied span; suppressed on multi-span `replace_all`.**
   The token matcher returns spans; a non-`replace_all` edit applies exactly ONE span → map it to a
   new-source line range → excerpt (the shared path with line edits). A `replace_all` edit with >1
   span changes multiple, possibly distant regions — there is no single honest `changed_range`, so the
   handler leaves `changed_excerpt=""`/`changed_range=None` and the tool result notes the count instead
   (`"ok — compiled clean (N regions changed)"`). No misleading bounding-range excerpt.

6. **The agent loop changes in ONE place: widen the retry-cap trigger.** The new tools are
   `ToolDefinition`s the registry dispatches generically. The slice-12 consecutive-failed-edit cap
   counts only `tc.name == "edit_shader"`; widen the predicate to
   `registry.is_mutating(tc.name) and not ok` (keep the `and not ok` — a SUCCESSFUL mutating edit or
   ANY non-mutating tool falls to the `else` and resets, matching slice-12 D3). After slice 2 the
   mutating set is exactly {`edit_shader`, `replace_lines`, `insert_after`} — all edit-spiral-prone, so
   the widening is safe (todo trigger: a FUTURE non-spiral-prone mutating tool would wrongly count —
   revisit then). Also make the giveup `AgentError` note tool-agnostic — it currently says "my old_str
   kept not matching", which is false for a `replace_lines` out-of-range spiral; reword to "I couldn't
   apply that edit after N tries — I've stopped to avoid looping."

7. **Tool descriptions teach the line-number contract explicitly.** `replace_lines` /
   `insert_after` descriptions state: line numbers are the ones shown by `get_current_shader`; you must
   read first; `replace_lines` is 1-based INCLUSIVE `[start, end]` and an empty `new_text` deletes the
   range; `insert_after` uses `0` for top-of-file and `N` to append; `new_text` is inserted verbatim
   (you control its indentation, including any leading whitespace to match surrounding code). Write
   these as carefully as `_EDIT_SHADER_DESC`.

---

## Files touched
- **`shaderbox/copilot/capabilities.py`:** add `apply_line_edit: Callable[[int, int, str], EditResult]`
  to `CopilotCapabilities`; add `changed_excerpt: str = ""` + `changed_range: tuple[int, int] | None =
  None` to `EditResult`.
- **`shaderbox/copilot/config.py`:** add `edit_feedback_context: int = 2`.
- **`shaderbox/app.py`:** new `_copilot_apply_line_edit(start_line, end_line, new_text)` handler
  (split-on-`\n` list edit per L2, `release_program`→`compile`→`write_text`→`sync_editor_from_disk` per
  L4, wired into `CopilotCapabilities`); a `_changed_excerpt(new_text, line_range, context)` helper
  (line-numbered sub-range render over `new_text.split("\n")`, reusing `_number_lines`'s style);
  `_copilot_apply_shader_edit` populates the feedback fields (single span → new-source line range →
  excerpt; multi-span `replace_all` → none, per D5); `apply_line_edit=self._copilot_apply_line_edit` in
  the caps construction. The line helpers split with `split("\n")`, NEVER `splitlines()` (L1).
- **`shaderbox/copilot/tools/shader.py`:** two new `ToolDefinition`s (`replace_lines`, `insert_after`)
  with pydantic arg models + handlers; `replace_lines` validates `start <= end` (D2) before calling
  `caps.apply_line_edit(start, end, new_text)`; `insert_after` calls
  `caps.apply_line_edit(line+1, line, new_text)`. Success-message builders for all three mutating tools
  append `result.changed_excerpt` when present (and the `replace_all`-count note per D5). New
  `_REPLACE_LINES_DESC` / `_INSERT_AFTER_DESC`.
- **`shaderbox/copilot/agent.py`** (`run_turn`): change the cap predicate
  `tc.name == "edit_shader" and not ok` → `registry.is_mutating(tc.name) and not ok` (D6); reword the
  giveup `AgentError` note to be tool-agnostic (drop the `old_str` phrasing).
- **`tests/test_glsl_lex.py`:** unchanged (the lexer is untouched).
- **NEW `tests/` coverage (extend `test_copilot_loop.py` or a new `test_line_editing.py`):** the
  `_fake_caps` fake gains an `apply_line_edit` faithful to the App (`split("\n")` list edit) so the
  loop tests drive the new tools; the test set in Manual verification below.
- **Docs:** `roadmap.md` (020 row + banner: Slice 2 done, NEXT = 022 or semantic tools per trace);
  this spec.

## Manual verification
- `make check` + `make smoke` green.
- **Unit (the real proof):**
  - *Round-trip (L2):* a no-op `replace_lines(s, e, <the exact current lines s..e>)` yields a
    byte-identical source — proves `split("\n")`/`"\n".join` inverse, no seam bug.
  - *replace:* `replace_lines(2, 3, "X")` on `"a\nb\nc\nd"` → `"a\nX\nd"`; multi-line `new_text`
    expands correctly; empty `new_text` (`repl=[]`) deletes lines 2-3 → `"a\nd"` (no blank residue).
  - *insert endpoints:* `insert_after(0, "X")` → `"X\na\nb\nc\nd"` (top); `insert_after(4, "X")` on
    `"a\nb\nc\nd"` → `"a\nb\nc\nd\nX"` (append, separating `\n` present); `insert_after(2, "X")` →
    `"a\nb\nX\nc\nd"`.
  - *Trailing-newline L1 (the divergence guard):* on `"a\nb\nc\n"` the model SEES 4 lines (line 4
    empty). `insert_after(4, "X")` appends after the empty line; `replace_lines(4, 4, "z")` replaces
    the trailing empty line. The handler's line model must equal `_number_lines`'s `split("\n")` count
    — assert the tool's `N` matches the listing's last line number on a trailing-`\n` fixture.
  - *out-of-range fail-soft (L3):* `replace_lines(0, 2, …)`, `replace_lines(2, 99, …)`,
    `replace_lines(5, 3, …)` (start>end), `insert_after(-1, …)`, `insert_after(N+1, …)` each → error
    string naming the valid range, source UNCHANGED.
  - *insert-vs-error disambiguation (B1):* `replace_lines(5, 3, …)` ERRORS (tool-layer `start<=end`
    check) AND `insert_after(2, …)` (→ `apply_line_edit(3, 2, …)`) INSERTS — both via the same
    capability, proving the tool-layer guard resolves the overload.
  - *apply-feedback:* a clean line edit returns the changed range + `edit_feedback_context` lines on
    each side, correct 1-based numbers against the NEW source (line numbers reflect the post-edit
    shift). A single-span `edit_shader` returns the same; a multi-span `replace_all=true` `edit_shader`
    returns NO excerpt and the "N regions changed" count (D5).
  - *retry cap (D6):* a spiraling `replace_lines` (out-of-range every iteration) stops at
    `max_edit_retries`, surfacing the tool-agnostic `AgentError` (slice-12 C path, now via
    `is_mutating`); a mix (failed `replace_lines`, then a successful `edit_shader`) RESETS the counter.
- **Live (maintainer, UN-headless):** drive a real turn — "add a new uniform and use it" — and confirm
  the model uses `insert_after`/`replace_lines` by line number (read the transcript), the apply-feedback
  shows the changed region, and the edit compiles. Hand to maintainer with a one-line note.

## Open questions for the user
- **O1 — Widen the retry cap to all mutating shader tools (Design 6)?** I recommend YES: a model
  spiraling on `replace_lines` (e.g. repeatedly sending an out-of-range range) is the same failure the
  slice-12 cap was built for, and `ToolDefinition.mutating` already marks the set. The alternative
  (cap only `edit_shader`) would let a `replace_lines` spiral run to `max_iterations`. Default: widen.
- **O2 — `replace_lines` line-range inclusivity:** I default to **inclusive `[start, end]`** (replace
  lines start through end, both included) as the most natural reading of "replace lines 5–8". L4 pins
  the exact byte-span semantics; flag if you'd prefer a half-open `[start, end)` convention.

## Review history
- **Plan-locked (maintainer).** O1 = widen the slice-12 retry cap from `edit_shader`-only to all
  mutating shader tools (`registry.is_mutating(name)`). O2 = `replace_lines` range is inclusive
  `[start, end]`.
- **Pre-impl review (1 agent, adversarial) → SHOULD-NOT-LAND; all 4 blockers + the should-fixes folded
  in:** the newline/byte-span design was the failure surface. **A1+A3 (seam-newline + endpoint inserts):
  re-pinned L2 on the `split("\n")` LINE LIST** (`new_lines = lines[:s-1] + repl + lines[e:]`,
  `"\n".join` supplies every seam) — byte-exact by construction, no offset arithmetic, endpoints fall
  out of the slicing. **A2 (L1 trailing-newline divergence): L1 now mandates `split("\n")`, never
  `splitlines()`** (matches `_number_lines`, verified). **B1 (the `start>end` insert-vs-error overload):
  the `replace_lines` TOOL handler validates `start<=end` before the capability, so the inverted range
  is reserved exclusively for `insert_after`** — the capability never sees a `replace_lines`-originated
  inverted range. **C1 (multi-span `replace_all` feedback): D5 suppresses the excerpt + emits an
  "N regions changed" count** rather than a misleading bounding range. **D3/D4 (cap predicate +
  message): full predicate `is_mutating(name) and not ok`, tool-agnostic giveup note.** Tests added for
  the trailing-newline guard, the B1 disambiguation, and the multi-span feedback. No redesign — spec
  re-grounding the line model on the list + moving the disambiguation to the right altitude.
- **Post-impl review (1 agent, adversarial) → FIX-THEN-SHIP, the one real bug fixed:** the `edit_shader`
  apply-feedback over-reported the last changed line by one when `new_str` ended in `\n` (the end offset
  landed on the next, unchanged line) — source bytes were correct, the excerpt was misleading. Fixed:
  `end_off = start_off + max(len(new_str) - 1, 0)` in `_copilot_apply_shader_edit`. Two nits folded:
  `_copilot_edit_result` had no `self` → now the module-level free function `_edit_result`; the
  empty-`new_text` insert no-op is left as acceptable (matches the delete-no-excerpt posture). Added
  the byte-span→line-range regression test (incl. the trailing-newline case) + `_applied_result`
  excerpt/multi-span-count/compile-error tests. Reviewer confirmed the line-edit guard airtight against
  uncaught out-of-range, the B1 disambiguation correct, the retry-cap widening safe. 136 tests green.
