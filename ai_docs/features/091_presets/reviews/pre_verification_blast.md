# 091 pre-implementation review — verification & blast radius

Reviewer role: does the `## Verification` list catch a wrong implementation of each locked
decision, which invariants nothing reads, and what the new surfaces touch. Read-only; nothing
under `shaderbox/`, `tests/` or the spec was edited. Every finding below is a file read or a
command run, cited.

## Verdict: **PARTIAL**

Eight of the eleven items catch their decision. Three need rewriting before implementation, one
of them because the gate it specifies **cannot be written as described**:

| Item | Verdict | Why |
| --- | --- | --- |
| 1 `entry_points` | WEAK | Its named falsifier is not reachable from either example's shape. |
| 2 `plan_import` | CATCHES | Seven sub-cases, each with a distinct falsifier; GL-free; fits `test_pass_graph.py`'s style. |
| 3 `import_passes` e2e | MISSING FIXTURE | "the Bloom Chain example" does not exist; and its asserted pass count and sampler names are wrong for the artifact that does. |
| 4 rendered import | WEAK | Depends on item 3's fixture; the baseline it names ("the starter's black") is not black. |
| 5 group survives verbs | CATCHES | Its named falsifier is the real bug class and `with_passes` is the reason it holds. |
| 6 `group_runs` | CATCHES | Pure, fits `test_pass_strip_layout.py` exactly. |
| 7 theme invariant | **FAIL as written** | The second half ("putting `yellow_b` in `GROUP_TINTS` makes import fail") is not expressible as a test; the first half (subprocess) is the wrong shape for the claim. |
| 8 copilot table / `set_pass` | CATCHES | Fits `test_copilot_pass_tools.py`'s substring-on-`res.table` style. |
| 9 command registered | CATCHES | Verified by reading the test: two independent gates, both hard. |
| 10 modal state resets | CATCHES | Fits `test_pass_draft.py`'s App-driven style. |
| 11 smoke draws the strip | WEAK | Smoke never makes a grouped document, so the claim is untested as written. |

Plus **seven invariants with no check at all** (section B), two of them in the D5 save path.

One finding is free money and the spec misses it: `tests/test_project_management.py::test_every_popup_state_has_a_draw_call`
already gates D10's wiring. See B-0.

---

## A. Per verification item

### Item 1 — `entry_points` — WEAK

**What I ran.** Both shipped shapes, through the real `Document.effective_wiring()`:

```
# Radiance Cascades (shaderbox/resources/document_examples/77a84d27-…)
COMPILED wiring: {'cascade': {'u_df':'df','u_paint':'paint','u_prev':'cascade'},
                  'composite': {'u_cascade':'cascade','u_paint':'paint'},
                  'df': {'u_jfa':'jfa'}, 'jfa': {'u_prev':'jfa','u_seed':'seed'},
                  'paint': {}, 'seed': {'u_paint':'paint'}}
COMPILED roots: ['paint']

# the bloom-chain TEST FIXTURE (tests/fixtures/bloom_chain)
COMPILED wiring: {'blur': {'u_bright':'bright'}, 'bright': {'u_scene':'scene'},
                  'composite': {'u_blur':'blur','u_scene':'scene','u_trail':'trail'},
                  'scene': {}, 'trail': {'u_prev':'trail','u_scene':'scene'}}
COMPILED roots: ['scene']
```

The spec's counts are right: one root each. **But the named falsifier is unreachable from these
shapes.** The falsifier is "counting a self-read as an input makes the feedback root disappear",
and neither example has a self-reading *root*: `cascade` and `jfa` (RC) and `trail` (bloom) all
read a sibling as well as themselves, so they are non-roots either way. Under the bug they stay
non-roots; the answer is unchanged and the test stays green.

The sub-case "a self-reading root is still a root" is listed, so the intent is there — it just
needs a wiring the shipped examples do not provide.

**Rewrite.**

> 1. **`entry_points`** (`tests/test_pass_graph.py`, GL-free): over hand-built wirings —
>    Bloom-shaped and RC-shaped answer one root each (`scene`, `paint`); a wiring whose ONLY
>    edge is a self-read (`{"acc": {"u_prev": "acc"}}`) answers `["acc"]`; a two-input
>    compositor answers both its leaves; a pass with no wiring entry is a root.
>    Falsifier: counting a self-read as an input makes the `{"acc": …}` case answer `[]`.
>    (The self-read case must be its OWN wiring — in both shipped examples every self-reader
>    also reads a sibling, so the bug is invisible there; verified by running
>    `effective_wiring()` on both.)

Fixture: none. `tests/test_pass_graph.py` has zero fixtures and a `_plan` helper that
asserts invariants on every call — a pure `entry_points` test is a natural fit.

### Item 2 — `plan_import` — CATCHES

GL-free, in a new `tests/test_pass_import.py`, seven lettered sub-cases with distinct
falsifiers. Each is decidable from `ImportPlan`'s four fields. It fits `test_pass_graph.py`'s
house style (no fixtures, hand-built `Wiring` dicts, a "Falsifier:" comment per test).

**The (a) falsifier is real, and I confirmed the bug it guards against is the one an
implementer would actually write.** The prefix genuinely destroys name-rule wiring:

```
>>> passes = {'bloom_scene','bloom_bright','bloom_blur','bloom_trail','bloom_composite','main'}
>>> wired_pass(AutoSource(), 'u_bright', 'bloom_blur', passes)   -> None
>>> wired_pass(AutoSource(), 'u_scene',  'bloom_bright', passes) -> None
>>> wired_pass(AutoSource(), 'u_blur',   'bloom_composite', passes) -> None
>>> wired_pass(AutoSource(), 'u_prev',   'bloom_trail', passes)  -> 'bloom_trail'
```

So an implementation that copies entries and skips materialization produces a bundle where
every inter-pass edge is gone and only feedback survives — exactly (a)'s claim, and a bug that
would otherwise ship silently because `wired_pass` returns `None` rather than raising. Note for
the implementer: `u_prev` self-resolving means (a) must assert the self-read row is written as
`PassSource("bloom_trail")` and not skipped as "already correct" — the spec says this ("a
self-read points at the renamed self"); the test must check it, since it is the one sampler that
works either way today and would break the day a group is renamed.

### Item 3 — `import_passes` end to end — MISSING FIXTURE

Three separate defects, all mechanical:

**3a. There is no "Bloom Chain" example.** `shaderbox/constants.py:15-22` lists six:

```python
EXAMPLE_ORDER = [
    "53724dbd-…",  # UV Mango          (1 pass)
    "73ea2431-…",  # Media Input       (1 pass)
    "f90f5ff9-…",  # Text Rendering    (1 pass)
    "0b0d16bb-…",  # Fire              (1 pass)
    "8d454b7b-…",  # Night City        (1 pass)
    "77a84d27-…",  # Radiance Cascades (6 passes)
]
```

Bloom Chain is a **test fixture**, `tests/fixtures/bloom_chain/`, and
`tests/test_lazy_compile.py:30` says why: *"the five-pass bloom chain is a test fixture (it left
the shipped examples once Radiance Cascades covered multi-pass)"*. Radiance Cascades is the only
multi-pass shipped example. The spec names "the Bloom Chain example" in item 3, in item 2's
falsifier, and in D2/D3/D6's worked examples — the D2/D6 prose uses are fine as illustrations,
but item 3 is a test instruction and it names an artifact that is not there.

**3b. "the four files / four entries" is wrong.** The fixture has **five** passes —
`scene`, `bright`, `blur`, `trail`, `composite` (`ls tests/fixtures/bloom_chain/passes/`). With
`scene` substituted, four are copied, so "four" is right for the copied count but the spec says
"assert the four files under `passes/`" — the host also keeps its own `main`, so the directory
holds five. Say "the four copied files beside the host's own".

**3c. The sampler names do not exist in the fixture.** The spec asserts
`bloom_bright.u_src == {"pass": "main"}` and `bloom_composite.u_glow == {"pass": "bloom_blur"}`.
The fixture's real samplers, per `grep -n "uniform sampler2D" tests/fixtures/bloom_chain/passes/*`:

```
blur.frag.glsl:      u_bright
bright.frag.glsl:    u_scene
composite.frag.glsl: u_scene, u_blur, u_trail
trail.frag.glsl:     u_scene, u_prev
scene.frag.glsl:     (none)
```

`u_src` and `u_glow` are the **pre-069** names; feature 069 W-D renamed every sampler to
`u_<source pass>` (`ai_docs/features/069_tutorial_walk_findings/00_findings.md:65`). The correct
assertions are `bloom_bright.u_scene == {"pass": "main"}` and
`bloom_composite.u_blur == {"pass": "bloom_blur"}`.

**3d. The fixture is not reachable from the `app` fixture.** `tests/conftest.py:26-34`:

```python
def seed_tmp_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    documents = project / "documents"
    documents.mkdir(parents=True)
    shutil.copytree(
        DOCUMENT_EXAMPLES_DIR / STARTER_EXAMPLE_ID, documents / STARTER_EXAMPLE_ID
    )
    return project
```

Only the starter (UV Mango, one pass named `main`) lands in `app.ui_documents`. The other five
examples live in `session.ui_document_examples`, loaded from the read-only resources dir
(`project_session.py:506-508`). So item 3's source document must be loaded explicitly — which
is what `test_default_wiring.py:38` and `test_lazy_compile.py:31` already do:

```python
_BLOOM = Path(__file__).parent / "fixtures" / "bloom_chain"   # a test fixture, not shipped
...
shutil.copytree(_BLOOM, document_dir)
document = load_document_from_dir(document_dir).document
```

**Rewrite.**

> 3. **`import_passes` end to end** (`tests/test_pass_verbs.py`, the `app` fixture): the
>    five-pass bloom-chain FIXTURE (`tests/fixtures/bloom_chain` — copied into `tmp_path` and
>    loaded with `load_document_from_dir`, the way `test_lazy_compile.py:31` and
>    `test_default_wiring.py:38` already reach it; it is not a shipped example and not in
>    `app.ui_documents`) imported into the starter document under group `bloom` with `scene`
>    substituted by the starter's `main`. Reload from disk with `_reload`; assert the four
>    COPIED files beside the host's own under `passes/`, the four entries with
>    `group == "bloom"`, the rows `bloom_bright.u_scene == {"pass": "main"}` and
>    `bloom_composite.u_blur == {"pass": "bloom_blur"}` (069 W-D's `u_<pass>` names — `u_src`
>    and `u_glow` are the pre-069 spellings and are not in the fixture), the self-read row
>    `bloom_trail.u_prev == {"pass": "bloom_trail"}`, the output `bloom_composite`, and that the
>    SOURCE document's passes still hold their own textures after `release()` of the host.
>    Falsifier: share the `Image` object instead of copying it and the source's sampler reads a
>    released texture.

`tests/test_pass_verbs.py` already has the needed machinery: the `app` fixture, `_reload(app,
document_id)` for the disk round-trip, and several tests that reload and re-assert graph entries
(e.g. `test_rename_moves_the_file_the_edges_and_the_output`, which reloads and asserts
`reloaded.passes["consumer"].uniform_values["u_src"] == PassSource("scene")`). So the item fits
the file; only the artifact it names is wrong.

### Item 4 — A rendered import is not black — WEAK

Two problems. First, it inherits item 3's fixture gap. Second, the baseline is wrong: it says
"assert the output canvas's mean is above **the starter's black**", but the starter is UV Mango,
a procedural gradient — it is not black. A mean-above-a-wrong-baseline assert can pass for a
second reason (the host's own pass contributing), which is exactly what step 7 forbids.

The materialization bug this guards is already caught by item 2(a) as a structural assert on
`ImportPlan.sources`, which is strictly sharper than a pixel mean. The pixel test still earns
its place as a consumer check (does the *rendered* document read the materialized rows) — but it
needs a falsifiable baseline.

**Rewrite.**

> 4. **A rendered import reads its bundle, not black** (`tests/test_document_graph.py`'s
>    `gl_ctx` + `_document` helpers): build a two-pass host whose `main` renders a known
>    constant, import the bloom fixture with `scene` substituted by `main`, render, and assert
>    the output canvas's red is the value the bundle produces FROM that constant — not merely
>    "above the starter", which UV Mango's gradient satisfies whether or not the bundle is
>    wired. Falsifier: drop the materialization (item 2a) and the output reads the
>    unbloomed/black value instead.

Note `test_document_graph.py` uses a module-scoped `gl_ctx` standalone context and a `_document`
helper that builds a document from raw GLSL strings — which is the right home for a constructed
host, and avoids the `app` fixture entirely.

### Item 5 — The group survives every existing verb — CATCHES

The named falsifier (`_graph_renamed` rebuilt from a fresh `PassEntry()`) is the real bug class,
and today's code is already immune for a documented reason. `project_session.py:143-150`:

```python
def _graph_renamed(graph: PassGraph, old: str, new: str) -> PassGraph:
    entries = {
        (new if name == old else name): entry for name, entry in graph.passes.items()
    }
```

It re-keys the existing `entry` object rather than constructing one, and `with_passes`'s
docstring states the principle: *"The funnel is the point; the COPY inside it still goes through
`model_copy`, so a field added to this model tomorrow survives every edit rather than silently
resetting."* Same for `_graph_without` (`:137`). So D1's "rides the per-entry salvage unchanged"
is true of the rename/delete family.

`tests/test_graph_persistence.py` is the right home: it has `_write_document` (writes
`graph.json` straight to disk) and `load_document_from_dir` + `UIDocument.save(parent, name,
rebind=False)` round-trips, with no `app` fixture. The "`graph.json` without the key loads as
`""`" sub-case is a direct `_write_document` case.

### Item 6 — `group_runs` — CATCHES

Pure function, and `tests/test_pass_strip_layout.py` is exactly its home: that file imports only
`theme` and `tiles_per_row`, has no fixtures, and already pins `tiles_per_row` against two
named falsifiers. The three sub-cases (consecutive members form a run; an outside pass splits
into two; an ungrouped pass is a run of one) are decidable from the return value, and the named
falsifier (grouping by name instead of adjacency merges the split) is the bug an implementer
would write — a `defaultdict(list)` keyed by group name is the obvious first draft and it
produces exactly that.

### Item 7 — The theme invariant — **FAIL as written**

Both halves are wrong, and the second is not fixable by rewording.

**7a. "putting `yellow_b` in `GROUP_TINTS` makes import fail" is not expressible as a test.**
The invariant D8 specifies is an **import-time `assert` in `theme.py`**, beside the SELECT
invariant at `theme.py:206`:

```python
assert COLOR.SELECT not in _accent_primaries, (
    f"theme invariant: SELECT={COLOR.SELECT} collides with an accent preset's "
    ...
```

A test cannot make that assert fire. I ran it:

```
$ uv run python -c "import shaderbox.theme as t; importlib.reload(t)"
reload ran, no assert fired (module body re-executed with ORIGINAL literals)
```

`importlib.reload` re-reads the file, so it re-evaluates the *original* `GROUP_TINTS` literal;
and mutating `theme.GROUP_TINTS` after import does nothing, because the assert already ran. To
watch the assert fire you would have to rewrite `theme.py` on disk and import it in a
subprocess — which is mutating the live working tree mid-run, the thing `conventions.md` bans
outright (*"never mutate the live working tree while…"*, the 064 race). So as specified this is
a gate that can only be "verified" by not running it.

Note also that the spec's own prose is self-contradictory here: D8 says the assert lives *"beside
the `SELECT` invariant"* (i.e. in `theme.py`) while item 7 says *"`tests/test_theme.py` breaks it
by putting `yellow_b` in the tuple"*. Those are two different mechanisms. And
`tests/test_theme.py` today holds neither — it is a GL-free file about `load_color` /
`throttle_color` bands, with no subprocess, no palette sweep, and no accent invariant (I read
it end to end; the only assert family is `load_color(0.49) is COLOR.STATE_OK` and siblings).
The "SELECT invariant" the spec points at is in `theme.py`, not in that test.

**7b. The subprocess is the wrong shape for a stable-hash claim.** The claim is "`group_tint` is
stable across processes". A subprocess comparison demonstrates it but does not *pin* it: it
passes under `zlib.crc32` and it also passes under any other deterministic function, so it
cannot tell a correct implementation from a different-but-also-stable one, and it costs a
process launch per run. More to the point, the bug it exists for is `hash()`, and `hash()` on a
`str` is salted per process — so the honest one-line gate is a hardcoded expected index, which
fails immediately under `hash()` (a salted value has a 1-in-6 chance of matching, and it changes
every run, so the test is red on essentially every invocation rather than flaky-green):

```
$ uv run python -c "import zlib; print(zlib.crc32(b'bloom') % 6, hash('bloom') % 6)"
3 5      # crc32 is 3 every run; hash() was 5 this run and differs the next
```

**Rewrite (two items, both pure, both in `tests/test_theme.py`).**

> 7. **Group tints are stable and collide with nothing** (`tests/test_theme.py`, GL-free):
>    (a) `group_tint` is pinned by VALUE — `group_tint("bloom") is COLOR.GROUP_TINTS[3]` and
>    two more names at their crc32 indices, computed once and written into the test as
>    literals. Falsifier: `hash()`, which is salted per process, so the pinned index fails on
>    essentially every run rather than flaking green (`zlib.crc32(b"bloom") % 6 == 3` every
>    run; `hash("bloom") % 6` differs per process — both verified).
>    (b) `set(GROUP_TINTS)` is disjoint from the accent primaries and from every `STATE_*` hue,
>    asserted in the test over `theme`'s own `_ACCENTS` and `COLOR`, and the tuple has no
>    duplicate (`len(set(GROUP_TINTS)) == len(GROUP_TINTS)`) — else two groups draw the same
>    colour. Falsifier: put `yellow_b` (`COLOR.ACCENT_PRIMARY`, `theme.py:150`) in the tuple
>    and (b) goes red.
>
>    D8's import-time assert stays in `theme.py` beside the SELECT one as the belt — but the
>    GATE is (b), because an import-time assert cannot be falsified from a test: `importlib.reload`
>    re-reads the original literals and post-import mutation is too late (both run), and the
>    only way to trip it is rewriting `theme.py` on disk mid-run, which `conventions.md` bans.

### Item 8 — The copilot table and `set_pass(group=)` — CATCHES

Three decidable sub-cases, and they fit `tests/test_copilot_pass_tools.py` exactly: that file
calls `app.copilot_backend.set_pass(...)` positionally and asserts on `res.table` by substring
(`assert "glow [output]: runs 12, target f4 x0.5, linear" in res.table`) and on `res.error` by
substring. So "shows `group fx` in the echoed table" is a one-line substring assert, `group=""`
clears it by the same means, and "an invalid group name is an error, not a silent no-op" is
`assert not res.ok` plus a substring of the message — the `assert ... .ok` / `.error` idiom the
file already uses.

One note for the implementer, not a verification gap: `set_pass` is positional all the way down
(`tools/passes.py:76-89` → `capabilities.py:458-469` → `backend.py:1385`), so `group` is a new
positional on three signatures; the existing tests call it positionally with nine args and will
need the tenth. That is a mechanical update, not a hole.

### Item 9 — The command is registered — CATCHES, and it is two gates not one

The spec says `test_command_registry_coverage.py` "already fails on an unrouted `CommandId`".
Verified by reading the file — it is true, and it is stronger than the spec implies. Two
independent hard gates, both strict set-equality:

```python
def test_every_command_id_has_a_spec() -> None:
    assert set(SPEC_BY_ID) == set(CommandId)

def test_every_command_id_has_a_handler(app: Any) -> None:
    assert set(app.command_callbacks) == set(CommandId)
```

So `CommandId.IMPORT_PASSES` needs **both** a `COMMAND_SPECS` row and an `app.command_callbacks`
handler; either alone is red. A third test only checks help-text membership for specs that set a
`default_chord`, so a chordless `IMPORT_PASSES` skips it. The spec's "Files touched" says
`commands.py` gets "`CommandId.IMPORT_PASSES` + palette entry" and `app.py` gets "the
`IMPORT_PASSES` command binding" — both halves are named, so the spec is consistent with the
gate. Worth making explicit in the item that the binding is the second half, since a spec row
without a handler is the easier omission.

There is no separate hotkey table: `commands.py` has one flat `COMMAND_SPECS` list of
`CommandSpec(id, label, chord, category)` rows which serves palette, chord and cheatsheet at
once (`CommandSpec(CommandId.ADD_PASS, "Add pass", _chord(K.a, K.mod_alt), C.TOOLS)` is the
whole registration of `ADD_PASS`).

### Item 10 — The modal's state resets — CATCHES

`tests/test_pass_draft.py` is the right shape to copy: it drives real App methods
(`app.open_add_pass()`, `app.close_pass_settings()`, `app.create_pass_from_draft()`) and reads
`app.pass_draft`, never constructing the draft itself, and it asserts that closing discards the
draft AND creates nothing. The analogous `ImportDraft` sequence (open → select a source →
cancel → reopen → assert the prefilled slug and `keep` substitutions) is decidable on
`app.import_draft` fields. It needs the `app` fixture (GL) because `open_import_passes` will
live on `App`.

One addition the item should carry, because it is the reset that actually has a bug shape: D10
says *"picking a host pass in a combo resets that entry point's handovers to all of its
readers"* and *"Changing the source resets the group buffer to D2's slug, the substitutions to
`keep` and the handovers to empty"*. Those are two different resets with different triggers, and
an implementation that does one and not the other leaves a stale `handovers` set pointing at a
previous source's passes — which `plan_import` then rejects (D4 rejects "a handover naming a pair
that does not read a fed pass"), so the symptom is a permanently-disabled Import button with a
confusing message. Add: *"selecting a second source after setting handovers on the first leaves
`handovers` empty, and re-picking a host pass in a combo refills it with that pass's readers"*.

### Item 11 — Smoke — WEAK

**What `make smoke` runs** (`Makefile:smoke` → `scripts/smoke.py`): ~200 frames of
`update_and_draw` against a throwaway tmp project seeded with all six shipped examples plus a
script document, in an invisible glfw window.

**Does the strip draw?** Yes, and it deliberately draws the multi-pass case —
`scripts/smoke.py` at frame 42:

```python
multi = next((i for i, u in app.ui_documents.items() if len(u.document.passes) > 1), "")
assert multi, "smoke: no multi-pass document to draw the strip with"
app.set_current_document_id(multi)
```

That reaches Radiance Cascades' six-tile strip, so the strip's draw path is exercised.

**But no document in the smoke has a group**, so D7's outline code — the one thing item 11
claims to check — never executes. The item as written ("the strip with a grouped document draws
without an assert") describes a state the smoke does not create, so it passes whether or not the
outline path works. That is the "unwired mechanism counts as absent" failure from step 7,
applied to the check itself.

The parenthetical names the real hazard correctly: the outline draws on the **parent's** draw
list while each tile is its own child window (`preview_cell` opens `begin_child`), which is the
`SetCursorPos`/jitter class the `/imgui-ui` skill covers. That hazard is worth smoking — it just
needs a grouped document to exist.

**Rewrite.**

> 11. **Smoke**: beside the frame-42 multi-pass strip, stamp a group onto two of that
>     document's passes (`session.set_pass_group(multi, name, "smoke_group")` on two
>     NON-adjacent members in `strip_order`, so both the single-run and the split-run paths
>     draw) and keep them grouped for the rest of the loop, so the outline, its label and the
>     `bordered=False` member tiles all execute on the parent draw list under the real
>     frame loop. Falsifier: the outline's `add_rect` inside a tile's child window instead of
>     on the parent's list — today no smoke document carries a group, so the group path runs
>     zero times and the step passes whether or not it works.

Also worth adding there, since it is one line and covers D7's wrap case: the smoke already
exercises the strip at one window width only. A group whose run wraps is "two runs with the
label on each" (D7) — that is `group_runs`' job and item 6 covers it purely, so the smoke need
not resize.

---

## B. Invariants with no test

For each: the guarantee, the line that will READ it, and the check that goes red if the read is
cut.

### B-0 — D10's draw wiring is ALREADY gated, and the spec does not say so

`tests/test_project_management.py:636-668` — `test_every_popup_state_has_a_draw_call`. It AST-parses
`shaderbox/ui.py`, collects names imported from `shaderbox.popups`, collects the subset actually
*called*, and asserts:

```python
unwired = sorted(imported - called)
assert not unwired, ...
assert len(called) == len(PopupState) - 1, (
    f"{len(PopupState) - 1} PopupState members need a draw call; "
```

I ran the same AST walk against today's tree: **7 imported, 7 called, `len(PopupState) - 1 == 7`.**
So adding `PopupState.IMPORT_PASSES` makes the second assert `7 != 8` and the suite goes red
until `draw_import_passes` is both imported and called in `ui.py`. The test's own comment states
the bug class it exists for: *"A state with no draw call is INVISIBLE at runtime — the popup
mutex suppresses every render, nothing draws, and the state simply persists, which looks
identical to a healthy modal."* That is precisely D10's failure mode, and it is free.

Add to the verification list:

> 12. **The modal is wired into the popup chain** —
>     `tests/test_project_management.py::test_every_popup_state_has_a_draw_call` already goes red
>     on a `PopupState` member whose draw is not called in `ui.py` (it AST-parses the call, so a
>     mention in a comment does not satisfy it). Verified: 7 called == `len(PopupState) - 1` today.

This is also the only enum-iterating consumer of `PopupState` (section C), so it is the single
place a new member is structurally checked.

### B-1 — The group survives `sync_documents_from_disk` — covered, but only transitively

`sync_documents_from_disk` (`project_session.py:550-603`) re-reads a changed dir through
`_load_one_document_from_disk` → `load_document_from_dir`, which rebuilds the graph from
`graph.json` at `document.py:902-905`:

```python
document.graph = graph.with_passes(
    {name: graph.passes.get(name, PassEntry()) for name in document.passes}
)
```

**The reader is that line**, and it preserves `group` because it carries the whole `entry` object
(the `PassEntry()` default only fires for a pass FILE with no entry — a pass file with no graph
entry has no group to lose). So a reload is safe, and item 5's "a reload reads it" sub-case
covers the disk round-trip.

The gap is narrower than "the group is lost": it is the **mid-session** reload. An import writes
`graph.json`, which bumps `document.json`'s mtime only if `document.json` changed too — and
`save_ui_document` rebaselines `_document_json_mtimes` (`project_session.py:392-398`) precisely so
the next frame does not read its own write back as external. So the import's own write does not
trigger a sync. **No gap.** Recording it as a false trail rather than a finding.

### B-2 — An import while the copilot is mid-turn — COVERED STRUCTURALLY, no test needed

`widgets/pass_list.py:165-192` wraps the whole strip, `add pass` included, in
`imgui.begin_disabled(app.copilot_turn_active)` … `imgui.end_disabled()`. An `import…` button
placed "beside `add pass`" (D10) inherits that for free.

But the **palette command** `IMPORT_PASSES` does not go through the strip. The palette route
reaches `app.command_callbacks[IMPORT_PASSES]` directly. `app.py` guards other verbs explicitly
with `if self.copilot_turn_active: return` (lines 1490, 1711, 1718, 2024) and has a
`_copilot_busy_blocked` helper at `:922` that notifies and refuses. So:

> **Add to verification.** The palette route is gated too: with `app.copilot_turn_active = True`,
> `app.open_import_passes()` does not open the modal (`popup_state` stays `CLOSED`) and pushes the
> busy notification, the way the other mid-turn-guarded verbs do. The READER is the guard at the
> top of `open_import_passes`; the falsifier is deleting it, which leaves the palette able to open
> an import dialog over a document the copilot is rewriting — a button-only guard passes today's
> suite either way.

### B-3 — The D4 rejection reaching the dialog every frame — NO TEST, and it needs one

D10 says *"The plan is recomputed each frame the dialog draws, which is how the button's enabled
state and the message stay honest"*. That is a per-frame recompute, and the failure mode is the
cheap one: compute the plan **once** on selection, cache the rejection string, and the button
then stays disabled (or enabled) after the user fixes (or breaks) the group name. The symptom is
a dialog that will not let you import with a perfectly good name.

Nothing in the verification list reads this. The reader is the dialog body's `plan_import(...)`
call; there is no pure seam to test it through, because it lives inside a draw function.

> **Add.** `tests/test_pass_import.py` (pure) owns the plan; the dialog's per-frame recompute is
> checked where `test_pass_settings_layout.py` checks the gear — monkeypatch the draw body and
> pump frames. Concretely: open the dialog with a group name that rejects (`"2bad"`), pump a
> frame, assert the draft's cached message is the rejection; set the group buffer to a valid
> name WITHOUT changing the source, pump one more frame, assert the message is now empty.
> Falsifier: compute the plan on selection only and the second frame still reports the stale
> rejection. (If the plan result is not stored on `ImportDraft`, this needs a recorder
> monkeypatched over the plan call, the `_gear_sizes` pattern at
> `tests/test_pass_settings_layout.py`.)

### B-4 — The `ImportDraft` reset on Escape — NO TEST as specified

Item 10 covers "cancelling", but the Escape path is a **different code path** from a Cancel
button, and this repo has the scar: `close_pass_settings`'s docstring says *"The one funnel both
close paths reach: Escape closes the popup before the body draws, so a commit inside the body is
unreachable on that frame."* Escape is handled in `hotkeys.py:376-388`:

```python
if app.popup_state == PopupState.PASS_SETTINGS:
    app.close_pass_settings()
elif app.popup_state == PopupState.PROJECTS and app.projects_input_owns_esc():
    ...
elif (
    app.popup_state != PopupState.SHADER_LIB_PICKER
    ...
):
    app.popup_state = PopupState.CLOSED
```

**This is the finding:** `PASS_SETTINGS` gets a named branch that calls its close funnel; every
other state falls through to the final `else`, which sets `popup_state = CLOSED` **and nothing
else**. So unless `IMPORT_PASSES` gets its own branch, Escape closes the dialog and leaves
`app.import_draft` populated — the next open shows the previous source, group buffer,
substitutions and handovers, and D10's "Changing the source resets…" never fires because the
source did not change. That is a live bug the spec's file list does not mention (`hotkeys.py` is
absent from `## Files touched`).

> **Add to `## Files touched`:** `shaderbox/hotkeys.py` — the `IMPORT_PASSES` Escape branch, so
> Escape reaches `close_import_passes` rather than the bare `popup_state = CLOSED` fallthrough.
>
> **Add to verification.** Escape, not just Cancel: with the dialog open and a source selected,
> drive the Escape path (`app.close_import_passes()` for the unit check, and the `hotkeys.py`
> branch existing for the wire) and assert `app.import_draft` is back to its initial state.
> Falsifier: rely on the `hotkeys.py` fallthrough and the draft survives Escape, so reopening
> shows the previous source — exactly the bug `close_pass_settings` exists to prevent for the
> gear.

### B-5 — Handover rows on HOST passes surviving `UIDocument.save` — NO TEST, and this is the one I would fix first

D6 writes handover rows as explicit `PassSource`s on **host** passes. `UIDocument.save`
(`ui_models.py:398-...`) rebuilds the uniform block per pass from the **live program**:

```python
for pass_name, render_pass in self.document.passes.items():
    if render_pass.program is None:
        existing = _existing_rows(dir, pass_name)
        if existing:
            meta["uniforms"][pass_name] = existing
        continue
    rows: dict[str, Any] = {}
    render_pass.seed_uniform_values()
    for uniform in render_pass.get_active_uniforms():
        ...
```

Two distinct hazards, neither covered:

**(i) A host pass with no program carries rows forward from DISK.** If a host pass has not
compiled at import time, the `continue` branch writes `_existing_rows(dir, pass_name)` — the rows
already on disk — and the handover row just written into `uniform_values` is **silently
discarded**. The host document in the `app` fixture is warmed (`conftest.py` calls
`document.render()`), so a test built on the fixture would not see this; a real host with a
never-drawn branch pass would.

**(ii) A handover row on a sampler the program does not declare is dropped.** The loop iterates
`get_active_uniforms()`, so a row keyed on a sampler that is not in the program is not written.
D4 computes handovers from `host_wiring`, which for an **uncompiled** host pass is explicit rows
only (`Document._reads_of`, verified below) — so D6's "default readers computed from
`host_wiring`" can name a (pass, sampler) pair that is invisible, or miss one that exists.

The second is the same compile-first problem D3 already solved for the SOURCE. D3 says *"after
every source pass has been compiled"* and gives the reason (`_bring_chain_online` alone stops at
the output's chain). **The spec never says the HOST must be compiled**, and `host_wiring` has
exactly the same lazy-compile hole. I measured it:

```
UNCOMPILED effective_wiring (bloom fixture, all AutoSource):
  {'blur': {}, 'bright': {}, 'composite': {}, 'scene': {}, 'trail': {}}   -> 5 "roots"
COMPILED:
  {'blur': {'u_bright':'bright'}, …}                                      -> 1 root (scene)
```

So on an uncompiled document `effective_wiring()` is empty and every pass looks like a root. For
the source that is D3's stated reason to compile. For the **host** the consequence is that D6's
reader list comes back empty, every handover checkbox is absent, and the insertion silently does
nothing — the bundle is copied but nothing reads it. The dialog would look correct and the
result would be wrong.

> **Add to D6 / D4's contract** (not merely to verification): `host_wiring` is the host's
> `effective_wiring()` **after every host pass has been compiled**, for the same reason D3
> compiles the source — an uncompiled pass answers with its explicit rows only, so a
> never-drawn host pass contributes no readers and the insertion silently hands over to nobody.
> Measured: `effective_wiring()` on the uncompiled bloom fixture is `{'blur': {}, …}` (every
> pass a root); compiled it is the real five-edge wiring.
>
> **Add to verification.** In item 3's end-to-end, give the host a second pass that reads `main`
> and has NOT been rendered; import with `main` substituted; assert after reload that the
> handover row landed on that pass (`grade.u_main == {"pass": "bloom_composite"}`). Falsifier
> (i): compute `host_wiring` without compiling the host and the reader is never offered, so the
> row is absent. Falsifier (ii): a host pass whose program is None at save time carries its
> stale disk rows forward (`ui_models.py`, the `if render_pass.program is None: existing =
> _existing_rows(...)` branch) and the handover is dropped with no error.

### B-6 — A source pass that failed to compile — NO TEST

D3 says *"A source pass whose compile fails contributes its explicit rows only, and the import
notification names it."* Two guarantees, neither checked. The notification is the kind of
spec'd-but-unwired safety step 7 calls out by name.

> **Add.** `tests/test_pass_import.py` covers the pure half: a source wiring where one pass
> contributes no edges still plans, and the plan (or `import_passes`' return) names that pass.
> The consumer half belongs in item 3's file: import a source dir one of whose `.frag.glsl` files
> does not compile, assert the import still lands the other passes AND that the notification text
> names the broken one. Falsifier: swallow the compile failure and the user gets a bundle with a
> silently-black member — `wired_pass` returns `None` for a missing edge rather than raising, so
> nothing else reports it.

### B-7 — The source is the shipped example, in the read-only resources dir — NO TEST, low risk

I checked whether anything in D5 writes to the source. D5's writes are all host-side
(`passes/<host name>.frag.glsl`, `media/<host name>/`, the host's `graph.json`). The source is
read three ways: `Pass.compile()` (D3) which reads only; `source.text` for the copied shader;
and `uniform_values` for the copy. Examples are loaded as live `UIDocument`s from
`self._document_examples_dir` (`project_session.py:506-508`), so their `source.path` points into
`shaderbox/resources/document_examples/…`.

The one hazard worth pinning: `UIDocument.save` is the funnel, and it writes to the dir it is
given — so **`save_ui_document` must be called on the HOST's `UIDocument`, never on the source's**.
The spec says "save through `save_ui_document`" without naming which, and a plausible
implementation that saves both (to "persist the compiled state") would write into the resources
dir, dirtying a tracked shipped example.

> **Add.** Import from a shipped example and assert the example's directory is byte-identical
> afterwards (mtime + content of `graph.json` and every `passes/*.glsl`). Falsifier: call
> `save_ui_document` on the source `UIDocument` and the shipped example is rewritten in the
> working tree. Cheap, and it is the only check that the import is read-only on its source.

### B-8 — The `ui_uniforms` merge keyed by hash across two documents — NO TEST

D5: *"`ui_uniforms` rows are merged from the source's `ui_state` for hashes the host does not
have, so a sampler's `texture` input type and a drag's range come along."*

The key is `get_uniform_hash` (`shaderbox/util.py:78-85`):

```python
key = f"{u.name}_{u.array_length}_{u.dimension}_{u.gl_type}"
hash = hashlib.md5(key.encode()).digest()
```

**Name and shape only — no pass, no document.** Two consequences the spec should state:

1. The merge is safe across the rename: a copied pass's `u_amount` has the SAME hash as the
   source's, so the row needs no re-keying. Good — and it means an implementation that tries to
   re-key by the new pass name would be wrong.
2. **A host row wins silently.** If the host already has a `uniform float u_amount` with a
   different tuned range, the source's row is skipped ("for hashes the host does not have"),
   and the imported pass's drag gets the host's range. That is the right call, but it is invisible.
3. The merge's effect is erased by the **prune** in the same `save`: `ui_models.py:466-471` drops
   every row whose hash is not in the live set —

   ```python
   live_rows = {get_uniform_hash(u) for render_pass in self.document.passes.values()
                for u in render_pass.get_active_uniforms() if u.name not in TABLE_UNIFORMS}
   stale_rows = [h for h in self.ui_state.ui_uniforms if h not in live_rows]
   ```

   gated on `if live:`. The copied passes must therefore be **compiled before the save**, or
   their rows are merged in and pruned straight back out in the same call. `import_passes`
   builds each `Pass` the way `add_pass` does, and `add_pass` calls `render_pass.compile()`
   explicitly (`project_session.py:915-919`) — so following `add_pass` exactly is what makes
   this work, and deviating from it loses every merged row.

> **Add.** After item 3's reload, assert a merged row survives: give the source a uniform with a
> non-default `input_type`/range in its `ui_state.ui_uniforms`, import, reload, and assert the row
> is present on the host with that input type. Falsifier: skip `compile()` on the copied passes
> and `UIDocument.save`'s prune (`ui_models.py:469`, gated on `if live`) drops every merged row
> in the same save that wrote it — the merge then passes paper review and persists nothing.

### B-9 — The outline for a group whose members are not contiguous — COVERED by item 6

D7: *"Contiguity is not a rule: when a rewire puts an outside pass between two members, the
group simply draws as two runs."* Item 6's second sub-case is exactly this, purely. The drawn
half is B-0/item 11's territory. No gap beyond the item-11 rewrite.

### B-10 — `group_slug` — covered, and the spec already pins it

D2 names a pure function and three inputs (`Bloom Chain` → `bloom`, `Radiance Cascades` →
`radiance`, `2D SDF` → `g_2d`). The verification list does not give it a numbered item, but D2's
own text says "pinned by a test over those three inputs". Worth promoting into the list so it is
not lost — it is one line in `test_pass_graph.py` and it has a real falsifier (a document named
`2D SDF` yields `2d`, which fails `_PASS_NAME_RE` and makes every import from it reject with a
message about the group name the user never typed).

---

## C. Blast radius

### C-1 — `PassEntry(` construction and `document.graph.passes` rebuilds

`git grep -n "PassEntry()"` in `shaderbox/` + `scripts/` returns **11 production sites**:

| Site | Shape | Preserves an unknown `group`? |
| --- | --- | --- |
| `pass_graph.py:164` `with_target` | `passes.get(name, PassEntry())` then `entry.model_copy(update={"target":…})` | **Yes** — `model_copy` keeps every other field. |
| `project_session.py:921` `add_pass` | `{**passes, name: PassEntry()}` | N/A — a NEW pass, group `""` is correct. |
| `project_session.py:1039` `set_pass_iterations` | `.get(name, PassEntry())` + `model_copy(update={"iterations":…})` | **Yes.** |
| `project_session.py:137` `_graph_without` | filters the dict, carries `entry` | **Yes** — delete drops the whole entry with its group, which D1 wants. |
| `project_session.py:143` `_graph_renamed` | re-keys, carries `entry` | **Yes** — this is item 5's falsifier, and it is already immune. |
| `popups/pass_settings.py:114` | `.get(name, PassEntry())` read-only for display | Read only. |
| `document.py:317` | the default one-pass graph | N/A. |
| `document.py:460`, `:574`, `:802` | `.get(name, PassEntry())` reads of `.target`/`.iterations` | Read only. |
| `document.py:904` `load_from_dir`'s `with_passes` fill | `{name: graph.passes.get(name, PassEntry()) for name in document.passes}` | **Yes** — carries the parsed entry; the default fires only for a pass FILE with no entry. |
| `app.py:1088`, `:1092` | `entry.target != PassEntry().target`, `entry.iterations != PassEntry().iterations` | Comparisons against field defaults, not constructions. **These are the sites to check**: if they are a "is this entry non-default" test used to decide whether to show something, a new `group` field is NOT in the comparison and a grouped-but-otherwise-default pass reads as default. Worth one look at impl time. |
| `copilot/backend.py:1286`, `:1331` | `.get(name, PassEntry())` reads | Read only; `:1286` is `_pass_table`, which D9 extends. |

**Nothing rebuilds an entry field-by-field.** Every mutation goes through `model_copy` or carries
the object, which is `with_passes`' stated purpose. So D1's claim that the rename/delete/target/
iterations family preserves the group is **true of the code as it stands** — the blast radius of
adding the field is genuinely one field.

Two sites outside that set still need the implementer's eye:

- **`copilot/backend.py` `set_pass`** — `_configure_pass` then `_pass_rename`. The rename goes
  through `session.rename_pass` (so the group rides along), and `_configure_pass` must use
  `model_copy` for the new `group` rather than constructing. Item 8 covers the behavior.
- **`PassDraft.entry: PassEntry = field(default_factory=PassEntry)`** (`ui_models.py:643`) — the
  add-pass draft. A new pass gets `group == ""`, which is right; but D9's modal gains a `group`
  row, and in CREATE mode that row would be editing `draft.entry.group` (a frozen model, so
  `model_copy`) rather than calling `set_pass_group` (which needs an existing pass). The spec's
  D9 says "committing writes `session.set_pass_group(...)`" and does not say what the row does in
  create mode. `tests/test_pass_draft.py` already asserts the draft's target and iterations land;
  the group row in create mode is an unstated case. **Flagging for the spec, not a verification
  item**: either the group row is hidden in create mode, or `create_pass_from_draft` carries
  `draft.entry.group`.

### C-2 — `PopupState` iteration and matching

`grep -rn "PopupState"`: the overwhelming majority are `== PopupState.X` / `!= PopupState.X`
comparisons and assignments — `app.py:557,998,1329`; `hotkeys.py:366,376,379,385,388`;
`ui.py:275,480,503`; one guard at the top of each popup module (`popups/help.py:25,31`,
`popups/projects.py:37,43`, `popups/examples.py:45,64`, `popups/pass_settings.py:42`,
`popups/emoji_picker.py:16,22`, `popups/settings.py:59,70`, `popups/lib_picker/__init__.py:46,52`);
`scripts/smoke.py:207,273,293,295,304,317`; plus tests. Opens funnel through
`app.py:1002 _open_popup`.

**There is no dispatch table.** Each popup module guards its own `draw` with an early return, and
`ui.py` has an `if/elif` chain. So a new member needs: the enum row, an `_open_popup` call, a
`draw_import_passes(app)` call in `ui.py`, and the module's own early-return guard. Three of the
four are structurally gated by B-0's AST test; the module's own guard is not, and a missing one
means the dialog draws under every other modal.

**Exactly one site iterates the enum**: `tests/test_project_management.py:664`
(`len(called) == len(PopupState) - 1`). That is B-0. `scripts/smoke.py:141` does an `isinstance`
check only.

### C-3 — `preview_cell` callers and `bordered=True`

`preview_cell` is at `ui_primitives.py:1213` with **5 call sites**:

| Caller | Affected by `bordered: bool = True`? |
| --- | --- |
| `widgets/pass_list.py:117` (the pass tile) | The one that passes `bordered=False` for members. |
| `widgets/uniform.py:192` (a sampler row's thumbnail) | No — default keeps today's behavior. |
| `widgets/document_grid.py:22` (document/example cards) | No. Note D10 reuses this via `draw_document_preview_button`, so the import dialog's cards inherit the bordered default — correct. |
| `exporters/telegram.py:672` (empty sticker slot) | No. |
| `exporters/telegram.py:691` (filled sticker slot) | No. |

Defaulting True means **zero behavior change at four of five sites**. Good.

**But the mechanism is not a draw-list rect — it is a child-window flag** (`ui_primitives.py:1265`):

```python
with imgui_ctx.begin_child(
    f"##preview_cell_{id_}",
    size=imgui.ImVec2(cell_w, cell_h),
    child_flags=imgui.ChildFlags_.borders,
    ...
```

`border_color` only *tints* that flag's border via `push_style_color(imgui.Col_.border, …)`. So
`bordered=False` means dropping `ChildFlags_.borders`, and in imgui a bordered child's content
region is inset by the border size while an unbordered one is not — so **member tiles' images and
footers would shift by ~1px relative to non-member tiles in the same strip**, and the cell's
`avail` changes, which `preview_cell` uses to size the image (`avail = imgui.get_content_region_avail()`).
D7's locked answer #3 says "the cards keep their size", and the outer size is indeed fixed by
`size=ImVec2(cell_w, cell_h)` — but the inner layout is not.

> **Add to verification (item 6's file or item 11).** `bordered=False` must not move the tile's
> contents: assert the cell's inner content region is the same with and without the border (pad
> by the border size when it is dropped), or accept the shift explicitly in D7. This is measurable
> the way `tests/test_pass_settings_layout.py` measures the gear — inside a real imgui frame with
> the app font, via a monkeypatched body recorder. Falsifier: drop `ChildFlags_.borders` with no
> compensating pad and grouped tiles' pictures sit 1px off their ungrouped neighbours, which is
> the misalignment class the maintainer reports by eye.

Also note the accent/error border must still win (D7 says so): at `pass_list.py:102-105` the
border is `COLOR.STATE_ERROR if errors else COLOR.ACCENT_PRIMARY if is_output else None`. So the
member-tile call becomes `bordered=border is not None` rather than `bordered=False` — otherwise a
grouped output pass loses its accent ring. Worth stating in D7; the spec's parenthetical says the
borders "still win" without saying how.

### C-4 — `tiles_per_row` callers

`grep -rn "tiles_per_row"`: **one production caller**, `widgets/pass_list.py:175`, plus the
definition at `:32` and `tests/test_pass_strip_layout.py:18-27`. D7's "tiles_per_row is
untouched, since gaps do not change" is correct and its blast radius is nil.

### C-5 — `git grep -n "PassEntry()"`

Full output is in C-1. In production code: `shaderbox/` 11 sites + `scripts/smoke.py:185`. In
tests: ~45 sites, all `PassEntry()` or `PassEntry(target=…, iterations=…)` constructions in
`test_document_graph.py`, `test_graph_persistence.py`, `test_canvas_presets.py`,
`test_pass_hot_reload.py`, `test_render_for.py`, `test_pass_verbs.py:81`
(`assert document.graph.passes["bright"] == PassEntry()`), `test_tutorial_build.py:243`. **All of
them keep passing** with a defaulted `group: str = ""` — including the equality assert at
`test_pass_verbs.py:81`, because a freshly added pass has `group == ""` and `PassEntry()` does too.
The remaining hits are in `ai_docs/features/069_*` prose.

---

## D. Fixture cost

**Pure (no GL, no display, no `app`):**

- item 1 `entry_points`, item 6 `group_runs`, D2's `group_slug` → `tests/test_pass_graph.py` /
  `tests/test_pass_strip_layout.py`. Both files have **zero fixtures**.
- item 2 `plan_import` → new `tests/test_pass_import.py`, GL-free as the spec says. `pass_import.py`
  importing `pass_graph` only is what makes this hold.
- item 7 (rewritten) → `tests/test_theme.py`, which is GL-free today and must stay so.
- item 9 → `test_every_command_id_has_a_spec` is fixture-free; `test_every_command_id_has_a_handler`
  takes `app`.

**Needs the `app` fixture (GL + an invisible glfw window):** items 3, 8, 10, B-2, B-3, B-4, B-8,
and the C-3 inner-region measurement. The fixture skips cleanly where there is no GL
(`pytest.importorskip("glfw")` then `pytest.skip("no GL")`), so these are not a portability
problem — but they are the slow ones: the fixture builds a real `App` per test (window, GL
context, all six example documents) and the Makefile runs `-n 8 --dist loadgroup` for exactly
that reason.

**Needs a standalone moderngl context, not `app`:** item 4 (rewritten) belongs in
`tests/test_document_graph.py`, whose module-scoped `gl_ctx` is `moderngl.create_standalone_context()`
with a comment warning that one context recipe per process is the rule. Cheaper than `app` and
the right home for a constructed host.

**The Bloom Chain reachability answer, directly:** **No — it is not reachable from the `app`
fixture, and item 3 does need it loaded from disk.** `seed_tmp_project` copies exactly
`DOCUMENT_EXAMPLES_DIR / STARTER_EXAMPLE_ID` (UV Mango, one pass), so `app.ui_documents` holds one
single-pass document. And there is no Bloom Chain in the resources dir at all (C/A-3a) — the
artifact is `tests/fixtures/bloom_chain/`, reached by `shutil.copytree` + `load_document_from_dir`
exactly as `tests/test_lazy_compile.py:31` and `tests/test_default_wiring.py:38` do it. The
`app` fixture's OTHER five examples are in `session.ui_document_examples`, loaded from the
read-only resources dir — usable as an import SOURCE (which is what D10's Examples tab is), and
Radiance Cascades is the only multi-pass one among them.

`seed_extra_document(app, new_id)` (`conftest.py:37-44`) exists for a second **project**
document, but it copies the starter again — so it gives a second single-pass document, not a
multi-pass source. An item-3 test wanting a project-tab source must copy the bloom fixture into
`app.paths.documents_dir` and call `app.session.sync_documents_from_disk()`.

---

## False trails

Things I probed that turned out fine, so nobody re-spends the time:

- **The group surviving a reload / `sync_documents_from_disk`.** Safe. `load_from_dir`'s
  `with_passes` fill (`document.py:902-905`) carries the parsed entry object, and
  `save_ui_document` rebaselines `_document_json_mtimes` (`project_session.py:392-398`) so the
  import's own write is not read back as an external change. B-1.
- **`PassEntry` being rebuilt field-by-field somewhere.** It is not — all 11 production sites
  either `model_copy` or carry the entry. `with_passes`' docstring says this is deliberate. C-1.
- **D4's "name-rule wiring does not survive the prefix".** True, and I measured it rather than
  trusting it: `wired_pass(AutoSource(), 'u_bright', 'bloom_blur', …)` is `None`. A-2.
- **D3's "exactly one entry point" for both examples.** True. Measured one root each (`scene`,
  `paint`). A-1.
- **`test_persistence_completeness.py` being tripped by a new `PassEntry` field.** No. Its roster
  keys on module FILENAMES that contain `"json.load"`, and `document.py` is already in the
  `exempt` set. A new field is covered by `test_graph_persistence.py`'s salvage tests instead.
- **`test_button_tiers.py` needing an allowlist update for the new surfaces.** No. It
  AST-walks `Path(ui_primitives.__file__).parent.rglob("*.py")` — directory-wide, so a new
  `shaderbox/popups/import_passes.py` and a new `import…` button in `pass_list.py` are scanned
  automatically. They must go through the four tiers (`standard_button` etc.), which the
  existing `add pass` button already does.
- **`test_pass_settings_layout.py` breaking on D9's new `group` row.** It asserts the modal's
  settled WIDTH equals `SIZE.PASS_SETTINGS_W` and that its height follows content bounded by
  the display height. A new row only risks the height bound, which a one-row addition will not
  reach.
- **`Image(value.texture)` losing the asset on save.** It does drop `file_details` (the
  `isinstance(src, moderngl.Texture)` branch at `media.py:120-135` leaves `FileDetails()`), but
  `_uniform_entry` re-serializes from the PIL image via `value.save(...)`
  (`ui_models.py:342`), so the asset is written from pixels and persistence is fine.
- **`Video` re-opened from its file.** Sound. `Video.__init__` needs a live path and raises
  otherwise (`media.py:185-201`), and a loaded `Video`'s path is
  `document_dir / media/<pass>/<name>.<ext>` (`document.py:243-249`), which exists on disk.
- **`make smoke` not drawing the multi-pass strip.** It does — frame 42 picks the first document
  with more than one pass and asserts one exists. The gap is only the missing GROUP. A-11.
