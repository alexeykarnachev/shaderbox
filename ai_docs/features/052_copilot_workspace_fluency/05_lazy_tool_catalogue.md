# Slice 0 — Lazy tool catalogue (the parked D5 lever)

Infrastructure prerequisite: this slice pays for the new tool surface. Independent of slices 1-4 —
it can land first (recommended) or the new tools ship eager and accept the token tax.

## Why now

Constraint C-1 (`00_grounding.md`): every tool is `eager=True`; `registry.specs_for(names)` — the
lazy loader — exists but is **never called**. Feature 020/027 parked this "D5" lever, but its own
stated trigger was *"when `eager_specs()` exceeds ~16"* (`18_render_publish_tools.md`) — and the
registry is **already at ~24 eager tools today**, so the trigger is ALREADY BLOWN before this feature
adds anything. Slice 0 is overdue, not gold-plating. Adding the slice-2/3/4 tools makes it acute: the
tools block is re-billed on EVERY iteration (skill §6 — billed input is the SUM across iterations), and
the rare ops (`bind_media`, `unbind_media`, `rename_node`, `duplicate_node`, `set_canvas_size`,
`delete_lib_file`, `import_node`) dilute attention from the hot rules on every turn where they aren't
used.

**Reclaim tokens TODAY:** 7 of the current ~24 eager tools are the cold Telegram/YouTube integration
tools (6 Telegram: `set_telegram_token`, `telegram_connect`, `list/select/create/delete_telegram_pack`;
1 YouTube: `set_youtube_credentials`) — re-billed every turn, used almost never. Demoting them to lazy
in this slice is the single fattest win, independent of the new tools.

## Goal

An eager CORE of hot tools carried in `tools=` at turn start, plus a lazy LONG-TAIL the model pulls in
only when a turn needs it — so a texture/canvas/import turn loads those schemas, and a pure-GLSL turn
never pays for them.

## Design shape

- **Eager core** = the per-turn-hot tools (read_shader / edit_shader / write_shader / set_uniform /
  grep / read_lib / probe_render / the script trio / render + the working-set reads). These stay
  `eager=True`.
- **Lazy long-tail** = the new slice-2/3/4 tools + arguably the integration tools (Telegram/YouTube
  connect + pack CRUD are already cold — a candidate to demote and reclaim tokens TODAY).
- **The load mechanism (LOCKED — mechanism (a), the explicit model-visible path):** one eager
  `load_tools(names)` meta-tool the model calls; the engine injects `specs_for(names)` into `tools=`
  for the rest of the turn. A lightweight CATALOGUE block in the prompt (name + one-line each, RARE
  volatility) tells the model what's loadable. This mirrors how the coding-agent harness itself does
  deferred tools. **Rejected: auto-inject by intent** (pre-load a group when the working set signals
  it) — it's a hidden heuristic the model can't see, exactly the shape skill §4 cautions against; a
  model-visible explicit load is the solid choice even at the cost of one round-trip.
- **Turn-scoped loaded set**: once loaded, a tool stays in `tools=` for the remaining iterations of
  that turn; the next turn resets to the eager core.
- **Serialization MUST sort (round-1 correction).** `eager_specs()` returns dict-insertion order and
  `specs_for(names)` returns the model's `names` order (`registry.py:36,38`) — neither sorts today, so
  `load_tools(["b","a"])` vs `["a","b"]` would produce different `tools=` bytes → different cache key.
  Sort the eager+loaded UNION by name at assembly and pin it with a byte-stability assert. (The earlier
  "sorted" claim asserted a property no code provided — this makes it real.)
- **The mid-turn load busts the prefix cache ONCE — acknowledged tradeoff, not free.** `tools=` sits at
  the FRONT of the cached prefix (before system + history), so injecting a tool mid-turn invalidates
  the cache for the whole system+history on that ONE iteration (a full uncached re-bill of the turn so
  far). This is unavoidable for any lazy scheme; it is net-positive only because the eager saving
  applies to EVERY turn while the bust is paid once, on the rare turn that needs the tool. Measure it
  (below), don't assume it.

## Out of scope

- Per-tool token accounting UI — measure with `scripts/token_probe.py` (cold AND warm — skill §6), not
  a live gauge.
- Removing any eager tool to "make room" — this slice adds the lazy path, it does not re-tier the
  existing hot core beyond optionally demoting the already-cold integration tools.

## Files touched

- `copilot/agent.py` — build `tools=` from `eager_specs()` + the turn's loaded set; handle the
  `load_tools` call (mechanism a) by extending the loaded set + continuing the loop.
- `copilot/tools/registry.py` — `specs_for` already exists; add the loaded-set tracking + the
  `load_tools` handler.
- `copilot/tools/base.py` — flip the new + integration tools to `eager=False`.
- `copilot/prompt.py` — the lazy-tool CATALOGUE block (RARE volatility; names + one-liners).

## Manual verification

- **Falsifier (lazy actually saves tokens):** `token_probe.py` a pure-GLSL turn BEFORE and AFTER — the
  iteration-0 `tools=` byte count drops by the demoted tools' schema size. A turn that needs media
  loads them (assert the media tool appears in `tools=` only AFTER `load_tools`).
- **Falsifier (loaded set persists in-turn, resets next turn):** load media tools mid-turn, assert
  they're in `tools=` for the next iteration; start a fresh turn, assert `tools=` is back to the eager
  core.
- **The load-time cache bust is measured, not assumed:** a WARM-turn probe that loads a tool mid-turn
  reports the cache-miss cost of that one iteration; confirm the per-turn eager saving (over a realistic
  turn mix) exceeds the amortized bust. A cold vs warm probe gives opposite verdicts — measure both
  (skill §6).
- **Byte-stability (the sort):** assert `load_tools(["b","a"])` and `load_tools(["a","b"])` produce
  identical `tools=` bytes (the union is sorted by name). Without the sort this test goes red.

## Verified-safe invariant (do NOT defensively "fix")

- **No orphaned tool_call/tool_result across turns.** History collapses to one NL `TurnSummary` at
  commit (`agent.py:402`, `_commit_turn`) — raw `tool_call`/`tool_result` pairs are never replayed — so
  a next-turn eager reset cannot orphan a call to a now-unloaded tool. Within a turn, `tool_result`
  pairs by id and does not require the tool to be present in `tools=`. Recorded so it isn't guarded
  against speculatively (round-1 reviewer confirmed against the code).
