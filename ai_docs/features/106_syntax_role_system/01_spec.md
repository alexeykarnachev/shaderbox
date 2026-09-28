# 106 — One syntax role system, and a palette that can actually be swapped

**STATUS: SHIPPED.** D1-D8a implemented and gated. D9 (a light theme) is deliberately out
of scope and carries its own trigger.

**Two requirements, and the second is why the first matters.**

1. A colour means a ROLE, and the role means the same thing in every language and on every
   surface.
2. **Changing the theme means editing ONE table.** Not hunting literals, not touching the
   kind tables, not knowing which of five surfaces reads which function.

This is not about which hex any role gets. It is about there being one answer per role,
named for the role, and one place where roles become colours.

## What is true today, measured

The layering is already RIGHT in principle, and `theme.py` states the contract itself:
*"a palette swap carries it, which is the whole contract of `_P` being the only place
literal colours live."* `_P` is a named palette (gruvbox-hard) and nothing outside
`theme.py` references it. That is the design to finish, not to replace.

`kind_color` / `kind_slot` (`syntax_colors.py`) feed FIVE surfaces, so a colour choice is
app-wide rather than editor-local:

| surface | reads | file |
|---|---|---|
| editor text | `kind_slot` -> pushed span / word class | `tabs/code.py:418,476,504` |
| completion popup rows | `kind_slot` | `tabs/code.py:901` |
| uniform panel | `kind_color` | `widgets/uniform.py:295` |
| graph canvas node rows | `kind_color` | `widgets/pass_graph.py:387` |
| editor palette | `editor_palette(language)` | `app.py:1872` |

**MEASURED, and the phrasing matters because two earlier readings of this got it wrong in
opposite directions.** Walk every kind against a palette and you find disagreements — 2 in
the GLSL palette (`PY_DEFINITION`, `PY_DECORATOR`) and 3 in the Python one
(`ENGINE_UNIFORM`, `PASS_SAMPLER`, `WIRABLE_SAMPLER`). Walk each kind against ITS OWN
language's palette and you find none.

Both statements are true and only the second describes what the app draws, because a buffer
is one language. **But the qualifier is load-bearing and has to be said every time**: an
unfiltered walk reports five failures, and a filtered walk that does not say it filters is
the shape of error that produced a false finding earlier in this work. The filter is
`kind.name.startswith("PY_")`, which `test_intel_sources.py:172` already applies.

So: nothing renders the wrong colour today, and the safety of that rests entirely on one
language per editor — which is what D5 gates.

## The defects

- **D-1. The palette is NOT swappable, and the contract saying it is has already been
  broken.** FOUR hardcoded RGB tuples sit outside `_P` at `theme.py:111-114`, plus one more
  at `:168-170` — `(250/255, 189/255, 47/255, 0.18)` is gruvbox yellow written longhand,
  beside the `_P["yellow_b"]` it duplicates. Each is its neighbour's colour at 0.18 alpha
  and should be `fade(...)` of it. A palette swap leaves all five behind, tinting the app
  with the old theme's accents.
- **D-2. `SYN_PY_DEFINITION` is the only role token named after a LANGUAGE.** Every other
  names a role: `SYN_KEYWORD`, `SYN_BUILTIN`, `SYN_IDENT`, `SYN_OUTPUT`. A Python `def`
  name and a GLSL function declaration are the same role — *a name this buffer declares*.
  Two tokens for one role is how two colours for one role begin.
- **D-3. `PY_DECORATOR` draws `SYN_NUMBER`.** Not a decision; a free token whose value
  happened to be unused. Whoever next changes the number colour silently changes decorators.
- **D-4. There is no `type` role.** `GLSL_TYPE` borrows `SYN_KEYWORD`, so `vec3` and `if`
  are one colour, and a Python annotation has no role to map to.
- **D-5. Slots 7 and 8 each carry TWO colours** (7 is `#83a598` in GLSL, `#fabd2f` in
  Python; 8 is `#8ec07c` and `#d3869b`). That is the ONLY reason `editor_palette` takes a
  language. The per-language palette is a consequence of role duplication, not a feature.

### The second palette, found by probing rather than reading

`shaderbox/resources/graph_canvas/canvas.theme` is a WHOLE SECOND PALETTE — 27 literal
RGBA values, parsed by `graph_canvas/ffi.py::parse_theme`, loaded by
`widgets/pass_graph.py::canvas_theme`. A `_P` swap does not touch one byte of it.

It is not a duplicate by accident: its own comments say the port colours are taken from the
editor's syntax colours *"so a uniform on a node is the colour its name is in the code"*.
That is this feature's consistency requirement, maintained BY HAND across two files.

**It has NOT drifted the way an earlier draft of this spec claimed**, and the correction
matters because that claim was the argument for generating the file:

| field | `canvas.theme` | claims to be | `theme.py` | |
|---|---|---|---|---|
| `input` | `#8ec07c` | `SYN_PASS_SAMPLER` | `#8ec07c` | exact |
| `output` | `#fe8019` | `SYN_OUTPUT` | `#fe8019` | exact |
| `control` | `#9ea3b8` | `SYN_IDENT` | `#ebdbb2` | differs, DELIBERATELY |

The first two are 4-decimal roundings that round-trip perfectly. The earlier reading
truncated instead of rounding and reported two false drifts — the same class of error as
the unstated filter above, in the same spec, and it is why every number here is now
re-derived rather than carried forward.

`control` genuinely differs, and the file's own comment explains why: desaturated and cool
on purpose, sitting 71 degrees of hue from the nearest kind colour, because the cream it
replaced sat 18 degrees from the script green and made a plain row read as a script-driven
one. That is a tuned exception, not an unmaintained copy.

Review found two real drifts the table missed — `border` `#504944` against `bg_3`
`#504945`, and `state_hovered` `#928374`, which is `gray` rather than the `bg_4` it claims.

**So the honest problem is not "the mirror rotted". It is that the file picks `_P` entries
for measured reasons and nothing records WHICH entry**, so a palette swap silently keeps
gruvbox values that were chosen against gruvbox greys.

## Design decisions

- **D1. Three layers, and a theme change touches only the first.**
  `_P` (named palette: what colours exist) -> `ROLE_COLOR` (what each role looks like) ->
  `_KIND_ROLE` (what each kind IS). A new theme replaces `_P` and nothing else. A
  re-mapping of roles to colours touches `ROLE_COLOR` and nothing else. The kind table is
  about meaning and should almost never change.
- **D2. Roles are language-neutral and named for what a name IS.** The set, derived from
  what the kinds need: `keyword`, `type`, `builtin`, `declaration_type`,
  `declaration_function`, `decorator`, `member`, `ident`, `number`, `string`, `comment`,
  `operator`, plus the four that are about THIS APP's domain rather than a language —
  `engine_uniform`, `script_uniform`, `pass_sampler`, `output`. Sixteen.

  `type` and `declaration_type` are different roles and both are needed: `vec3` at a USE
  site is the language's own type, while `Behavior` at its `class` statement is a type this
  buffer declares. Same question a `builtin` / `declaration_function` pair answers for
  callables.

  **`decorator` is its own role**, added because review found `PY_DECORATOR` fitting NONE of
  the others: it is not a declaration (it names something declared elsewhere), not a builtin,
  and not a type. It was the one kind that would fail the "every kind has a role" gate on day
  one. It is language-neutral in principle — an annotation-like marker attached to a
  declaration — even though only Python produces one today.
  **Revisit if** a third language needs a role none of these express.
- **D3. A TYPE declaration and a FUNCTION declaration are two roles.** Maintainer decision:
  they must differ, as in any other editor. So the role set carries `declaration_type` and
  `declaration_function` rather than one `declaration`.

  **The ROLES are language-neutral; the SPLIT happens in the producer, and it has to.**
  An earlier draft said "in the role table, never in the producer" — review showed that is
  not buildable: `ROLE_COLOR` is keyed by role and `_KIND_ROLE` by kind, and one kind
  (`PY_DEFINITION`, `intel/python.py:211`) is emitted for both cases, so there is nothing to
  key two roles on. The producer must emit two kinds. It already branches on
  `isinstance(node, (Function, Class))` one line earlier, so this is a two-line change.

  That does NOT make the distinction language-specific, which is the point the earlier
  phrasing was reaching for: the roles are shared, so a GLSL `struct` name would map to
  `declaration_type` the day GLSL grows one.

  **Say plainly that it does not have one today.** There is no struct parsing
  (`intel/members.py`: "no user structs in any shader this app ships"), so
  `declaration_type` has exactly ONE producer — which is the shape D2 exists to avoid, and
  is acceptable here only because the role is defined by what a name IS rather than by which
  language emitted it.
- **D4. `self`/`cls`/dunders map to `builtin`**, not `keyword` — they are names the language
  provides, which is what `builtin` means, and it keeps `keyword` meaning "reserved word".
- **D5. The per-language palette is DELETED, and a role's class is GLOBAL.** This reversed
  twice and the history matters, because each reversal was a measurement rather than a
  preference. Draft one said delete it. Draft two measured the budget -- eight roles needing
  a host class against three -- and kept it as a forced decision. The editor library then
  widened `Theme.syntax` to `[16]Color`, so classes 10-15 exist and 1-9 did not move.

  With fifteen classes, every role gets its own number and a class means ONE thing in every
  buffer. `editor_palette()` takes no argument. The per-language remapping existed only to
  fit eight roles into three slots and has no other justification, so it goes.

  **The premise it rested on is now gated rather than trusted.** The reuse was sound only
  because one editor holds one language; upstream confirms the language is per handle, one
  lexer over the whole buffer, with no embedded-snippet, doc-view or diff mode anywhere in
  the architecture. That is no longer load-bearing here -- but the gate stays, because the
  SECOND language selector (`tab.kind`) is still independent of the first and nothing
  asserts they agree.
  **Revisit if** a buffer can ever hold two languages, which would make a global class
  number wrong rather than merely unnecessary.
- **D6. Slots are DERIVED from role colours, never hand-written.** Roles sharing a colour
  share a class; the lexer's own 1-6 are reused where the colour already matches, so a host
  class is spent only on a role the lexer has no colour for. Hand-written slots are how 7
  and 8 came to mean two things.

  **The budget now fits with room.** Eight roles need a host class -- `type`,
  `declaration_type`, `declaration_function`, `decorator`, `engine_uniform`,
  `script_uniform`, `pass_sampler`, `output` -- against nine available (7-15). No role is
  collapsed and nothing the maintainer asked for is given up.

  A class past 15 is refused rather than clamped, so a ninth host role fails at the call
  instead of drawing a colour the library invented. The ceiling is counted from the mirrored
  enum, never written as a literal.
- **D6a. The four `GRAPH_PORT_*` colour tokens are DELETED, not allowlisted.** Review
  flagged them as surviving a hue swap while breaking a light theme, because `_muted` keeps
  the palette's HUE and substitutes a hardcoded saturation and lightness. Both true -- but
  MEASURED, they have **zero consumers**: nothing in the app reads them. `canvas.theme`
  superseded them and says so in its own comments, explaining that it deliberately does not
  use them because on a node row they "land between 0.25 and 0.39 on every channel and every
  row reads as the same grey-brown". Dead tokens do not need an allowlist.

  This matters beyond the deletion: the headline gate would have reported them as leaks and
  sent an implementer to fold them into `_P`, destroying a derivation that was correct. **A
  gate that flags a correct thing is worse than no gate**, so the gate's domain is the
  tokens something actually draws.
- **D7. Every literal colour the app DRAWS lives in `_P`.** The four `_ACCENTS` alpha-fill
  tuples fold back via `fade()` of the `_P` entry they duplicate. `notifications.py`'s
  `_DEFAULT_COLOR` is a module-level snapshot of `COLOR.STATE_OK` taken at import and used
  as a default ARGUMENT, so it freezes at whatever the palette was when the module loaded —
  it becomes a call-time read. Gated, because the prose contract already existed and did
  not hold.
- **D8. The canvas palette names a `_P` ENTRY per field, not a role.** An earlier draft
  said "generate the role-derived fields, leave the tuned ones alone". Review showed that
  split cuts through fields that are BOTH: `control` is named as `SYN_IDENT` and is
  contrast-tuned away from it, and `text`/`text_dim`/`pin` mirror `fg_0`/`fg_2`/`fg_1` with
  comments recording the measured ratios (4.79, 4.44) that chose which `fg`. Generating
  those from a role would overwrite documented decisions and regress the bugs the comments
  say they fixed — and the first exception would be `control`, the field the old D8 cited
  as its own motivation.

  **The thing a swap must carry is WHICH PALETTE ENTRY, not which role.** Every field picks
  a `_P` entry for a measured reason; the reason is the tuning and stays in the comment, and
  the entry is what a new palette redefines. So each field is emitted as its `_P` entry, and
  the gate asserts the file's value equals that entry. `control` stops being ambiguous
  because it is pinned to an entry rather than to a role it deliberately differs from, and a
  light theme re-tints every field without anyone re-measuring — which is the actual ask.
  **Revisit if** the library grows a way to express a role rather than a literal.
- **D8a. `PY_API` moves to `builtin`, for the same reason `self` does.** Review found the
  spec applying one argument to two identical cases and only naming one: `PY_API`
  (`ScriptContext`, `Vec3`, the names the ENGINE provides) currently draws as `keyword`,
  which is the same category error as `self` drawing as `keyword`. Both are names the
  environment supplies rather than reserved words. Fixing one and not the other is how the
  table drifts back.
- **D9. A light theme is OUT OF SCOPE and the spec says why.** Swapping `_P` carries the
  hues; it does not carry the assumptions. `modal_window_dim_bg` is a fixed black veil,
  `_muted()` takes an ABSOLUTE lightness (so a fill tuned to sit brighter than a dark
  surface would sit darker than a light one), six alpha constants are tuned for
  alpha-over-dark, and `canvas.theme`'s contrast measurements are against dark greys.
  Review measured the full extent: **34 `fade()` call sites with hardcoded alphas** from
  0.08 to 1.0, plus `ACCENT_TINT_ALPHA` and `GROUP_FILL_ALPHA`, every one a contrast
  assumption against a dark ground that `ROLE_COLOR` does not reach. And `theme.py`'s
  import-time invariant block asserts hue distinctness between roles and accents, so a new
  palette can HARD-FAIL at import before the app draws — arguably correct, but it means
  "replace `_P` and nothing else" needs the caveat that the palette must pass that check.

  **This feature makes a dark-palette swap work and names light as unfinished** rather than
  claiming a generality it has not built. Saying "one table" without this caveat would be
  the claim that fails the first time someone tries it. **Trigger:** the maintainer wants a
  light theme — at which point `_muted` takes the surface it must contrast against, and the
  alphas become roles rather than literals.

## Files

- `shaderbox/theme.py` — `_P` stays the only literal-colour home; the four `_ACCENTS` alpha
  leaks folded in via `fade()`; `ROLE_COLOR` added; language-prefixed `SYN_*` tokens and the
  four dead `GRAPH_PORT_*` colour tokens removed.
- `shaderbox/syntax_colors.py` — `_KIND_ROLE` becomes the hand-written table;
  `kind_color`/`kind_slot` derive from it; `editor_palette()` loses its parameter.
- `shaderbox/intel/symbols.py` — kinds added only if D3/D4 need them.
- `shaderbox/intel/python.py` — `PY_DEFINITION` splits into two kinds at the existing
  `isinstance(node, (Function, Class))` branch (D3).
- `shaderbox/app.py` — `get_session` drops the language argument to `editor_palette`.
- `shaderbox/notifications.py` — `_DEFAULT_COLOR` becomes a call-time read (D7).
- `shaderbox/resources/graph_canvas/canvas.theme` — each field names its `_P` entry (D8).
- Tests: `test_intel_sources.py`, `test_semantic_highlight.py`, plus the palette-swap gate.

## Gates

Each broken and watched to fail before it is believed; the commit names the break.

- **A palette swap carries everything the app draws.** Replace `_P` wholesale with a probe
  palette in which every entry is distinct and unmistakable, then assert NO `COLOR.*` value
  and no generated canvas field reports a colour outside it.

  **It must REWRITE THE SOURCE and import in a SUBPROCESS.** MEASURED: `_P` -> `COLOR` ->
  `_KIND_COLOR` is three import-time copies, so mutating `_P` in-process changes nothing and
  `importlib.reload` re-executes the literal and loses the swap. The obvious monkeypatch
  construction is a no-op that passes -- a gate that never makes contact, which is the
  failure this project has hit most. Say so in the gate's own text. This is D7/D8's gate and the
  one that makes "easy re-theming" checkable rather than claimed. It must reach
  `notifications._DEFAULT_COLOR` specifically, since that one is an import-time snapshot and
  a gate that only walks `COLOR.*` would miss it. Break by restoring one hardcoded tuple.
- **Every `canvas.theme` field equals the `_P` entry it names** (D8). Break by editing one
  field. This is what makes a palette swap reach the canvas; it does NOT check the tuning,
  and the gate's own text says so, because which entry a field picks is a measured decision
  no checker can re-derive.
- **A slot means ONE thing per language.** The 7/8/9 reuse is safe only because one editor
  holds one language (D5). Assert `get_session` fixes the language at creation, and that
  every kind's slot resolves under ITS language's palette. Break by feeding a Python kind
  through the GLSL palette — which is the error that produced a false finding earlier in
  this work and is worth a test rather than a memory.
- **Every `SymbolKind` has a role**, enumerated from the enum. Break by adding a member
  without one.
- **Every role has a colour and a slot**, enumerated from the role type, not from a list
  written beside it.
- **A kind's slot draws what `kind_color` reports**, walked over every kind. This is the
  invariant that keeps the five surfaces agreeing and NO test asserts it today. Break by
  pointing one kind's role at a different colour.
- **No role token is named after a language** — a name check over the role type, with its
  own text saying it covers naming and not mapping.
- **A tab's KIND and its language agree.** Review found a second, independent language
  selector: completion and the Python word-class feed dispatch on `tab.kind == "script"`
  (`code.py:724,779,1190`) while the palette is chosen from the PATH (`app.py:1866,1871`).
  Nothing asserts the two agree. They do today — the only `kind="script"` site uses
  `script_path_for`, always `.py` — but the 7/8/9 reuse rests on it, and it is what breaks
  silently when someone adds a fifth tab kind. Assert
  `(tab.kind == "script") == (language_for_path(tab.path) == Language.PYTHON)` for every
  open tab. Break by opening a `.py` under a non-script kind.

## Out of scope

- **Matching any particular external editor's theme.** The roles are ours; the colours are
  the theme's. **Trigger:** none — chasing another editor's highlight groups is what
  produced the two language-prefixed tokens this feature removes.
- **The GLSL lexer's own slots 1-6.** The editor library owns them.

## Open questions for the user

None. The one that was open -- whether a class name differs from a function name -- is
answered: they differ, as in any other editor (D3).
