# 079 — the final review

A convergence loop over everything the feature landed (`0e17e64..HEAD`, eleven commits, 63 files).
Each round's verdicts are written here so the loop survives a context reset.

## Reviewers

Four per round: three sonnet for the bulk enumeration, one opus where the judgement is the
deliverable.

| Agent | Model | Anchor |
|---|---|---|
| Decision coverage | sonnet | the spec's `## Locked decisions`, D1..D13, each demonstrated by execution |
| Deleted-symbol sweep | sonnet | the explicit list of symbols deleted or renamed in this feature |
| Doc-to-code fidelity | sonnet | `CLAUDE.md`, `conventions.md`, `dev_flow.md`, the `imgui-ui` skill |
| Behaviour verification | opus | the eleven commit messages' own claims, each probed |

The non-self-authored anchors: the vendored editor library (W-C's measurement is against its real
lexer output), and `make gates`' exit code.

## Round 1

### Deleted-symbol sweep (sonnet) — FAIL, one live break, fixed

`.claude/skills/shader-lab/render_document.py` imported and called `normalize_output` from the
deleted `shaderbox.scripting.outputs`. Verified independently: the import raises
`ModuleNotFoundError`, and running the script reached that line before any argument parsing.

The reason the gates missed it is the finding under the finding: pytest runs over `tests/`, which
imports the package but never the standalone helpers under `.claude/skills/`, `scripts/` and
`dogfood/`. Those are invoked by hand, so a symbol they depend on can be deleted with every gate
green.

Fixed at the root rather than the instance: the script now passes the value straight to
`coerce_uniform_value`, mirroring `behavior.py::coerce_one` after the wrapper types went, and
`tests/test_out_of_suite_scripts.py` imports every `shaderbox.*` module all 35 such scripts name.
Mutation-tested by reintroducing the bad import — the gate names the script and the module.

Everything else the sweep found was historical narration in `ai_docs/` (legitimate — a past event
does not drift) or an unrelated word. `_ENTRY_LABEL_W` is gone everywhere; no `ghost_button(` call
site or deleted-tier `button(` usage remains in code.

### Doc-to-code fidelity (sonnet) — CLAUDE.md PASS, imgui-ui PASS, conventions FAIL, dev_flow PARTIAL

Six stale claims, every one verified against the real file before fixing:

| Where | Was | Now |
|---|---|---|
| `conventions.md` script-value rule | "a bare number for a scalar; `Vec2/3/4`/`Array`/`Text` for the shaped kinds" | plain Python, shaped against the live uniform, no wrapper types |
| `conventions.md` import claim | "a real `from shaderbox.scripting import Vec2, Vec3, …` all work" | the narrowed `__import__` and the three names it admits |
| `conventions.md` exec-seam bullet | "the exec globals are the REAL builtins" + the old injected list | the real builtins with ONE substitution, `__import__` |
| `dev_flow.md` scripting map | `outputs.py` (deleted) | `api_doc.py` (real, and the docstring gate's subject) |
| `dev_flow.md` copilot map | no `tools/passes.py` | its three lazy tools listed |
| `behavior.py::ScriptBehavior` docstring | the wrapper types — and this is what `K` SHOWS the user | the plain-Python contract |

The last one matters most: it is not a comment but a live user-facing string, and it contradicted
its own module docstring 38 lines above.

What held: every `CLAUDE.md` claim (no `@staticmethod`, no `TYPE_CHECKING`, no `__future__`, the
skills list), the imgui-ui skill's tier table and its pinned imgui-bundle version, the button-tier
convention, the docstring gate, `StoppedKey`'s home, and the copilot eager/lazy map apart from the
one omission.

### A stray probe left in `tests/`

One reviewer wrote `tests/_zz_probe_d9_test.py` despite the prompt saying not to, and it failed the
gate. Removed. Its subject was D9 (the error strip's placement), the one decision with no gate
because only one App per process may drive `update_and_draw` — so its claim is worth reading in the
behaviour agent's report rather than dismissing with the file.

### Decision coverage (sonnet) — PASS, all 13 covered

Every locked decision demonstrated by execution rather than by reading. Two worth naming:

- **D9** (the error strip below the image) is the one decision with no permanent gate, because only
  one App per process may drive `update_and_draw`. The agent probed it in a real frame anyway and
  measured the strip landing at the image's true bottom, one `cell_h` below where the interaction
  button ends. That is the geometry the fix intended.
- **D8** (the viewer at full alpha) has no direct render test either; the mechanism is a
  `push_style_var` wrapping the viewer inside the outer disabled scope, unambiguous in source.

Its one FAIL — `test_editor_ffi.py` — was the in-flight re-vendor, not a 079 regression. It read
the concurrent working-tree churn as "two other Claude sessions"; it was this session, mid-vendor.

## Editor re-vendor: `6d526c6` -> `38cadbc`

W-C's remaining half is closed. The editor session reframed the ask: `gl_FragColor`, `texture2D`
and `textureCube` are not builtins in the wrong color, they are names 460 core REMOVED, so
dropping them from the lexer's builtin list is a correctness fix rather than a new precedence
rule. Measured after vendoring: 30 orange glyphs for `gl_FragColor` plus a user's `fragColor` twice,
`gl_FragCoord` still a builtin.

Two corrections to this session's own earlier work:
- **`gl_FragData` was never a builtin.** Re-measured on the OLD library, it recolored fine. The
  claim that both names were affected came from reading `_BUILTIN_OUTPUTS` rather than measuring
  each name.
- **`abi_probe.py` is a sixth vendored artifact** the handover list omitted; the signature-table
  gate fails without it. Reported back.

### Behaviour verification (opus) — PARTIAL: three real, one false positive

Probed every behavioural claim the eleven commit messages make. Three held up under my own
re-verification and are fixed; one was a timing artifact.

**REAL — the import hint advised a deleted type.** `_import_hint` fell back to `'Vec3'` when the
exception carried no `name`, and it appended itself to the import GATE's own message — so a wrong
import produced advice that recreates the error: "not Vec3 -- import it: `from shaderbox.scripting
import Vec3`". The fallback outlived the type it named. The hint now fires on a `NameError` alone,
for the name the error actually carries.

**REAL — a bare sampler key went silent.** D5 keeps sampler/block keys errors, and W-E's `_binds`
helper collapsed "not declared" and "declared but a sampler" into one falsy answer on the broadcast
path. `{'u_tex': 1.0}` was silent while `{'main': {'u_tex': 1.0}}` errored. The broadcast branch now
separates the two: nowhere declared is silent (D5's first half), declared-and-refused is an error on
that pass (D5's second half). Gated.

**REAL — `dev_flow.md`'s step 7 survived.** The commit claimed to remove the manual-check apparatus
and removed four incidental mentions while leaving the actual step standing. Step 7 is now the
verification-design guidance it mostly already was, with the go-and-look framing gone.

**FALSE POSITIVE — `gl_FragColor`.** The agent measured 12 orange glyphs and called the commits and
the banner wrong. It was reading a tree where the new `.so` had been copied in mid-run but the docs
had not caught up. Settled from the primary source rather than either measurement: `gl_FragColor`
IS in `src/lex_glsl.odin`'s builtin list at `6d526c6` and is NOT at `38cadbc`. Both measurements
are correct for their own binary. The stale roadmap banner it noticed was real and is rewritten.

**Also noted, not a defect:** the W-L geometry figure (`open` at x=50) is font-metric dependent —
56 on the agent's read, decomposing exactly as the label's width plus one `SPACE.MD`. The spec now
states the decomposition rather than the machine-specific number.

## Round 2

### Fix verification (sonnet) — PASS, all four hold

The bare sampler key errors on both paths and an undeclared key stays silent; the import hint
returns a string only for a NameError on one of the three importable names, and "" for every
gate message and every deleted name; the editor's 100 exports match the binding and the vendored
signature table in both directions, with zero argtype mismatches.

Two residual observations, both checked:

- **`abi_probe.py` cannot run from its vendored location** — its `ROOT` is the upstream repo's
  layout, one directory higher than ours. Not a defect: it is vendored as the reference table the
  signature gate PARSES, and that gate already says so in as many words, stopping its exec above
  the `CDLL` line for exactly this reason. Left alone.
- **`dev_flow.md`'s Maintainer habits still said to "confirm the behavior in the actual app (run
  it)"** — an instruction to the agent to go and look, which is both the gating that went and
  something this box cannot do. Rewritten to demand a probe that would fail if the claim were
  false.
