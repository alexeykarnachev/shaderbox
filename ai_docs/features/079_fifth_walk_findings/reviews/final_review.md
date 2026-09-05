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
and `textureCube` are not builtins in the wrong colour, they are names 460 core REMOVED, so
dropping them from the lexer's builtin list is a correctness fix rather than a new precedence
rule. Measured after vendoring: 30 orange glyphs for `gl_FragColor` plus a user's `fragColor` twice,
`gl_FragCoord` still a builtin.

Two corrections to this session's own earlier work:
- **`gl_FragData` was never a builtin.** Re-measured on the OLD library, it recoloured fine. The
  claim that both names were affected came from reading `_BUILTIN_OUTPUTS` rather than measuring
  each name.
- **`abi_probe.py` is a sixth vendored artifact** the handover list omitted; the signature-table
  gate fails without it. Reported back.
