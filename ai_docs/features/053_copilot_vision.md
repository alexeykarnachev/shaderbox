# 053 — Copilot vision (the agent's real eye)

Give the render-blind copilot a *good* look at its own output — the primary channel through which the
render's content enters the agent's token stream. A first cut already shipped (`probe_render` appends a
one-shot `describe_image` read, commit `6151ca4`); this feature makes that channel actually good:
intent-grounded, anti-sycophantic, temporal-aware, cost-disciplined, and user-configurable with live
capability validation.

Grounded in the actor-model (`/copilot-llm-agent-design`): the copilot **knows its intent** → passes it
as DATA (corollary-1, don't make it hope the eye guesses); the vision model **reports what it sees on the
pixels** and never a success/beauty verdict (asked "did it succeed?" it synthesises + sycophants; asked
"what do you see?" it copies from the image); the **copilot owns the "intent met?" judgment** (it has both
the intent and the observation). Judge = copilot, eye = witness.

## Goal

1. **Intent-grounded look.** `probe_render` gains a `look_for` param — freeform "what I'm building / what
   to check". The vision call is grounded in it instead of a generic "inspect this".
2. **Anti-sycophantic, two-tact vision prompt.** Tact 1: an unbiased baseline read (coherence /
   orientation / framing / text / artifacts) — always. Tact 2 (only with `look_for`): answer it as an
   OBSERVATION under a default-NO stance ("if you can't clearly see it, say it's NOT there"). Never a
   verdict of success or beauty.
3. **Temporal sight.** For a node that ANIMATES, send a 3-frame contact sheet (one composite image) so the
   eye judges the motion arc, not one arbitrary instant. STATIC → single frame. One vision call either way.
4. **Cost discipline + crisp output.** Terse fixed-shape plaintext (not rambling bullets, not JSON). Cache
   the read by frame-content hash (reuse the no-op detection). On-demand only — never in the per-edit
   auto-probe.
5. **User-configurable model + enable, in Settings.** `vision_enabled` + `vision_model` persisted on
   `CopilotIntegration`, surfaced in Settings → Copilot, pushed onto the live config.
6. **Live capability validation.** On setting a vision model, the engine checks — free + instant, via
   OpenRouter `/api/v1/models` `architecture.input_modalities` — whether it accepts images, and shows a
   live badge (checking / supports vision / no image input / couldn't verify). No silent fail at first probe.

## Out of scope (each with a trigger)

- **A "you haven't looked yet" fact before ending a visual turn.** Tempting, but it's a conscience-guard
  that fails the better-model test (a strictly better model looks when appropriate; the tool description
  already teaches it). **Trigger:** a dogfood trace where a capable model repeatedly reports a visual task
  done without ever probing → revisit as a fact-on-channel (engine rides "no visual look this turn" on the
  turn-end result), never a prompt plea.
- **A dropdown / autocomplete of vision-capable models in Settings.** v1 keeps the existing freetext
  "Model"-style field + a live validity badge (consistent with the main model field). **Trigger:** users
  mistype model ids often enough that the badge alone doesn't cut it.
- **A billed tiny-image probe to validate.** The `/models` modality metadata is OpenRouter's authoritative
  "accepts image input" signal — free + instant. **Trigger:** a model advertises image input in metadata
  but rejects our actual image payload (would show as a probe-time failure the badge said was fine).
- **Structured JSON vision output via forced tool-call.** Terse fixed-shape plaintext is enough and
  cache-friendly; gpt-4o-mini on json_schema is finicky. **Trigger:** the copilot mis-parses the plaintext
  read often enough to warrant a schema.
- **Vision in the per-edit auto-probe.** Cost. On-demand `probe_render` only. **Trigger:** never, unless
  vision cost drops ~10x.

## Design decisions (locked)

1. **`look_for: str = ""` on `probe_render`** (and threaded to `describe_image(png, hint, is_strip)` as the
   hint). Intent as data — the copilot already synthesises this string when it decides to look.
2. **Two-tact vision system prompt.** `_VISION_SYSTEM` restructured: (1) always give the baseline
   observation read; (2) if a `look_for` is present, answer it explicitly, defaulting to NO/absent when not
   clearly visible. Both tacts are OBSERVATIONS — the prompt forbids "looks good/bad" and "the shader is
   correct". This is the anti-sycophancy guard, built into the prompt not a post-check.
3. **Copilot owns the verdict.** `prompt.py`: "probe_render's visual read is a witness, not a judge — it
   can be wrong on fine detail; YOU decide whether your intent was met; beauty/readability stays the user's
   eye." (Already partly there from `6151ca4`; extend for the witness/judge split + `look_for` usage.)
4. **Terse fixed-shape output.** The vision prompt asks for a compact labelled line
   (`coherent: … | orientation: … | framing: … | text: … | artifacts: … | look_for: …`), capped by
   `copilot_vision_max_tokens`. Appended to the facts line as today.
5. **Contact sheet for animation.** `_probe_png` renders frame0 at `t`; renders frame1 at
   `t + render_facts_motion_t`; if they differ beyond `_MOTION_EPS` (the existing motion threshold, on the
   RAW frame bytes) it renders a 3rd frame and tiles the three horizontally into one PIL image (thin
   separator gutter) → returns `(png, is_strip=True)`. If static, returns `(frame0_png, is_strip=False)`.
   `describe_image` gets `is_strip` and, when set, tells the eye it's a left→right time strip so it reads
   motion, not three unrelated scenes. (Motion is re-derived here at probe size, NOT shared with
   `_render_facts_for` — `probe_render` calls `_render_facts_for(node, t)` WITHOUT `motion=True`, so no
   facts-line verdict can contradict the strip; if that ever changes, both must share one computation and
   the same `_MOTION_EPS`.)
6. **Frame-hash vision cache — keyed on frame AND intent.** `describe_image` results are cached on the
   backend keyed by `(fast hash of the RAW frame bytes, look_for, is_strip)` — NOT the frame alone: the
   read depends on `look_for`, so two probes of the same frame with different `look_for` MUST miss (else
   the second returns a stale read that ignores the new intent — the exact bug the "look_for reaches the
   eye" check guards). Raw frame bytes (the already-trusted deterministic value the no-op detector uses),
   not PNG bytes. Bounded (per-node latest, small). Note: video/script-bound nodes advance a frame per
   render, so a same-`(node,t)` reprobe legitimately produces new pixels → a correct cache MISS (never a
   wrong hit), just no cache benefit there.
7. **Vision config read via LIVE GETTERS, mirroring the main model — NO `apply_limits` push.** The main
   model + key are already read live through getters the client holds (`get_model` /
   `get_api_key`, sourced from `integrations_store.copilot`); vision follows the SAME seam, not the
   `COPILOT_CONFIG`-push the numeric limits use. So: `vision_enabled: bool = CopilotConfig.
   copilot_vision_enabled` + `vision_model: str = CopilotConfig.copilot_vision_model` on
   `CopilotIntegration` (defaults MIRRORED from CopilotConfig — config stays the single source of truth for
   defaults, exactly like the limit fields at integrations.py:61-68; no duplicated literal). The client
   gains `get_vision_enabled` / `get_vision_model` closures reading those fields; `describe_image` reads the
   getters, not `COPILOT_CONFIG`. `CopilotBackend` gets a `vision_enabled: Callable[[], bool]` to skip the
   PNG render when off. **Settings edit = `save()` only** (the getter picks it up on the next probe) — NO
   `apply_limits` call, deleting the "persisted-but-not-live-until-restart" failure class entirely. This
   leaves `COPILOT_CONFIG.copilot_vision_enabled`/`_model` as default seeds only (unused at runtime);
   `copilot_vision_probe_size` / `_max_tokens` / `render_facts_motion_t` stay live constants.
8. **Capability check = metadata, not a billed call.** `fetch_model_image_support(api_key) ->
   dict[str, bool] | None` fetches OpenRouter `GET /api/v1/models` and returns `{model_id: "image" in
   architecture.input_modalities}` for every listed model. **Verified first-hand** against the live
   endpoint: the field is `architecture.input_modalities`; `openai/gpt-4o-mini` → `["text","image","file"]`
   (✓), `openai/gpt-3.5-turbo` → `["text"]` (✗). Returns the full dict so the UI distinguishes THREE
   states — a transient failure (offline / 5xx / timeout) returns `None`, a bounded ~15s httpx timeout
   (a transient error is not a negative result). httpx is already imported in `openrouter.py`.
9. **Async validation state on App, polled by the Settings draw — with a re-kick guard.** A
   `VisionModelProbe` on `App` (status: IDLE/CHECKING/READY/ERROR · `support: dict[str,bool]` · the key it
   was fetched for). Staleness is `status != CHECKING and (status == IDLE or fetched_key != current_key)` —
   the `!= CHECKING` clause is what stops a per-frame re-kick storm while a fetch for a just-changed key is
   in flight; the single-threaded draw solely owns the IDLE→CHECKING transition (that ownership, not a
   lock, is what makes "one thread max" true). On a stale check the draw records the key it is checking FOR
   and kicks ONE **daemon** thread — daemon is acceptable here (unlike telegram's `daemon=False` upload
   workers) because the op is an idempotent read-only `GET /models` writing only in-memory state, so an
   interpreter-shutdown kill leaks nothing. The worker builds its result locally then assigns
   `support = …; fetched_key = k; status = READY` with **status LAST** (atomic ref-assignment under the
   GIL → no lock needed for the draw's read). The draw renders a badge in a fixed-height `status_slot` (no
   jitter): `checking…` / `✓ supports vision` / `✗ no image input — vision will be skipped` /
   `⚠ model not recognized` (key present, id absent from the dict) / `⚠ couldn't verify (offline?)` (status
   ERROR). Never blocks the frame.
10. **Safe when misconfigured.** `describe_image` stays advisory (returns `""` when disabled or on any
    failure) so a non-vision model or an outage never breaks a probe — the Settings badge is the loud
    signal, the probe degrades quietly to facts-only.

## Files touched

- `shaderbox/copilot/config.py` — `copilot_vision_enabled`/`_model` become default seeds only (mirrored by
  CopilotIntegration, read at runtime via getters); adds `vision_models_fetch_timeout_s` for the /models check.
- `shaderbox/copilot/llm/openrouter.py` — rework `_VISION_SYSTEM` (two-tact); `describe_image(png, hint,
  is_strip)` reads `get_vision_enabled`/`get_vision_model`; add the two getter params to `__init__`; add
  `fetch_model_image_support(api_key) -> dict[str, bool] | None` (~15s timeout).
- `shaderbox/project_session.py` — construct the `OpenRouterLLMClient` (it lives here, not session.py) with
  the two new vision getters + wire the backend's `vision_enabled` callback + the 3-arg `describe_image`
  lambda, all sourced live from `integrations_store.copilot`.
- `shaderbox/copilot/backend.py` — `probe_render(node, t, look_for)`; `_probe_png` contact-sheet returning
  `(png, is_strip)`; the (frame,look_for,is_strip)-keyed vision cache; a `vision_enabled` ctor callback to
  gate the PNG render; thread `look_for`+`is_strip` into `describe_image`.
- `shaderbox/copilot/capabilities.py` — the `CopilotCapabilities` Protocol `probe_render(self, node, t, /)`
  gains `look_for` (else the backend no longer satisfies the Protocol → pyright fails). Add the
  `vision_enabled` capability if wired through the caps object.
- `shaderbox/copilot/tools/inspect.py` — `_ProbeRenderArgs` + the dispatch gain `look_for`; description updated.
- `shaderbox/copilot/prompt.py` — witness/judge split + how to use `look_for`.
- `shaderbox/exporters/integrations.py` — `vision_enabled` + `vision_model` on `CopilotIntegration`
  (defaults mirrored from `CopilotConfig`). NO `apply_limits` change (live-getter seam per D7).
- `shaderbox/app.py` — hold the `VisionModelProbe` state (constructed in `__init__`).
- `shaderbox/copilot/vision_probe.py` (NEW) — the `VisionModelProbe` dataclass + the `ensure_checked` /
  thread-kick + result storage (async state; the fetch fn lives in openrouter.py).
- `shaderbox/popups/settings.py` — the Vision subsection under Copilot (toggle + model field + badge in a
  `status_slot`); the vision handler `save()`s on change (NO `apply_limits`).
- **Tests (must update or `make check`/pytest goes red):** `tests/_caps.py` (`probe_render` Callable type +
  the `lambda _n, _t` stub → add `look_for`); `tests/test_probe_clock_and_turn_end.py` (`lambda n, t` stub +
  the `execute("probe_render", {...})` args); `tests/test_credential_redaction.py` (blast-radius: it builds
  the client directly — add the two vision getters); `tests/test_copilot_user_limits.py` (extend the
  defaults-mirror). NEW: `tests/test_vision_probe.py` covering the classifier, the /models fetch parsing,
  the vision cache (hit/miss on intent), disabled-skip, and a real-GL forward-time contact-sheet regression.

## Manual verification

Split into DETERMINISTIC (headless unit tests, run in `make check`/pytest on this display-less box) and
LIVE (a maintainer `make run` on a box with a display — imgui-ui §0: badge glyph / fixed-slot / no-jitter
rendering is display-only and CANNOT be verified headless). Each check fails for exactly one reason.

**Deterministic (headless — these are the real regression gates):**
- **look_for reaches the eye (WIRE, not behavior):** spy on the injected `describe_image`; call
  `probe_render(look_for="reads HELLO")`; assert the spy's `hint` arg CONTAINS `"HELLO"`. Falsifier: the
  wire is cut → empty hint → no `look_for:` segment. (The behavioral "does the model report the mismatch"
  half is LIVE/soft, below — it is NOT a regression gate.)
- **cache: hit on identical, MISS on new intent:** probe same node+t twice with the SAME `look_for` → second
  issues no `describe_image` call (call-counter spy). Probe same node+t with a DIFFERENT `look_for` → it
  DOES call (a miss). Falsifiers: two calls for identical (cache dead) / one call across differing look_for
  (D6 key bug — stale read).
- **contact-sheet dimensions:** `_probe_png` on an ANIMATES node returns `is_strip=True` + a ~3× wider PNG;
  on a STATIC node returns `is_strip=False` + single-frame dims. Falsifier: static→strip or animated→single.
- **classifier states:** unit-test `fetch_model_image_support` parsing against a fixture (or the vetted live
  dict): a known vision id → True, a known text-only id → False, an absent id → not-in-dict, a simulated
  fetch failure → `None`. Then the badge classifier maps those to ✓ / ✗ / ⚠-not-recognized / ⚠-couldn't-
  verify. Falsifier: text-only→✓, or a fetch failure collapsing to ✗ instead of ⚠.
- **enable toggle wired via the getter (CONSUMER named):** set `integrations.copilot.vision_enabled=False`;
  `probe_render` returns facts-only (no visual line) because `describe_image`/backend read
  `get_vision_enabled()`. Grep the READER: `describe_image` + the backend `vision_enabled` gate. Falsifier:
  the flag flips but the probe still calls vision ⇒ unwired.
- **persisted round-trip + defaults mirror:** `IntegrationsStore` with edited vision fields → `model_dump` →
  reload → fields survive; an old json missing the keys loads to defaults (no store-nuke). Assert
  `CopilotIntegration().vision_enabled/_model == CopilotConfig.copilot_vision_enabled/_model` (extend
  `test_integration_defaults_mirror_config_defaults`). Falsifier: a reverted value / a `ValidationError`
  wiping the store / a drifted default literal.

**Live (maintainer `make run`, deferred to a display box):**
- **Settings badge visual:** set a known vision model → ✓ badge; a text-only id → ✗; a typo id → ⚠ not
  recognized; pull the network → ⚠ couldn't verify. Confirm the `status_slot` height never jitters as the
  status changes, and no per-frame re-kick storm on rapid model-field edits (watch for thread churn).
- **look_for behavioral (soft):** on a node whose text reads "WORLD", `probe_render(look_for="reads HELLO")`
  → the live model's `look_for:` segment reports the mismatch, not a sycophantic yes. (Soft — model-quality,
  not a hard gate.)

## Open questions for the user

None outstanding — 1A/2A/3A confirmed; vision-model configurability + live validation added and refined
through pre-review (D7 live-getter seam, D9 re-kick guard, D8 three-state classifier verified first-hand).

## Review history

**Pre-implementation review (2 adversarial agents, both PARTIAL → resolved):**
- *Cache key omitted `look_for`* (both, BLOCKER) → D6 rewritten: key is `(raw-frame-hash, look_for,
  is_strip)`; added the "MISS on new intent" deterministic check.
- *`capabilities.py` Protocol + `tests/_caps.py` + `test_probe_clock_and_turn_end.py` omitted from blast
  radius; would break `make check`/pytest* (both, BLOCKER) → added to Files touched.
- *D7 data-flow backwards / duplicated literal* (both, SHOULD-FIX) → D7 rewritten to MIRROR the limit
  defaults from `CopilotConfig`.
- *Settings live-apply asymmetry: vision read via `COPILOT_CONFIG` (push) vs main model via live getter —
  save-only would be a restart-only no-op* (both, BLOCKER/SHOULD-FIX) → adopted reviewer 2's cleaner path:
  **live getters, drop the `apply_limits` push**; Settings edit = `save()` only. Deletes the failure class.
- *Badge is the sole verifier of the classifier and is display-only* (R2, BLOCKER) → split Manual
  verification: the classifier + wires are now headless unit tests; only the badge VISUAL is a live check.
- *Re-kick storm on key-change-during-fetch; daemon justification + explicit timeout* (R2, SHOULD-FIX) →
  D9 states the `!= CHECKING` staleness guard, draw owns IDLE→CHECKING, daemon rationale, ~15s timeout.
- *look_for-reaches-the-eye failed for two reasons* (R2, SHOULD-FIX) → split into a deterministic wire-spy
  gate + a soft behavioral live check.
- *D8 `/models` shape was relayed, not verified* (R2) → fetched the live endpoint first-hand;
  `architecture.input_modalities` confirmed; classifier upgraded to a three-state `dict[str,bool]`.
- *Nits:* hash RAW frame bytes not PNG; acknowledge video/script legit cache-miss; motion re-derived in
  `_probe_png` is fine while `probe_render` omits `motion=True` — all folded into D5/D6.
- *Confirmed non-issue (not a fabricated gap):* old integrations.json without the new keys loads clean via
  pydantic defaults (`extra:"forbid"` rejects only UNKNOWN keys) — no migration, per the hard rule.

**Post-implementation review (3 adversarial agents: code-correctness, architecture/conventions,
doctrine/spec-fidelity — 1 real bug, rest CLEAN):**
- *Contact-sheet frames ran BACKWARD off t=0* (correctness + doctrine, CONFIRMED, borderline blocker):
  `_probe_png` rendered frame1 at the ABSOLUTE `motion_t` (1.5) instead of `t + motion_t`, so a probe aimed
  at t=2.5 produced a reversed 2.5/1.5/0.5 strip labelled "early→late", and a probe at t≈1.5 mis-detected
  as STATIC. Invisible to the stub tests (all at t=0). FIXED: render `t, t+dt, t+2dt`; the motion test now
  compares `t` vs `t+dt`. Added `test_probe_png_renders_forward_in_time` (real GL, spies render u_time).
- *Anti-sycophancy wording could be harder for a cheap model* (doctrine, SHOULD-FIX) → added an
  evidence-gate sentence to tact-2: answer YES only if you can point to the pixels; if guessing, answer NO.
- *NITs accepted as-is:* the badge renders even when the Enabled toggle is off / when no key is set (a user
  editing the model wants validity regardless — harmless, `_VISION_BADGE` covers all 5 verdicts so no
  KeyError); an ERROR verdict doesn't auto-retry on an unchanged key (editing a field re-checks; a
  per-frame retry would reintroduce the storm). Two stale spec-doc lines (session.py→project_session.py,
  config.py new field) corrected. Architecture/conventions + unwired-mechanism audit + import-cycle +
  live-getter-seam-complete all returned CLEAN.
