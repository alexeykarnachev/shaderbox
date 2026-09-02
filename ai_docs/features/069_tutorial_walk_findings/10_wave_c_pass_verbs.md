# 069 W-C — Pass verbs: crash, commit, activate, hotkeys, first render

Implementation spec for wave W-C of feature 069. The parent spec (`01_spec.md § W-C`) fixes the
shape; this file fixes the code. Locked decisions D10 and D11 apply and are not re-opened, nor is
the target-only skip rule, the `drawn_frame` design, or the provisional Alt+P / Alt+A chords.

## Goal

The five pass verbs a person reaches from the strip — rename, add, activate, open settings, and
"show me what this pass draws" — stop lying. Renaming a pass through the gear no longer crashes the
frame that performs it; a name typed into the gear commits when focus leaves the field rather than
being silently discarded; adding a pass activates it (tab, output, gear) instead of leaving a dim
tile behind the old output; `OPEN_PASS_SETTINGS` and `ADD_PASS` reach the keyboard as commands the
chord-uniqueness test and the generated Help table pick up; and every pass in a reopened document
renders itself once, one pass per frame, so the strip shows pictures instead of black rectangles
that the user has to click to wake. Nothing about the steady state changes: after the sweep the
renderer still draws only the output chain, and the draw-once invariant in `pass_graph.py` still
guards it.

## Findings folded

Six, quoted verbatim from `00_findings.md`:

- **#9** (UX feature request, Pass settings gear): "add a keybinding to open settings for the
  currently selected pass (Alt+P?)"
- **#17** (CRASH, Pass settings gear — rename `main` → `paint`): "typed 'paint', hit Enter, the app
  crashed: `KeyError: 'main'` at `pass_settings.py:84 _draw_inputs(app, document_id, name,
  document.passes[name])`. After relaunch the name is 'paint'."
- **#18** (UX, Pass settings gear — name field): "first attempt: typed 'paint' without Enter — the
  field showed it but the Reads wiring still said 'main'; closed, reopened, 'main' was back. The
  name should auto-apply without Enter."
- **#25** (UX feature request, Passes strip — add pass): "add an 'add pass' hotkey as well."
- **#28** (UX, Passes strip — add pass): "when creating a pass, auto-activate it: open its code and
  render it."
- **#36** (UX design, Passes strip — tiles after app reopen): "I need to click each pass manually to
  trigger its redraw after I close/open the app. They don't initialize automatically."

## Out of scope

- **The prose cut on the gear's own strings** (#5, #7, #10 — the "(?)" Reads tooltip, the
  `f"size ({w}, {h})"` label, the `Pass settings — what it reads, what it draws into` gear tooltip)
  and the popup's auto-size: **W-B**. W-C edits `_draw_name` and `_draw_body` in the same file and
  leaves every literal exactly as it stands, so the two waves do not fight over the same lines.
- **The Resolution funnel and the size slider's disabled state for the output pass** (#1, #2, #3,
  #4): **W-A**. W-C reads `document.canvas_size` in `_draw_target` unchanged.
- **The strip's sublines, the naming rule, and default wiring by name** (#19, #37, D9, D12):
  **W-D**. `_draw_pass_tile` keeps its `sublines` argument in W-C.
- **Moving `NEW_DOCUMENT` off Ctrl+N and `TOGGLE_DOCUMENT_PLAY` off Ctrl+Space**, and the full
  chord ownership table: **W-E**. W-C adds two chords on the Alt tier, which the audit is free to
  move later; W-C does not wait for it.
- **The mouse, the clear-canvas command, and pass-qualified scripting**: **W-G**.
- **A builtin `u_mouse`**: decided as "none" by the parent spec, owned by **W-G**.
- **Committing on deactivate for the copilot chat input, the document-name field, the uniform
  value inputs, the emoji/lib search fields, and the YouTube/Telegram exporter form fields**: not in
  scope for any wave. § Design decisions item 6 enumerates all 22 `input_text` sites and states why
  each is in or out under one rule. The shader-library picker's four inline inputs ARE in scope —
  the parent spec's "every § 7.5 inline input" names them.

## Design decisions

### 1. `_draw_name` returns whether it renamed; the body skips the two sections that index by name

`_draw_name` becomes:

```python
def _draw_name(app: App, document_id: str, name: str) -> bool:
```

returning `True` when it performed a rename that succeeded this frame, `False` otherwise. A failed
rename (a name `_pass_name_error` rejects, or a document the core cannot find) returns `False`: the
document dict is untouched, so the body may safely continue.

`_draw_body` becomes:

```python
    imgui.separator_text("Pass")
    renamed = _draw_name(app, document_id, name)
    if not renamed:
        imgui.dummy((0.0, float(SPACE.MD)))
        _draw_inputs(app, document_id, name, document.passes[name])
        imgui.dummy((0.0, float(SPACE.MD)))
        _draw_target(app, document_id, name)

    imgui.dummy((0.0, float(SPACE.MD)))
    return not ghost_button("Close")
```

The guard skips **only** the two sections that index `document.passes` by the now-dead name —
`_draw_inputs` and `_draw_target`. Everything else draws on every path, and in particular the
`return not ghost_button("Close")` line still runs, so the Close row is submitted on the rename
frame and one Close click still closes the modal.

**Why not a plain early return.** Items 1 and 2 interact: clicking Close moves focus off the name
field, so on that same frame `is_item_deactivated_after_edit()` is True, `_commit_pass_name`
renames, and `_draw_name` returns True. A body that returned there would exit **before**
`ghost_button("Close")` was ever submitted, and imgui cannot report a click on a widget that was not
submitted — the click would be swallowed and the user would have to click Close a second time, with
the button drawn under an already-released cursor on the following frame. That is manual step 1's
own shape ("type, click elsewhere") aimed at the Close button, so it would be a defect introduced by
the crash fix. Guarding the two sections instead keeps the rename (which is what the user asked for
and which lands) and keeps the Close click.

The next frame reads `app.pass_settings_name`, which `App._on_pass_renamed` has already re-pointed
to the new name, and draws the full body against the live pass.

The falsifier this closes, reproduced headlessly against the code as it stands: with the modal open
on `main` and a rename to `paint` performed inside `_draw_name`, `_draw_body` raises
`KeyError: 'main'` at `pass_settings.py:84`. That is finding #17's crash verbatim, and the guard is
the whole fix — no try/except, no re-lookup of `app.pass_settings_name` mid-body. A mid-body re-read
would be the guard-pile shape (`conventions.md`: structural impossibility over guard-piles); the
skip makes the dead local unreachable instead of tolerated.

The rename frame therefore emits `separator_text` + the name row + the Close row. That is a legal
imgui frame — verified by execution across three consecutive frames — because the name row's
`label_row` / `input_text` / `same_line` / `help_marker` are already balanced when `_draw_name`
returns, and the modal's `begin_popup_modal` scope is closed by `modal_window`'s context manager
regardless of the body's return path.

Two further indexing sites exist inside the skipped sections and are not a second crash even if
reached: `_draw_inputs` and `_draw_target` both read
`document.graph.passes.get(name, PassEntry())`, a `.get` with a default, and the wiring combo's
`sorted(document.passes)` is a list build rather than an index.

### 2. The name field commits on deactivate-after-edit, with Enter as a shortcut (D11)

`_draw_name`'s input keeps `enter_returns_true` and gains the deactivate branch:

```python
    committed, app.pass_settings_name_buf = imgui.input_text(
        "##pass_settings_name",
        app.pass_settings_name_buf,
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    if committed or imgui.is_item_deactivated_after_edit():
        return _commit_pass_name(app, document_id, name)
```

`is_item_deactivated_after_edit()` is read on the line immediately after the `input_text` call,
before the `imgui.same_line()` / `help_marker` that follow — the item-scoped queries read the LAST
submitted item, so any intervening widget makes them read the wrong one.

`_commit_pass_name(app, document_id, name) -> bool` is a module-level free function (no
`@staticmethod`, per `conventions.md ## Code rules`):

```python
def _commit_pass_name(app: App, document_id: str, name: str) -> bool:
    new_name = app.pass_settings_name_buf.strip()
    if not new_name or new_name == name:
        app.pass_settings_name_buf = name
        return False
    error = app.session.rename_pass(document_id, name, new_name)
    if error:
        app.notifications.push(error)
        app.pass_settings_name_buf = name
        return False
    return True
```

Two properties beyond today's behaviour, both required by D11's "never per keystroke":

- **A rejected or empty name resets the buffer to the live name.** Today a rejected rename leaves
  the bad text in the buffer, so the next deactivate re-fires the same rejection and pushes a second
  notification. Resetting makes the rejection terminal: one notification, the field snaps back to
  what the pass is actually called.
- **No commit fires per keystroke**, because `is_item_deactivated_after_edit` is true only on the
  frame focus leaves an edited item. `rename_pass` moves a file, rewrites every edge, re-points the
  output and the open tab (`project_session.py::rename_pass`), and `_pass_name_error` rejects the
  intermediate `"p"` / `"pa"` states — exactly finding #18's stated reason.

`is_item_deactivated_after_edit` covers focus leaving the field for any reason, including the modal
closing while the field holds focus. It does **not** cover the Close button or Escape when the user
clicked away first (the field is already deactivated and the buffer holds an uncommitted edit) —
that is item 3.

### 3. Close and Escape commit a pending edit through one funnel

`draw_pass_settings`'s close branch and `hotkeys.py::_handle_escape` are the two paths that set
`PopupState.CLOSED` for this modal. Verified: `grep PopupState.CLOSED shaderbox/` finds
`pass_settings.py:67` (the body returned False — the Close button) and `hotkeys.py:303` (Escape);
no other site closes this popup. `dispatch_commands` (which contains `_handle_escape`) runs at
`ui.py:336`, ahead of `draw_pass_settings` at `ui.py:429`, so on the Escape frame the body never
draws and a commit inside the body is unreachable. That is exactly the mechanism by which an
un-entered edit is discarded today.

So the commit lives on `App`, at the one funnel both paths reach:

```python
    def close_pass_settings(self) -> None:
        """Close the gear, committing a pending rename first."""
        document_id = self.current_document_id
        name = self.pass_settings_name
        buf = self.pass_settings_name_buf.strip()
        if name and buf and buf != name and document_id in self.ui_documents:
            error = self.session.rename_pass(document_id, name, buf)
            if error:
                self.notifications.push(error)
        self.popup_state = PopupState.CLOSED
        self.pass_settings_name = ""
        self.pass_settings_name_buf = ""
```

Call order at the two sites:

- `popups/pass_settings.py::draw_pass_settings`, replacing the three-line close block:
  `app.close_pass_settings()` then `imgui.close_current_popup()`. The `imgui` call stays at the
  draw site — `App` owns no imgui popup-stack calls.
- `shaderbox/hotkeys.py::_handle_escape`, in the `elif app.any_popup_open():` branch: when
  `app.popup_state == PopupState.PASS_SETTINGS`, call `app.close_pass_settings()`; otherwise the
  existing `app.popup_state = PopupState.CLOSED` line runs as today. The lib-picker's
  `inline_input_owns_esc` guard above it is untouched.

`close_pass_settings` is idempotent against a name that did not change and against a rename that
already landed this frame via item 2 (after which `pass_settings_name` and `pass_settings_name_buf`
are both the new name, so `buf == name`). The `buf != name` guard is a cheap short-circuit rather
than the thing that makes it safe: `ProjectSession.rename_pass` itself returns `""` for
`new == old` without touching anything, so the callee is idempotent too and a redundant call would
be harmless.

A third close path exists in principle: opening another popup replaces `popup_state` through
`_open_popup`. In practice no control inside the gear opens another popup, and no command dispatches
while a modal is open (`popup_suppresses` returns True for every scope), so the pending edit cannot
survive into another popup. No hook is added there.

### 4. Add pass activates the pass, then opens the gear (D10)

`widgets/pass_list.py::_draw_add_input`'s success branch, today:

```python
            app.pass_add.close()
            app.open_pass_settings(name)
```

becomes activate-then-open:

```python
            app.pass_add.close()
            open_pass(name)
            error = app.session.set_output_pass(document_id, name)
            if error:
                app.notifications.push(error)
            app.open_pass_settings(name)
```

That is byte-for-byte what a tile click does (`_draw_pass_tile`'s `if result.clicked:` branch —
`open_pass(name)` then `set_output_pass`), minus the click path's `if not is_output` guard, which is
vacuous for a pass created this frame. The order matters and is D10's: activate first, so the gear's
`is_output` state and the viewer agree on the frame the modal opens.

`_draw_add_input` already receives `document_id`; it gains the `open_pass` callback as a parameter,
threaded from `draw(app, document_id, open_pass)` which already holds it:

```python
def _draw_add_input(
    app: App, document_id: str, open_pass: Callable[[str], None]
) -> None:
```

The copilot mirrors nothing: `grep -rn 'add_pass\|rename_pass\|delete_pass\|set_output_pass'
shaderbox/copilot/` returns no hits, so the copilot has no pass verbs at all and finding #28's "same
for the copilot's add-pass path if it has one" resolves to "there is none".

### 5. `OPEN_PASS_SETTINGS` (Alt+P) and `ADD_PASS` (Alt+A)

Two `CommandId` members and two `CommandSpec` entries in `shaderbox/commands.py`, both in
`CommandCategory.TOOLS` beside the other Alt-tier verbs, both `CommandScope.GLOBAL` (the default —
these act on the current document, not on a focused surface):

```python
    CommandSpec(
        CommandId.OPEN_PASS_SETTINGS,
        "Pass settings",
        _chord(K.p, K.mod_alt),
        C.TOOLS,
    ),
    CommandSpec(CommandId.ADD_PASS, "Add pass", _chord(K.a, K.mod_alt), C.TOOLS),
```

Both chords are free: the only `mod_alt` chords in `COMMAND_SPECS` today are Alt+S
(`OPEN_SETTINGS`), Alt+E (`EXAMPLES`) and Alt+/ (`TOGGLE_CHEATSHEET`). `K.p` is otherwise spoken for
only as Ctrl+P (`OPEN_LIB_PICKER`) and Ctrl+Shift+P (`OPEN_PALETTE`), which are different chord
ints. `route_flag` already returns `route_always` for any Alt chord, so both reach the dispatcher
while a text input is active — which matters for Alt+A specifically, since the add-pass input is
itself a text input the user may be sitting in.

The callbacks, in `App._build_command_callbacks`:

```python
            CommandId.OPEN_PASS_SETTINGS: self.open_pass_settings_for_panel_pass,
            CommandId.ADD_PASS: self.open_add_pass,
```

Two new `App` methods rather than lambdas, because both need a guard the lambda cannot carry
(the current document may not exist — the callbacks fire from a registry that knows nothing about
document state):

```python
    def open_pass_settings_for_panel_pass(self) -> None:
        document_id = self.current_document_id
        if document_id not in self.ui_documents:
            return
        self.open_pass_settings(pass_name_of(self.panel_pass(document_id).source.path))

    def open_add_pass(self) -> None:
        document_id = self.current_document_id
        if document_id not in self.ui_documents:
            return
        self.pass_add.open(self.session.paths.passes_dir_for(document_id))
```

`panel_pass(document_id)` returns a `Pass`, and `open_pass_settings` takes a NAME, so the name comes
from `pass_name_of(render_pass.source.path)` — `paths.pass_name_of` is already imported in `app.py`
(it is what `_on_pass_renamed` uses). Finding #9's "currently selected pass" is `panel_pass`'s
notion exactly: the active shader tab's pass when it belongs to this document, else the output.

`open_add_pass` calls the same `pass_add.open(...)` the ghost button calls
(`widgets/pass_list.py::draw`), so the inline input's one-shot `needs_focus` grabs the keyboard on
its first draw — no extra focus step, which is what finding #25's "then focusing the input" asks
for.

The Help shortcuts table picks both up with no code change: `help_content.py::_shortcuts_section`
enumerates `COMMAND_SPECS` filtered by category, and both new specs carry `C.TOOLS` and a non-zero
`default_chord`. The chord-uniqueness test (`tests/test_command_routing.py::
test_no_two_specs_share_a_chord_in_overlapping_scopes`) picks them up the same way — it loops
`COMMAND_SPECS` and needs no edit to cover them.

### 6. Which inline inputs D11 applies to, and which it does not

**One rule decides every site.** D11's commit-on-deactivate applies to an inline input whose commit
fires an **ACTION** — a rename, a create, an add, a pack title, a paste box that parses. There the
typed text is a request that has not happened yet, so losing it on click-away loses work, which is
finding #18 exactly. It does **not** apply to a **form field whose value IS the state**, written per
keystroke and applied by a separate Save / Apply / Connect control: there is nothing to commit,
because the write already happened. A live filter (a search query) is the degenerate case of the
second bucket — the value is the state and the state is consumed the same frame.

**The domain, enumerated.** Every `input_text` / `input_text_multiline` call site in `shaderbox/`:
**15 direct calls** (16 grep hits minus `app.py:179`, which is `config_input_text_cursor_blink`, a
flag rather than a call) **plus the 7 callers of `ui_primitives.labeled_text_input` /
`labeled_multiline_input`**, which are the measured sites for those two forwarders. Twenty-two
sites, every one classified below; none is left unlisted, and none falls outside the two buckets.
The tables below carry **23** rows for those 22 sites, because
`lib_picker/tree.py::_draw_inline_new_input` is ONE shared function drawn twice — once for a new
file and once for a new directory — and the two uses differ in what their commit creates, so each
gets a row while a single edit covers both. Nine rows are action inputs (six in scope, three whose
button stays the only commit) and fourteen are form fields.

**Action inputs — in scope for W-C (6 of the 9 action rows, over 5 call sites):**

The parent spec says "Same rule applied to `_draw_add_input` and **every § 7.5 inline input**", and
§ 7.5 is titled "Inline inputs inside modals (rename / new-file / new-dir)" — which names the
shader-library picker's inputs. They are in scope here, not deferred: no other 069 wave opens the
picker, and this repo does not park work in docs.

| Site | Action its commit fires | Commit semantics under D11 |
|---|---|---|
| `popups/pass_settings.py::_draw_name` | `rename_pass` | A click-away renames the pass; a name `_pass_name_error` rejects pushes one notification and snaps the buffer back to the live name; Enter is the shortcut; Escape closes the modal and commits through `close_pass_settings` (item 3). Items 2 and 3 carry the detail. |
| `widgets/pass_list.py::_draw_add_input` | `add_pass` | A click-away creates the pass and runs item 4's activate-then-gear block; a rejected name pushes its notification and leaves the input open with the text intact, since unlike rename there is nothing to snap back to; Enter is the shortcut; Escape cancels without creating (open question 3 covers why the Escape branch needs no suppression while the `x` button's does). |
| `popups/lib_picker/tree.py::_draw_inline_new_input` (new file) | `commit_file_new` → `create_file_in` | A click-away creates the `.glsl` file and, through the existing `on_create` callback, opens it; a rejected name (traversal or an existing target) already pushes its own notification from `_validate_target` and returns `None` without closing, so the input stays open with the text — no new error handling is needed, the commit function's contract already matches. Enter is the shortcut; Escape cancels. |
| `popups/lib_picker/tree.py::_draw_inline_new_input` (new dir) | `commit_dir_new` → `create_dir_in` | Same shape and the same shared function, so one edit covers both: a click-away creates the directory; a rejected name notifies and leaves the input open; Enter is the shortcut; Escape cancels. |
| `popups/lib_picker/tree.py::_draw_file_rename_input` | `rename_file` | A click-away renames the lib file; a self-rename is the callee's silent no-op that closes the input; a rejected name notifies from `_validate_target` and leaves the input open; Enter is the shortcut; Escape cancels. |
| `popups/lib_picker/preview.py::_draw_function_tag_editor` (new tag) | `shader_lib_tags.add` | A click-away adds the typed tag and clears the buffer; an empty or whitespace-only buffer commits nothing (the existing `and buf` guard); the `+ Add` button **stays** as the Enter-equivalent shortcut beside the field, so the site gains a third commit path rather than replacing one. Escape cancels via the picker's own `inline_input_owns_esc` route. |

**One ordering constraint the tag site adds.** `preview.py` reads
`picker_tag_input_focused = imgui.is_item_focused()` on the line immediately after its `input_text`,
and that slot is exactly where `is_item_deactivated_after_edit()` must also be read — both are
item-scoped queries on the last submitted item. Both are read there, into locals, on adjacent lines
before the `imgui.same_line()`; neither may move below the `+ Add` button. The existing
`picker_tag_input_focused` write must keep its current position and meaning, because
`inline_input_owns_esc` and the picker's outer Enter gate both depend on it.

**Action inputs that keep their button as the only commit (3 rows):**

| Site | Action | Why the button stays the only commit |
|---|---|---|
| `exporters/telegram.py::_draw_pack_row` (new pack title) | `_create_pack`, via the `Create` button | A click-away would create a sticker pack on Telegram — a remote, user-visible side effect — from a half-typed name. The explicit `Create` button is the right and only affordance. |
| `exporters/youtube.py` (client-secret paste box) | `_ingest_client_secret` parses the pasted JSON on change | It already commits per change, so nothing is lost on click-away; there is no deferred request to rescue. |
| `widgets/copilot_chat.py::_draw_credential_input` (`msg.gate_input`) | `answer_gate_credential`, via the `Save` button | The value is a **secret the user may still be typing**, and the commit sends it into a live turn. Committing on click-away would submit a partial credential. The `Save` button stays the only commit. |

**Form fields — out of scope, nothing to commit (14 of 22):**

| Site | The value IS the state, applied by |
|---|---|
| `popups/settings.py::_draw_copilot_config` (OpenRouter key) | written to `cfg.openrouter_key` and `integrations_store.save()`d the frame it changes |
| `popups/settings.py::_draw_copilot_config` (Model) | same write-and-save block |
| `exporters/youtube.py` (Title) | `rs.title`, read by the Upload button |
| `exporters/youtube.py` (Description) | `rs.description`, same |
| `exporters/youtube.py` (Tags) | `rs.tags_raw`, same |
| `exporters/telegram.py` (Bot token) | `self._tg.bot_token`, applied by the `Connect` button |
| `tabs/document.py` (document name) | `ui_document.ui_state.ui_name`, written per keystroke; also not a modal inline input |
| `widgets/uniform.py` (scalar text entry) | the uniform value, applied per keystroke into the live document |
| `widgets/uniform.py` (text uniform) | same |
| `widgets/copilot_chat.py` (chat input) | `app.copilot_input`. Enter SENDS, so this is the one field where a deactivate commit would be actively wrong: clicking away from a half-typed message would fire an LLM turn. |
| `popups/lib_picker/search.py` (picker query) | a live filter, applied the same frame |
| `popups/emoji_picker.py` (search) | a live filter, same |
| `ui_primitives.py::labeled_text_input` | a forwarder with no commit semantics of its own; its 7 callers are the measured sites and each is classified above |
| `ui_primitives.py::labeled_multiline_input` | same |

Two of these deserve their reason spelled out because they look like credential fields and are not:
`settings.py`'s OpenRouter key and `telegram.py`'s bot token both write and persist as the user
types, so a deactivate commit would duplicate a funnel that already fires rather than add one. The
copilot gate input is the field that genuinely holds an unsubmitted secret, and it is in the action
bucket with an explicit override.

### 7. The first-render sweep

**`Document.render` gains one parameter:**

```python
    def render(
        self,
        u_time: float | None = None,
        canvas: Canvas | None = None,
        target: str | None = None,
    ) -> None:
```

**The name collision.** The iteration loop already binds a local named `target`
(`document.py:425`: `target = canvas if (name == output and last) else None`). That local is renamed
to `draw_into` in the same edit, and `render_pass.render(canvas=draw_into, ...)` follows. Without
the rename the parameter is shadowed from the loop's first line onward and every later read of it is
silently the wrong value.

**The resolved output.** One rebinding at the top replaces three reads:

```python
        resolved = target if target is not None else self.graph.output_pass
        output = self.graph.output_pass
```

- `resolved` feeds the early-out guard (`if resolved is None or resolved not in self.passes:`),
  `plan_for_output(self.graph, resolved)`, and the cycle fallback (`order = [resolved]`). All three,
  per the parent spec's decision (1).
- `output` — the graph output, unchanged by `target` — feeds the two `name == output` comparisons:
  the full-size exemption (`if name != output:`) and the external-canvas choice
  (`draw_into = canvas if (name == output and last) else None`). That is decision (3), and the
  "unchanged" half is the load-bearing one: a target pass sizes by its own scale and never receives
  an external canvas, because a target render is never a preview or an export.

`self._graph_errors` is still assigned from `plan_for_output(self.graph, resolved)`. Its error half
is target-independent by construction: `plan_passes(graph)` walks every name in `graph.passes` and
returns the same error list whatever the requested target is; `_order_for` filters only the ORDER.
`Document.graph_errors` also has no production consumer — `grep` finds it read only in
`tests/test_document_graph.py`, `tests/test_radiance_cascades_example.py` and
`tests/test_pass_verbs.py` — so a target render cannot corrupt anything downstream.

**The skip, and its scope.** Two new fields on `Pass` (`shaderbox/core.py`, in `__init__`):

```python
        # The document frame this pass last drew in; -1 means never.
        self.drawn_frame: int = -1
        self.first_render_done: bool = False
```

In `Document.render`, at the top of the `for name in order:` loop body, **before** the size fixup:

```python
        for name in order:
            render_pass = self.passes[name]
            if (
                canvas is None
                and target is not None
                and render_pass.drawn_frame == self._frame
                and self._frame >= 0
            ):
                continue
```

Four conjuncts, each load-bearing:

- `canvas is None and target is not None` — **only a first-render sweep render participates in the
  skip.** This is the parent spec's "ONLY a target render skips passes already drawn this frame",
  and it is the condition that keeps the live loop correct: `ui.py:265` renders the output chain
  into `preview_canvas` and `ui.py:296` renders the same chain into each pass's OWN canvas in the
  same frame. Under an unscoped skip the second render would find every pass already stamped and
  draw nothing, and every strip thumbnail would go permanently black.
- `render_pass.drawn_frame == self._frame` — the pass already drew this frame, so the sweep leaves
  it alone. This is what stops an iterated feedback pass from advancing twice in one frame, since
  the whole iteration loop is skipped, not just one draw.
- `self._frame >= 0` — `_frame` starts at `-1` and only `begin_frame` advances it. The example
  documents in `app.ui_document_examples` are rendered by `ui.py:308-310` and NEVER passed to
  `begin_frame` (verified: the only `begin_frame` call sites in `shaderbox/` are `ui.py:246` and the
  two export loops), so their `_frame` is `-1` forever. The guard makes `-1 == -1` not a skip. It is
  belt-and-braces given that the examples popup issues no target renders, but it costs one
  comparison and removes a whole class of "why is this frozen" from the design.

**Where the stamps are written.** Immediately after the skip check, before the size fixup, in the
same loop body:

```python
            render_pass.drawn_frame = self._frame
            render_pass.first_render_done = True
```

Set on the DOCUMENT side, not inside `Pass.render`, and set on ATTEMPT rather than success — the
same posture `Document.first_render_done` already takes (066 D2). `Pass.render` returns early when
`compile()` leaves `self.program` None, so stamping inside it would leave a permanently broken pass
un-stamped and the sweep would re-elect it every frame forever, paying a compile each time. Stamping
here bounds the sweep at exactly one attempt per pass. It also means every render path stamps: the
preview render, the own-canvas render, an export render, and the sweep all mark what they drew,
which is what makes the sweep's skip meaningful in the first place.

Both fields carry the smallest comment that earns its line: `drawn_frame` states the invariant as
it is now, and `first_render_done` needs none because the name says it. The reason the skip tests
`>= 0`, and the whole story of why the setter moved out of `Pass.render`, live in this spec and in
the commit message — not in `core.py`, per `conventions.md ## Code rules` (a comment states the now
and never narrates development history).

**`Document.first_render_done` narrows by one conjunct** (`document.py:392-393`):

```python
        if canvas is None and target is None:
            self.first_render_done = True
```

so a sweep render never consumes the document-level first-render budget the live loop uses to admit
one NEW DOCUMENT per frame. The existing condition `canvas is None` is strictly wider, so this
change can only make the flag later, never earlier, and the existing test
`test_a_foreign_canvas_render_leaves_first_render_pending` still passes unchanged.

**The frame gate picks at most one pass per frame.** In `ui.py::update_and_draw`, in the
`if not app.any_popup_open():` document-render block, after the existing `document.render()`:

```python
        for document_id in tick_documents:
            ui_document = app.ui_documents.get(document_id)
            if ui_document is not None:
                document = ui_document.document
                document.render()
                pending = next(
                    (
                        name
                        for name, render_pass in document.passes.items()
                        if not render_pass.first_render_done
                    ),
                    None,
                )
                if pending is not None:
                    document.render(target=pending)
```

The order is deliberate: the output chain draws first and stamps its passes, so the `next(...)` scan
sees them as done and the sweep elects a pass genuinely outside the chain. One pass per document per
frame, so a six-pass document is fully drawn within six frames of the render set admitting it —
finding #36's "every tile shows a picture within a second" at any plausible frame rate.

`document.passes` is a plain dict, so the scan is insertion-ordered and deterministic; with the
per-pass stamp being monotonic (never cleared except by the pass being replaced), no pass can be
elected twice and the sweep terminates.

The sweep draws the elected pass's WHOLE ancestor chain, because `plan_for_output(graph, resolved)`
returns the transitive closure of `plan.reads` from `resolved` (`pass_graph.py::_order_for` walks it
with an explicit stack). Drawing a pass alone would sample black inputs and paint a wrong picture
into its tile, which is worse than the black tile it replaces. The ancestors it re-draws are exactly
the ones already stamped by the output render, so the skip eliminates them and the chain costs only
what is genuinely new.

**The stale wash keeps its meaning, narrowed.** `widgets/pass_list.py::draw` computes `live` from
`evaluation_order(document.graph, output)` and washes any tile outside it. That is unchanged: after
the sweep a pass outside the output chain shows a real picture under a grey wash, which reads as
"was drawn, is no longer live" instead of today's "never initialized".

**Export is untouched, by intent.** `target` is not threaded through `render_media`,
`_render_media_into`, `_render_image` or `_render_video`; all four keep calling
`self.render(u_time=..., canvas=...)`. An export renders the OUTPUT and only the output — a sweep
during an export would draw passes the exported frame does not contain, cost time per exported
frame, and (through the shared `drawn_frame` stamp) interact with the video loop's per-frame
`begin_frame`. The export path's `reset_feedback()` already sets `_frame = -1`, so even the stamp
comparison is inert there.

### 8. The imgui skill's § 7.5 is rewritten to the commit rule

`.claude/skills/imgui-ui/SKILL.md § 7.5`'s **Pattern** bullet today reads "Enter commits, Esc
cancels". It becomes the D11 rule: an inline input whose commit performs a TRANSACTION (a rename, a
file creation, a pass creation) commits on `imgui.is_item_deactivated_after_edit()`, with Enter
(`enter_returns_true`) as a shortcut and Esc as cancel; the deactivate query is read on the line
immediately after the `input_text` call, before any `same_line`; and a modal that can close while
the buffer holds an uncommitted edit commits at its close funnel, because a body that does not draw
cannot commit. A live filter (a search query) and a per-keystroke value field are named as the two
shapes the rule does not cover. The § 7.5 **Focus**, **Outer keyboard suppression** and
**Auto-expand** bullets are unchanged.

Ships in the same commit as the code, per the fleet rule that a rule with no gate is a wish — here
the gate is the tests in § Tests, and the rule lands with all six in-scope sites already converted
(the pass name field, the add-pass input, and the four shader-library picker inputs § 7.5 is
literally about), so the skill documents shipped behaviour rather than an intention.

## Files touched

| File | What changes |
|---|---|
| `shaderbox/popups/pass_settings.py` | `_draw_name` returns `bool` and commits on deactivate-after-edit via a new `_commit_pass_name` free function; `_draw_body` skips `_draw_inputs` / `_draw_target` on a rename while still drawing the Close row; the close branch calls `app.close_pass_settings()`. |
| `shaderbox/popups/lib_picker/tree.py` | `_draw_inline_new_input` (new file / new dir, one shared function) and `_draw_file_rename_input` commit on deactivate-after-edit as well as Enter; the existing Escape-cancel and `x` branches are unchanged. |
| `shaderbox/popups/lib_picker/preview.py` | `_draw_function_tag_editor` commits on deactivate-after-edit as well as Enter and the `+ Add` button; the `picker_tag_input_focused` write keeps its position and meaning. |
| `shaderbox/widgets/pass_list.py` | `_draw_add_input` takes `open_pass`, commits on deactivate-after-edit as well as Enter, and on success activates the new pass (tab + output) before opening the gear; `draw` passes `open_pass` through. |
| `shaderbox/commands.py` | Two `CommandId` members (`OPEN_PASS_SETTINGS`, `ADD_PASS`) and their two `CommandSpec` entries on Alt+P / Alt+A in `C.TOOLS`. |
| `shaderbox/app.py` | `close_pass_settings`, `open_pass_settings_for_panel_pass` and `open_add_pass` methods; two entries in `_build_command_callbacks`. |
| `shaderbox/hotkeys.py` | `_handle_escape` routes a PASS_SETTINGS close through `app.close_pass_settings()`. |
| `shaderbox/core.py` | `Pass.drawn_frame: int = -1` and `Pass.first_render_done: bool = False` in `__init__`. |
| `shaderbox/document.py` | `render` gains `target: str | None = None`; the iteration loop's `target` local is renamed `draw_into`; `resolved` feeds the guard / planner / cycle fallback while `output` keeps the two `name == output` comparisons; the target-only skip and the two per-pass stamps at the top of the loop body; `first_render_done` gains the `and target is None` conjunct. |
| `shaderbox/ui.py` | The document-render block elects at most one pass with `first_render_done == False` per document per frame and renders it with `render(target=name)`. |
| `.claude/skills/imgui-ui/SKILL.md` | § 7.5's Pattern bullet rewritten to the commit-on-deactivate rule. |
| `tests/test_pass_verbs.py` | The headless rename-through-the-popup test, the `_commit_pass_name` return test, and the add-pass-activates test. |
| `tests/test_lib_files.py` | New module: the picker inline-input commit-on-click-away test, driven through the imgui rig with an injected keystroke. |
| `tests/test_lazy_compile.py` | The frame-accounting tests for the sweep, including the two-output-renders-in-one-frame sibling and the broken-pass stamp case. |
| `tests/test_help_content.py` | The Help table pickup assertion. |
| `shaderbox/help_content.py` | No change — `_shortcuts_section` reads `COMMAND_SPECS`. Listed so a reviewer knows it was checked. |
| `shaderbox/pass_graph.py` | No change — `plan_for_output` already takes a target and `assert_plan_invariants` already guards the draw-once rule on the path that draws. Listed for the same reason. |

## Tests

Each named with the falsifier: the bug that makes it go red.

### `tests/test_pass_verbs.py::test_the_gear_body_survives_a_rename_mid_frame`

Drives `pass_settings._draw_body` through a real imgui frame on the `app` fixture (which already
creates an imgui context in `App.__init__`), with `app.pass_settings_name_buf` holding a new name.
Asserts no exception, that the pass is renamed, and that `_draw_body` returned `True` (keep open).

The rig, verified to work headlessly on this box: `imgui.new_frame()` → `imgui.begin("rig")` →
call the popup body → `imgui.end()` → `imgui.end_frame()`. No `render()` / backend call is needed
because nothing is presented.

**The real `_draw_name` runs — it is not stubbed.** A stub that replaces `_draw_name` supplies the
rename itself, so it cannot observe whether `_draw_name` returns a `bool` at all: an implementation
returning `True` unconditionally without ever renaming would pass. Instead the test monkeypatches
the **leaf**, `pass_settings._commit_pass_name`, to a wrapper that calls the real one and forces the
branch (standing in for the `is_item_deactivated_after_edit()` that a focus transition would
produce — drivable in the rig, but not worth five frames of setup when this test is about the
crash, not the trigger). The real
`_draw_name` then runs, performs the real rename through the real commit, and its return value is
what `_draw_body` consumes — so the test pins the plumbing end to end: commit → `_draw_name`'s
return → `_draw_body`'s guard.

**Also asserts the Close row was drawn on the rename frame**, via `_draw_body`'s return value:
`return not ghost_button("Close")` is the only statement that produces it, so a `True` return proves
the Close row was submitted rather than skipped. That pins finding 2's fix — a plain early return
would exit before the button and the return value would come from somewhere else (or the body would
return `True` from the guard, which is why the test asserts the rename AND the return together, and
why the third assertion below exists).

**Falsifier (crash):** with `_draw_body` continuing past a rename (today's code), the frame raises
`KeyError: 'main'` at `pass_settings.py:84`. Confirmed by running exactly this shape against the
current tree — it reproduces finding #17's traceback verbatim, so the test is red before the fix and
green after.

**Falsifier (Close):** a third assertion runs the body for a second and third frame with the buffer
now equal to the live name (no rename), and asserts `_draw_body` still returns `True` and that a
`ghost_button("Close")` press closes it. Verified by execution: the post-fix body runs three
consecutive frames clean, renames on frame 0, keeps `keep_open == True`, and re-points
`app.pass_settings_name` to the new name. Under a plain early return the rename frame never submits
the button; the observable difference is small enough that the two-clicks behaviour is ALSO pinned
as manual step 1b.

This test forces the branch rather than driving a focus transition because its subject is the crash,
which needs exactly one frame. The transition itself IS drivable headlessly (the five-frame recipe
in `test_a_picker_inline_input_commits_on_click_away` below), and that test pins the D11 trigger;
manual verification items 1, 1b and 2 cover the pass-settings field's own three close paths in the
real window.

### `tests/test_pass_verbs.py::test_a_rejected_rename_snaps_the_buffer_back`

Calls `pass_settings._commit_pass_name` directly (GL-free, no imgui frame — it takes `app`,
`document_id`, `name`) in three cases:

- a name `_pass_name_error` rejects (`"2fast"`) → returns `False`, pushes a notification, leaves
  `app.pass_settings_name_buf == name`;
- an existing pass's name → same;
- an accepted new name → **returns `True`** and the pass is renamed.

The third case is what pins `_commit_pass_name`'s positive return, which is the value `_draw_name`
forwards and `_draw_body` branches on. Together with the test above, the whole return path is
covered without any stub supplying a rename.

**Falsifier:** without the buffer reset, the bad text survives and the next deactivate re-fires the
same rejection, so the buffer assertion goes red. Without the `True` on acceptance, `_draw_body`
would draw the sections against the dead name and the crash test goes red.

### `tests/test_pass_verbs.py::test_add_pass_activates_the_new_pass`

Calls the same sequence `_draw_add_input`'s success branch runs (`add_pass`, then the activate
block) via a small helper the widget and the test share, or by asserting the post-conditions after
driving the widget through the imgui rig. Asserts: the new pass is `document.graph.output`, an
editor tab exists for its path, and `app.popup_state == PopupState.PASS_SETTINGS` with
`app.pass_settings_name` naming it.

**Falsifier:** today's code opens only the gear, so the output stays on the previous pass and the
output assertion goes red.

### `tests/test_lazy_compile.py::test_every_pass_renders_once_within_n_frames`

Loads the five-pass bloom example (`_BLOOM`, already a fixture constant in that module) with an
output whose chain does not cover every pass, then runs the sweep loop by hand: for N frames, call
`document.begin_frame(frame)`, `document.render()`, then elect the first pass with
`first_render_done == False` and `document.render(target=name)`. Asserts every pass has
`first_render_done` within `len(document.passes)` frames.

**A second case pins the stamp-on-attempt posture, and it needs a pass that fails to compile.**
`_BLOOM`'s five passes all compile, so on that fixture a pass is stamped whether the stamp is
written on attempt or on success — the mutation is invisible. So the second case calls
`release_program("this is not glsl")` on one OFF-CHAIN pass first (the idiom
`test_a_broken_source_is_attempted_once` already uses in this module), then runs the same sweep
loop. Asserts the broken pass's `first_render_done` is True within `len(document.passes)` frames and
that no pass is elected twice.

**Falsifier:** without the sweep, a pass outside the output chain never gets `first_render_done` and
the first case goes red. With the stamp written inside `Pass.render` instead of `Document.render`,
the broken pass returns early at `core.py:358-359` and is never stamped — so it is re-elected every
frame, the sweep does not drain, and the second case goes red on the frame budget.

### `tests/test_lazy_compile.py::test_the_steady_state_draws_only_the_output_chain`

After the sweep has completed, run further frames and count draws per pass by wrapping each
`Pass.render` with a counter (a per-pass counter incremented in a monkeypatched bound method, or by
reading a texture the pass writes). Asserts that in a steady-state frame the passes drawn are
exactly `evaluation_order(document.graph, document.graph.output)` and that no pass draws twice.

**Falsifier:** a missing skip makes an ancestor of the elected pass draw twice in one frame, which
the per-pass counter sees directly. The unscoped-skip regression (every thumbnail blanking) is
**NOT** observable here — it needs the two output renders per frame that only `ui.py` issues
(`:265` preview + `:296` own canvas), and a frame loop calling `render()` once has nothing for the
skip to suppress. Its own test is the sibling below.

### `tests/test_lazy_compile.py::test_two_output_renders_in_one_frame_both_draw`

Inside one `begin_frame(f)`, call `document.render(canvas=foreign)` (standing in for `ui.py:265`'s
preview render), then `document.render()` (standing in for `ui.py:296`'s own-canvas render), then
`document.render(target=x)` for an off-chain pass. Asserts, via the same per-pass counter, that the
SECOND call drew the full output chain — not zero passes.

**Falsifier:** dropping the `target is not None` conjunct from the skip makes the second call find
every pass already stamped by the first and draw nothing, so the counter assertion goes red. That is
the thumbnail-blanking regression round 3 caught in the parent spec, and this is manual step 9's
headless half. It is the single most likely way this wave breaks something, and it is the one test
here whose absence would let the regression ship green.

### `tests/test_lazy_compile.py::test_a_target_render_does_not_complete_the_document_first_render`

`document.render(target=some_pass)` on a freshly loaded document; assert
`document.first_render_done` is still False, then `document.render()` and assert it is True.

**Falsifier:** leaving the condition at `canvas is None` lets a sweep render consume the live loop's
per-frame first-render budget, and the first assertion goes red. This is the exact sibling of the
existing `test_a_foreign_canvas_render_leaves_first_render_pending`, which stays as it is.

### `tests/test_lazy_compile.py::test_the_skip_does_not_fire_without_a_frame_counter`

Render a document twice with `target=` set and no `begin_frame` call between (so `_frame == -1`),
and assert the elected pass drew both times.

**Falsifier:** dropping the `self._frame >= 0` conjunct makes `-1 == -1` a skip, and the second
render draws nothing. This is the example-document freeze the round-3 review found.

### `tests/test_command_routing.py` — chord uniqueness, no edit

`test_no_two_specs_share_a_chord_in_overlapping_scopes` loops `COMMAND_SPECS`, so the two new specs
are covered the moment they are added.

**Falsifier:** binding `OPEN_PASS_SETTINGS` to a chord an existing GLOBAL spec already owns (Alt+S,
say) makes it red. Verified by reading the test: it compares every pair and asserts
`not scopes_overlap`, and both new specs are GLOBAL, which overlaps everything.

### `tests/test_help_content.py::test_shortcuts_section_lists_every_bound_command`

New assertion: every `CommandSpec` with a non-zero `default_chord` has its label AND its
`chord_to_str(default_chord)` in the shortcuts snippet.

**Falsifier:** the existing test only asserts that each populated CATEGORY appears and that the Help
command's own chord appears, so a new command in an already-populated category ships undocumented
and nothing goes red. With the new assertion, adding `ADD_PASS` without it reaching the generator
fails. (It does reach it — the generator enumerates `COMMAND_SPECS` — so this test pins the wire
rather than fixing a bug, which is what `dev_flow.md`'s "name the line that READS it" asks for.)

### `tests/test_lib_files.py::test_a_picker_inline_input_commits_on_click_away` (new module)

**The deactivate transition IS drivable headlessly** — verified by execution, and this corrects the
earlier draft of this spec, which assumed it was not. The recipe, confirmed on this box:

1. frames 0-1: `imgui.set_keyboard_focus_here(0)` before the input, so it becomes active;
2. frame 2: `imgui.get_io().add_input_character(ord("X"))` BEFORE `new_frame()` — a real keystroke,
   which is what makes the later deactivate an "after **edit**" rather than a bare deactivate;
3. frame 3: `imgui.set_keyboard_focus_here(1)` to move focus to the next item;
4. frame 4: `is_item_deactivated_after_edit()` reads **True** on the first input.

Without the injected character the transition fires `is_item_deactivated()` but NOT
`is_item_deactivated_after_edit()`, which is exactly the distinction D11 rests on — so a test that
only moves focus would pass against an implementation that committed on any deactivate.

This one test drives ONE picker input (the file-rename input in
`popups/lib_picker/tree.py::_draw_file_rename_input`, chosen because its commit is a real filesystem
rename that the test can assert on disk) end to end through the rig: open the rename input on a
seeded lib file, run the five-frame recipe with a new name in the buffer, and assert the file moved
on disk and the input closed.

**Falsifier:** with the site left on Enter-only (today's code), the click-away commits nothing, the
file keeps its old name, and the assertion goes red. With the commit wired to a bare
`is_item_deactivated()` instead, a focus move with no edit would also rename — a second case with no
injected character asserts the file did NOT move, which goes red under that mistake.

The other three picker sites (new file, new dir, new tag) share the rule and two of them share the
very function this test drives (`_draw_inline_new_input` serves both new file and new dir), so they
are covered by manual verification items 12-14 rather than by three more rig tests. The pass-settings
name field and the add-pass input are covered by their own tests plus manual items 1, 1b, 2 and 3 —
the same recipe would work for them, and adding it there is cheap if the manual step ever proves
awkward.

## Manual verification

The parent spec's W-C line, one falsifiable step per item.

1. **Rename by typing and clicking elsewhere.** Open a document with at least two passes. Open the
   gear on `main`, type `paint` into the name field, and click on the Reads combo below it without
   pressing Enter. Expect: the modal stays open, the title row reads `paint`, the Reads section
   redraws against `paint`, and no crash. Fails if the app closes, if the field snaps back to
   `main`, or if the frame throws.
1b. **Rename by typing and clicking Close.** Reopen the gear, type a new name, and click the Close
   button directly without pressing Enter. Expect: the rename lands AND the modal closes on that one
   click. Fails if it takes two clicks — that is the interaction item 1 guards against, where the
   deactivate commit fires first and a plain early return would exit before the Close button was
   ever submitted.
2. **Rename by typing and pressing Escape.** Reopen the gear, type `scene`, press Escape. Reopen the
   gear. Expect: the name is `scene`. Fails if it reads the old name — that is finding #18's
   "closed, reopened, `main` was back" and the Escape path is the one item 3 exists for.
3. **A rejected rename.** Open the gear, type the name of another existing pass, click away. Expect:
   one notification naming the collision, and the field showing the pass's real name again. Fails if
   two notifications appear (the buffer was not reset) or if the field keeps the bad text.
4. **Add pass activates.** Click `add pass`, type `blur`, press Enter. Expect, on the same frame:
   the `blur` shader tab is open in the editor, the viewer shows `blur`'s output, `blur`'s tile
   carries the accent border (it is the output), and the gear is open on `blur` with its size row
   showing the output-pass help text. Fails if the viewer still shows the previous output, or if the
   tile is dim.
5. **Alt+P opens the active pass's gear.** With a shader tab focused on a non-output pass, press
   Alt+P. Expect: the gear opens on THAT pass, not on the output. Then click the Document tab (so no
   shader tab is active) and press Alt+P: the gear opens on the output pass. Fails if either press
   opens the wrong pass, or if the chord is dead while an input is focused. With the gear ALREADY
   open, Alt+P does nothing — `popup_suppresses` returns True for every scope, so a modal owns the
   frame. That is correct, not a dead chord, and the same holds for Alt+A.
6. **Alt+A opens the add-pass input with the caret in it.** Press Alt+A from anywhere in the app with
   no modal open, and type immediately without clicking. Expect: the characters land in the add-pass
   input. Fails if the input opens unfocused (the one-shot `needs_focus` did not arm).
7. **Alt+P and Alt+A appear in Help.** Open Help (F1) → Keyboard shortcuts. Expect: "Pass settings
   Alt+P" and "Add pass Alt+A" under Tools. Fails if either is missing.
8. **Reopen on a six-pass document.** Quit and reopen the app on the Radiance Cascades example (six
   passes, output `composite`). Watch the strip. Expect: every tile shows a real picture within a
   second, and the tiles outside the output chain carry the grey stale wash OVER a picture, not a
   black rectangle. Fails if any tile is still black after a couple of seconds.
9. **The strip does not blank.** With the app settled on a multi-pass document, watch the thumbnails
   for several seconds. Expect: every tile in the output chain animates. Fails if the thumbnails
   freeze or go black — that is the unscoped-skip regression, and it is the single most likely way
   this wave breaks something.
10. **The examples popup still animates.** Open Examples (Alt+E) and watch the grid for a few
    seconds. Expect: the example thumbnails keep rendering. Fails if they freeze after one frame —
    that is the `_frame == -1` regression.
11. **Export is unchanged.** Render an image from the Render tab on a document with an off-chain
    pass. Expect: the same image as before the wave. Fails if the export time visibly grows or the
    picture changes.
12. **Lib picker: new file by click-away.** Open the shader library (Ctrl+P), right-click a
    directory → New file, type a name, and click on the tree elsewhere without pressing Enter.
    Expect: the `.glsl` file is created and opens. Fails if nothing happens, or if a name that
    collides silently does nothing (it must show the existing "rejected: target exists"
    notification and leave the input open).
13. **Lib picker: new directory by click-away.** Same, via New dir. Expect: the directory appears.
    Same shared function as step 12, so a failure in one implicates both.
14. **Lib picker: new tag by click-away.** Select a function, type a tag in the add-tag field, and
    click elsewhere in the preview pane. Expect: the tag pill appears and the field clears. Then
    repeat using the `+ Add` button and using Enter: all three must work, since the button stays as
    the shortcut rather than being replaced. Fails if any of the three paths stops adding, or if
    Escape stops cancelling the picker's inline inputs.

## Verified / corrected premises

Every citation and claim the parent spec's W-C section makes, checked against the tree at
`faccf0e`. Line numbers below are the real ones as of this reading.

| Parent-spec citation or claim | Verdict |
|---|---|
| `_draw_name` performs the rename in the same frame the body continues with a stale local | **Confirmed.** `pass_settings.py:75` reads `name = app.pass_settings_name` once; `_draw_name` at `:81` renames; `:84` indexes `document.passes[name]`. |
| The crash is `KeyError` at `pass_settings.py:84 _draw_inputs(app, document_id, name, document.passes[name])` (#17) | **Confirmed, and reproduced.** Driving `_draw_body` through a headless imgui frame with a rename inside `_draw_name` raises `KeyError: 'main'` at `pass_settings.py`, line 84, in `_draw_body`. |
| `rename_pass` pops `document.passes[old]` at `project_session.py:821` (#17) | **Corrected.** `ProjectSession.rename_pass` is defined at `project_session.py:806` and the `document.passes.pop(old)` line is `:821` (`:822` is `old_path = render_pass.source.path`). Per `conventions.md ## Code rules` the durable citation is the symbol, `ProjectSession.rename_pass`, not either number. |
| `_on_pass_renamed` re-points `app.pass_settings_name` at `app.py:565` (#17) | **Confirmed.** `app.py:565-567` is `if self.pass_settings_name == old_name:` and the two re-points, inside `App._on_pass_renamed` (defined at `:550`). |
| `input_text(..., enter_returns_true)` at `pass_settings.py:96` (#18) | **Confirmed.** `:96` is the `committed, app.pass_settings_name_buf = imgui.input_text(` line; the flag is at `:99`. |
| `open_pass_settings` resets the buffer to the live name at `app.py:895` (#18) | **Confirmed.** `App.open_pass_settings` is at `:894`; `:896` is `self.pass_settings_name_buf = name`. (The finding cites `:895`, which is `self.pass_settings_name = name` — the adjacent line of the same two-line reset.) |
| `_pass_name_error` rejects intermediate names, `project_session.py:116` (#18) | **Confirmed.** `_pass_name_error` is defined at `project_session.py:116`. |
| `rename_pass` is transactional, `project_session.py:818` (#18) | **Confirmed as a claim, corrected as a location.** The transactional comment block is `:818-820`; the function starts at `:806`. |
| Alt+P is free; `K.p` is Ctrl+P (`commands.py:167`) and Ctrl+Shift+P (`:172`) (#9) | **Confirmed.** `:167` is the `OPEN_LIB_PICKER` spec line, `:172` the palette's `_chord(K.p, K.mod_ctrl, K.mod_shift)`. The only `mod_alt` chords in the table are Alt+S (`:175`), Alt+E (`:176`) and Alt+/ (`:183`). Alt+A is free. |
| A handler goes in `app.py:487`'s dispatch table (#9) | **Corrected.** The dispatch table is `App._build_command_callbacks`, `app.py:479-516`; `:487` is the `CommandId.HELP` entry inside it. The table is the right place; the line is off. |
| "Currently selected pass" = `App.panel_pass(document_id)` at `app.py:571` (#9) | **Confirmed.** `panel_pass` is defined at `app.py:571`. It returns a `Pass`, not a name — the callback must convert via `pass_name_of(render_pass.source.path)`, which the parent spec's `open_pass_settings(panel_pass(...).name)` shorthand elides (`Pass` has no `.name` attribute). |
| The chord-uniqueness test and the generated Help table pick the new commands up from `COMMAND_SPECS` | **Confirmed for both, with one gap.** `test_command_routing.py::test_no_two_specs_share_a_chord_in_overlapping_scopes` loops `COMMAND_SPECS`; `help_content.py::_shortcuts_section` enumerates it per category. But `test_help_content.py::test_shortcuts_section_covers_every_populated_category` asserts only that each populated category appears and that the Help command's chord appears — a new command in an already-populated category (which `C.TOOLS` is) is not pinned. Hence the new assertion in § Tests. |
| `add pass` is a ghost button at `pass_list.py:178` (#25) | **Corrected.** The `if ghost_button("add pass"):` line is `pass_list.py:173`. |
| Add pass opens only the gear, at `pass_list.py:193` (#28) | **Corrected.** `_draw_add_input`'s success branch is `pass_list.py:193-201`; `:193` is `if committed:` and the `app.open_pass_settings(name)` is `:201`. The claim holds. |
| A tile click does `open_pass(name)` + `set_output_pass` at `pass_list.py:122` (#28) | **Corrected.** The click branch is `pass_list.py:123-130` (`if result.clicked:` at `:123`, `open_pass(name)` at `:126`, `set_output_pass` at `:128`). `:122` is a blank line. |
| "Same for the copilot's add-pass path if it has one (check `copilot/tools/document_ops.py`)" (#28) | **Refuted — there is none.** `grep -rn 'add_pass\|rename_pass\|delete_pass\|set_output_pass' shaderbox/copilot/` returns nothing. The parent spec already states this ("Copilot has no pass tools — nothing to mirror"); confirmed. |
| `Document.render` draws only `evaluation_order(graph, output)` at `document.py:385` (#36) | **Corrected.** `Document.render` is defined at `document.py:386`. The claim holds: `:398` plans for the output and `:404` loops that order. |
| `document.py:410` skips the scale for the output; `:411` sizes the others from `document.canvas_size` (#4, #2) | **Confirmed.** `:410` is `if name != output:` and `:411` is the `target_size(self.canvas_size)` call. |
| A never-drawn pass's canvas is a fresh black texture, `core.py:181` (#36) | **Corrected.** `core.py:181` is inside `Pass.set_target` (`self.canvas = Canvas(size=size, ...)`). The `__init__` allocation is `core.py:157-159`. Both allocate a zeroed texture, so the claim holds. |
| The "don't pay for unread branches" rule lives at `pass_graph.py:367` (#36) | **Confirmed.** `evaluation_order` is defined at `pass_graph.py:367` and its docstring carries that sentence. |
| The stale wash + corner tick is `pass_list.py:153` / `ui_primitives.py:969` (#36) | **Corrected for `pass_list.py`, confirmed for `ui_primitives.py`.** `pass_list.py:153` is a comment line about the stale tick; the wash is applied by passing `stale=` to `preview_cell` at `pass_list.py:119`, computed from `live` at `:157-161`. `ui_primitives.py:969` is the `stale` docstring line inside `preview_cell`. |
| "`target` (or the graph output when None) feeds the early-out guard, `plan_for_output`, and the cycle fallback" | **Confirmed, three sites.** `document.py:395` (the guard), `:398` (`plan_for_output(self.graph, output)`), `:400-403` (`if not order: order = [output]`). Each reads the local `output`, so the change is one rebinding plus three substitutions. |
| "A target draws its WHOLE ancestor chain" comes free from the planner | **Confirmed.** `pass_graph.py::_order_for` walks `plan.reads` transitively with an explicit stack (`:384-392`) and returns them in `plan.order`, so `plan_for_output(graph, any_pass)` already yields the ancestor chain. |
| "The two `name == output` comparisons use the graph output, unchanged" | **Confirmed, and one hazard the parent spec does not name.** The comparisons are `document.py:410` and `:425`. `:425` already binds a local named `target` (`target = canvas if (name == output and last) else None`), which a parameter of the same name shadows for the rest of the loop. The local is renamed `draw_into` in this wave. |
| `Pass.first_render_done` is a new field on `core.Pass` | **Confirmed.** `Pass.__init__` (`core.py:139-168`) has neither `first_render_done` nor `drawn_frame`. The parent spec's Files list names `core.py` for `first_render_done` only; `drawn_frame` lands there too. |
| "`Pass.drawn_frame: int`, set by every `render` against the document's `_frame` from `begin_frame`" | **Corrected on the setter.** It cannot be set by `Pass.render`: that method returns early at `core.py:358-359` when the program failed to compile, so a broken pass would never be stamped and the sweep would re-elect it every frame. `Document.render` sets both fields in its own loop, on ATTEMPT — the same posture `Document.first_render_done` already takes. |
| "Output renders never skip: the preview render (`ui.py:265`) and the own-canvas render (`ui.py:301`)" | **Confirmed for `:265`, corrected for `:301`.** The preview render is `ui.py:265` (`document.render(canvas=app.preview_canvas)`). The own-canvas render is `ui.py:296` (`ui_document.document.render()`), inside the `for document_id in tick_documents:` loop. `:301` is inside the examples-popup branch's `next(...)` generator. The underlying claim — that the two output renders in one frame must both draw — is exactly right and is the reason the skip is conditioned on `target is not None`. |
| "The examples popup (`ui.py:298`) renders whole documents and never a target" | **Confirmed as a claim, corrected as a location.** `:298` is a comment; the examples branch is `ui.py:297-310`, with the render at `:310`. The claim holds and matters twice over: those documents also never get `begin_frame`, so their `_frame` is `-1` forever (verified: the only `begin_frame` call sites in `shaderbox/` are `ui.py:246`, `document.py:527` and `document.py:619`). Hence the `self._frame >= 0` conjunct. |
| "Target renders are issued only by the main frame gate over ticked documents, where `_frame` is defined" | **Confirmed by construction** — this wave adds the only `render(target=)` call site, and it is inside the `tick_documents` loop, which `ui.py:246` has already called `begin_frame(app.frame_idx)` over. |
| "`Document.first_render_done` keeps its meaning (output drawn, `canvas is None and target is None`)" | **Confirmed as the right narrowing.** Today's condition is `canvas is None` (`document.py:392-393`). The new conjunct is strictly narrower, so `test_lazy_compile.py::test_a_foreign_canvas_render_leaves_first_render_pending` is unaffected. |
| "Export is untouched: `target` is not threaded through `_render_image` / `_render_media_into`" | **Confirmed.** `_render_image` is `document.py:520`, `_render_media_into` is `:668`; both call `self.render(...)` with `canvas=` only, and `render_media` (`:641`) calls `reset_feedback()` (which sets `_frame = -1`) before either. No edit needed on any of the three. |
| "The draw-once invariant assert in `pass_graph.py` stays the guard" | **Confirmed.** `assert_plan_invariants` (`pass_graph.py:311`) runs inside `plan_for_output` (`:363`), which every render path calls, so a target render is asserted on the same terms as an output render. |
| A target render overwrites document-scoped `_graph_errors` | **Confirmed as a fact, refuted as a hazard.** `document.py:398` assigns `_graph_errors` from `plan_for_output`. Its error half comes from `plan_passes(graph)`, which walks every name in `graph.passes` and is target-independent; `_order_for` filters only the order. And `Document.graph_errors` has no production consumer — `grep` finds it read only in three test modules. |
| `tests/test_pass_verbs.py` imports `_strip_order` from `widgets/pass_list.py` (W-D's note, relevant here because W-C edits that file) | **Confirmed.** `tests/test_pass_verbs.py:28`. W-C changes `_draw_add_input`'s signature and `draw`'s call to it, neither of which `_strip_order` touches. |
| `tests/test_pass_editor_wiring.py:173` is the test that names the region machinery (W-E's target, listed here because W-C touches neither) | **Confirmed.** `:173` is `def test_a_summon_from_a_non_editor_region_yields_the_editor_back`. W-C does not touch it. |
| A test already exercises `Document.render` frame accounting | **Confirmed.** `tests/test_lazy_compile.py::test_a_foreign_canvas_render_leaves_first_render_pending` pins the `canvas is None` half of `first_render_done`. It is the natural home for the sweep's tests, and its module fixtures (`_TUNED`, `_BLOOM`, a standalone GL context) are what those tests need. |
| The imgui-context test rig from `/imgui-ui § 0` can drive `_draw_body` headlessly | **Confirmed by running it.** `App.__init__` calls `imgui.create_context()` (`app.py:172`) and builds a `GlfwRenderer` (`:193`), so the `app` fixture already carries a live context; `new_frame` / `begin` / body / `end` / `end_frame` runs clean and reproduces #17's crash. No test in `tests/` creates an imgui frame today, so this is the first. |
| "`is_item_deactivated_after_edit()` cannot be driven headlessly" (this spec's own earlier claim) | **Refuted by execution.** Focusing an input with `set_keyboard_focus_here(0)`, injecting a real keystroke with `io.add_input_character(...)` before `new_frame()`, then moving focus with `set_keyboard_focus_here(1)` makes the query read True two frames later. Without the injected character only `is_item_deactivated()` fires, never the after-edit form — which is the distinction D11 rests on, so the recipe also gives the mistake its own falsifier. The D11 wiring is therefore pinned by a test, not only by manual steps. |
| `/imgui-ui § 7.5` says Enter commits (the text W-C rewrites) | **Confirmed.** § 7.5's Pattern bullet reads "`imgui.input_text(..., flags=enter_returns_true)` — Enter commits, Esc cancels." |
| Escape closes the gear before its body draws | **Confirmed, and it is why the commit cannot live in the body.** `dispatch_commands` (containing `_handle_escape`, which sets `PopupState.CLOSED` at `hotkeys.py:303`) is called at `ui.py:336`; `draw_pass_settings` is called at `ui.py:429`. On the Escape frame the body's first guard returns. |

Corrected or refuted: **14** (11 corrections, 3 refutations). Two of these came after the first
draft: round 1 of review corrected one of the corrections itself (the `document.passes.pop(old)`
line is `:821`, not `:822` — the row now cites the symbol instead), and this spec's own claim that
`is_item_deactivated_after_edit()` could not be driven headlessly was refuted by running it.

## Open questions

Each carries a robust default, marked as such; none blocks implementation.

1. **Should the sweep elect a pass per document or one across all documents?** Default, taken:
   **one per document per frame.** With "Render all documents" off, `tick_documents` holds one
   document and the two are identical; with it on, a per-document sweep converges N documents in
   `max(passes)` frames instead of `sum(passes)`, and the cost is bounded by the same set the frame
   already renders. Revisit if a project with many render-all documents stutters on load.
2. **Does the sweep need to re-arm after a hot reload of a not-live pass?** Default, taken:
   **no.** `watch.py::_reload_pass_if_changed` calls `release_program`, which calls `invalidate()`;
   neither clears `first_render_done`, so an edited off-chain pass keeps its last picture until it
   is next drawn. Finding #36's option A mentions "after add pass / hot reload of a not-live pass" as sweep
   triggers; add pass is covered by D10 (the new pass becomes the output and draws immediately), and
   the hot-reload case shows a stale picture under the stale wash rather than a black tile — which
   is the state the wash already means. Clearing `first_render_done` in `invalidate()` is the
   alternative and it is one line — but it is a **behaviour** change, not a cosmetic one: it breaks
   the sweep's termination property from "each pass is elected at most once" to "at most once per
   invalidate", so an off-chain pass being actively edited re-enters the sweep on every save. That is
   still bounded (one election per save) and probably fine, but a later wave must not take it as
   free.
3. **Should `_draw_add_input`'s deactivate-commit fire when the user clicks the `x` cancel
   button?** Default, taken: **no.** The cancel button's own branch calls `app.pass_add.close()`,
   and because the button is drawn after the input, the deactivate would fire on the same frame and
   create the pass the user just cancelled. The implementation reads the deactivate immediately
   after the `input_text` and captures its result, then applies it only if the cancel branch did not
   run. If that proves fiddly in practice, the alternative is to check
   `imgui.is_item_deactivated_after_edit()` and skip the commit when the mouse is over the cancel
   button — worse, and only if the first shape fails.
   The **Escape** branch, which also runs after the input and also calls `close()`, needs no
   suppression: on the Escape frame the item is still active so the deactivate reads False, and on
   the next frame `_draw_add_input` is not drawn at all (`widgets/pass_list.py`'s
   `if app.pass_add.is_open:` guard). Only the cancel button's branch — which runs in the same frame
   as a deactivate the click itself caused — needs the capture-then-apply shape.
4. **Where does `Pass.drawn_frame` reset when a pass's canvas is reallocated?** Default, taken:
   **nowhere.** `set_target` replaces the canvas, and the next frame's output render or sweep
   re-stamps it; a stale stamp can only suppress a redraw within the SAME frame the target changed,
   which is one frame of the old picture. In practice it is zero frames: the target only changes
   from the gear, the gear is a modal, and the sweep lives inside `ui.py`'s
   `if not app.any_popup_open():` branch — so no sweep runs on the frame a target changes at all.
   Adding a reset in `set_target` would be defensible; it is not added because nothing observes the
   difference.

## Review history

**Round 1, pre-implementation review** (`reviews/wave_c_pre.md`, one reviewer, correctness & design
plus verification & blast-radius): parent-bullet coverage PASS, locked-decision fidelity PASS,
sweep-algorithm correctness PASS (all three scenarios hand-traced, termination proved), test
falsifiability PARTIAL. Seven findings, **all accepted, none rejected.**

Folded:

1. **The `input_text` census was partial** (7 of 22 sites unlisted, two of them credential fields —
   the settings API key and the copilot gate secret). Item 6 is rewritten around ONE rule stated
   once: D11 applies to an input whose commit fires an ACTION, not to a form field whose value IS
   the state and is applied by a separate Save / Apply / Connect control. All 22 sites are now
   classified. (Round 1 landed this as 2 in scope with 5 deferred by wave boundary; the override
   below moves the four picker inputs in scope, so the current split is 6 in scope, 3 whose button
   stays the only commit, and 14 form fields.)
2. **The Close button would have needed two clicks** after an un-entered rename: clicking Close
   fires the deactivate commit, and a plain early return exits before `ghost_button("Close")` is
   submitted, so imgui cannot see the click. Item 1 now guards only the two sections that index
   `document.passes` by the dead name; the Close row draws on every path. The header changed with
   it, and manual step 1b pins the behaviour.
3. **The rename test could not observe `_draw_name`'s return** while stubbing `_draw_name`. It now
   monkeypatches the leaf `_commit_pass_name` instead, so the real `_draw_name` runs and the whole
   plumbing is pinned; `test_a_rejected_rename_snaps_the_buffer_back` gains a third case asserting
   the accepted-rename `True`.
4. **The steady-state test over-claimed its falsifier**: the thumbnail-blanking regression needs the
   two output renders per frame that only `ui.py` issues, which a one-render-per-frame loop cannot
   reproduce. The claim is withdrawn from that test and given its own sibling,
   `test_two_output_renders_in_one_frame_both_draw`.
5. **The once-within-N-frames test could not see the stamp-on-success mutation** because every
   `_BLOOM` pass compiles. A second case breaks one off-chain pass with
   `release_program("this is not glsl")` first, which makes the mutation visible as a sweep that
   never drains.
6. **`project_session.py:822` corrected to `:821`**, and the row now cites
   `ProjectSession.rename_pass` as the durable reference per `conventions.md ## Code rules`.
7. **Two history-narrating comments cut** from the proposed code: `close_pass_settings`'s docstring
   loses its `(069 D11)` tag, and the two `Pass` fields lose their four-line story — `drawn_frame`
   keeps a one-line statement of the invariant, `first_render_done` needs none. The reasons live
   here and in the commit message. The `close_pass_settings` idempotence sentence also now credits
   the callee, which returns `""` for `new == old` on its own.

Reasoning corrections folded into the open questions: clearing `first_render_done` in `invalidate()`
is a behaviour change to the sweep's termination property, not a cosmetic one (OQ 2); the Escape
branch of `_draw_add_input` needs no commit suppression and why (OQ 3); the gear is a modal so no
sweep runs on the frame a target changes, making OQ 4's window zero frames rather than one (OQ 4).
Manual step 5 gained the `popup_suppresses` clause: Alt+P is inert while the gear is already open,
and that is correct.

**Override after review, from the maintainer's parent-spec rule (applied by the main session).**
Round 1 accepted item 6's deferral of the four shader-library picker inputs "to whichever wave next
opens the picker". That is a deferral to nowhere — no 069 wave opens the picker — and this repo does
not park work in docs. The parent spec says the rule applies to "`_draw_add_input` and **every § 7.5
inline input**", and § 7.5 is titled "Inline inputs inside modals (rename / new-file / new-dir)",
which names those very inputs. So `lib_picker/tree.py`'s new-file, new-dir and file-rename inputs
and `lib_picker/preview.py`'s new-tag input move **in scope**, with per-site commit semantics stated
and their two files added to § Files touched. The in-scope set becomes 6 of the 9 action rows; the
remaining table is retitled "Action inputs that keep their button as the only commit" because
nothing is deferred any more — the Telegram pack title, the YouTube paste box and the copilot
credential gate stay out for the reasons already given, and those are properties of those sites, not
scheduling.

Folding the override turned up one more correction, this time to this spec's own claim: **the
deactivate-after-edit transition IS drivable headlessly.** Focusing an input, injecting a real
keystroke with `io.add_input_character(...)`, then moving focus makes
`is_item_deactivated_after_edit()` read True two frames later — and without the injected character
only the bare `is_item_deactivated()` fires, which gives the D11 trigger its own falsifier. The
"commit-on-deactivate is manual only" note is replaced by
`test_a_picker_inline_input_commits_on_click_away`, and manual items 12-14 cover the three sibling
picker sites.

The reviewer independently reproduced finding #17's `KeyError` against the current tree, verified
the proposed body is a balanced imgui frame across three frames, simulated the sweep for three graph
shapes, and re-ran every grep the spec cites. Its § False trails list (the `graph_errors` overwrite,
the `Pass` / `Document` `first_render_done` name pair, whether the sweep can write into the preview
canvas, whether the examples popup can issue a target render, whether `add_pass` can create a pass
that is already the output) is recorded there as probed-and-fine; none is re-litigated here.
