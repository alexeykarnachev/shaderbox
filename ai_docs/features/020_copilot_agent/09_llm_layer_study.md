# 020 Copilot — 09: LLM-layer study (OpenRouter seam yanked from cc-server)

> Research report, angle: **the low-level LLM layer only.** What reusable client seam + cross-cutting
> concerns (cost/usage tracking, tooled responses, structured output) to lift from the maintainer's
> existing Python LLM code, and the clean module skeleton ShaderBox should adopt. **RESEARCH ONLY —
> no production code here, only design + module shapes.**
>
> **Overrides `04_llm_integration.md`.** That report assumed the **Anthropic SDK** (Messages API,
> `input_schema`, `tool_result` blocks, prompt caching). The maintainer has since decided: **provider
> is OpenRouter** (OpenAI-chat-completions-shaped), cheap model TBD (codex / grok / gemini — model
> choice deferred, it's a string id). Everywhere `04` says "Anthropic / input_schema / tool_result
> block / official anthropic SDK," read instead "OpenRouter / `parameters` / `role:"tool"` message /
> the `openai` SDK (or raw httpx) pointed at OpenRouter." §-by-§ delta is called out in §3.0.
>
> Primary source studied (re-read on disk, cited file:line below): **cc-server** `core/ai/llm/*` +
> `core/ai/chat/agent.py` (Python, OpenAI-shaped, already routes through OpenRouter, has cost tracking
> + tool-use loop). Seam-shape comparison: **ovelia** `llm/api.py`+`llm.py` (the cleanest
> provider-neutral Protocol, OpenRouter-backed, has the IPv4 pin + structured-output seam) and
> **marginalia** `agent.ts` (the raw OpenRouter SSE wire shape, TS).

---

## 0. Recommendation (read this first)

**Lift ovelia's *seam shape*, lift cc-server's *cost-extraction + tool-loop mechanics*, drop
cc-server's entire *billing/account/operation-role* server machinery.** The two are complementary:
ovelia already did the "strip cc-server down to a desktop-shaped provider-neutral seam" exercise (it's
single-tenant-ish, OpenRouter-backed, has the IPv4 pin, has structured output). cc-server is the
authority on the *details* — how usage+cost are pulled off the response (`_extract_usage_from_response`,
`client.py:45`), how a tool turn is shaped and looped (`agent.py:237`), the double-escape repair
(`agent.py:128`). ShaderBox should be: **ovelia's interface + cc-server's extraction guts − the server billing.**

### 0.1 The LLM-client seam (the one interface)

A small **provider-neutral `LLMClient` Protocol** (ovelia's `ILLMService` is the template, but
ShaderBox's is **synchronous** — the copilot loop runs on the existing worker thread, not an async
server runtime, so no `AsyncIterator`/`async def`). Three methods cover the whole feature:

```
class LLMClient(Protocol):
    def stream(messages, *, tools=None, max_tokens) -> Iterator[LLMStreamEvent]: ...
    def complete_text(prompt, *, max_tokens) -> str: ...                          # non-streamed convenience
    def complete_structured(prompt, *, schema: type[T], max_tokens) -> T: ...     # JSON-schema response
```

- `stream(...)` is the workhorse: chat-with-tools, streamed. Yields the **typed stream events**
  (text deltas, tool-call-started/completed, done-with-usage). The **agent loop lives ABOVE this** and
  consumes the events — the client does NOT loop, does NOT execute tools, does NOT assemble prompts.
- `complete_structured(...)` is the structured-output seam (§4) — only if a v1 capability actually
  needs it (a planning/triage step). Otherwise it's a stub-for-later; don't build it speculatively.
- The model id is a **constructor field** of the OpenRouter impl (read from settings), NOT a per-call
  param — there's one provider, one user-chosen model. (cc-server's per-call model-role resolution is
  server cruft — §7.)

### 0.2 The module skeleton (the deliverable the maintainer asked for)

A `copilot/` package whose **LLM layer is one leaf sub-package** with strict one-way imports:

```
copilot/
  llm/
    api.py        # typed seam: dataclasses + LLMClient Protocol. Imports NOTHING from copilot. LEAF.
    openrouter.py # the impl: openai/httpx → OpenRouter. Imports llm/api only.
  prompt.py       # prompt assembly: system prompt + history → list[LLMMessage]. Imports llm/api only.
  tools/          # tool registry + specs + executors (OTHER agent's angle — 02). Imports llm/api for LLMToolSpec.
  agent.py        # the loop: orchestrates client.stream + tools + prompt. Imports llm/api, prompt, tools.
```

Import direction is a strict DAG, leaf-first (no cycles, satisfies ShaderBox's "no `if TYPE_CHECKING`"
ban — there's nothing to break): `api.py` ← everything; `openrouter.py` ← only `api`; `prompt.py` ←
only `api`; `agent.py` ← `api`+`prompt`+`tools`. **The four boxes never import sideways** (prompt
doesn't import tools, the client doesn't import the loop). Detail in §5.

### 0.3 Top 3 things NOT to port from cc-server

1. **The whole billing/usage-counter stack** (`LLMService` wrapper, `IUsageService.record_llm_usage`,
   `core/billing/usage/counter.py`'s per-account day/month/total buckets, microdollar integer storage,
   limit-checking). It's multi-tenant SaaS metering. ShaderBox is single-user; the user pays their own
   OpenRouter bill. Cost tracking collapses to **one in-memory rollup dataclass** (§2).
2. **`LLMTask` / `LLMOperation` / `billable` / the `_OPERATION_ROLES` model-role map** (`api.py:24-60`,
   `llm.py:42-56`). That's "which of N billable server operations is this, and which account-toggled
   model does it map to." ShaderBox has one operation (copilot chat) and one user-chosen model. Delete
   the whole concept.
3. **The fallback-provider + GigaChat multi-client routing** (`client.py:128-158`, `_get_fallback`,
   `_fallback_model_map`), the embeddings path (`embed_texts`, the fixture loader), and the tiktoken
   `tokens.py` pricing/encoding machinery. ShaderBox has one provider, no embeddings, and gets token
   *counts + cost* straight from the OpenRouter response (§2.2) — no local tokenizer needed.

---

## 1. The reusable LLM-client seam — typed dataclasses

The seam is a set of **frozen dataclasses** (provider-neutral wire-agnostic types) plus the Protocol.
This is a near-direct lift of **ovelia `llm/api.py`** (the cleanest existing version) with cc-server's
`ToolCallData`/`LLMUsageInfo` cross-checked. ovelia already proves this exact shape works against
OpenRouter. Adapted to ShaderBox (sync, single-model):

```python
# copilot/llm/api.py — the seam. No `openai`/`httpx` import here; no copilot import here.

@dataclass(frozen=True)
class LLMToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]   # JSON Schema. NOTE: OpenRouter/OpenAI key is `parameters`,
                                 # NOT Anthropic's `input_schema` (the 04 report's name).

@dataclass(frozen=True)
class LLMToolCall:
    id: str                      # tool-call id; must echo back on the matching tool result message
    name: str
    arguments: str               # RAW JSON STRING — caller parses (may be malformed on truncation,
                                 # and may be double-escaped on some providers — §3.4)

@dataclass(frozen=True)
class LLMMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None           # None only on an assistant tool-only turn
    tool_call_id: str | None = None      # required when role == "tool"
    tool_calls: list[LLMToolCall] | None = None   # present on assistant turns that called tools

@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0        # OpenRouter returns this directly (§2.2) — no local pricing table

# stream events (ovelia llm/api.py:97-123) — the discriminated union the loop matches on
@dataclass(frozen=True)
class LLMTextDelta:        text: str
@dataclass(frozen=True)
class LLMToolCallStarted:  index: int; id: str; name: str
@dataclass(frozen=True)
class LLMToolCallCompleted: index: int; id: str; name: str; arguments: str  # raw JSON
@dataclass(frozen=True)
class LLMDone:            finish_reason: str; usage: LLMUsage   # "stop"|"tool_calls"|"length"|"content_filter"

LLMStreamEvent = LLMTextDelta | LLMToolCallStarted | LLMToolCallCompleted | LLMDone
```

**Why these exact types** (file:line provenance):

- `LLMToolCall.arguments: str` (raw JSON, caller parses) — cc-server `api.py:90-95` (`ToolCallData`)
  AND ovelia `api.py:66-71` (`LLMToolCall`) both make the same call: **the client returns the raw
  arguments string; parsing + validation is the loop's job.** This is the right seam — the client must
  not assume the args are well-formed (truncation, double-escape). cc-server's note at `api.py:95`:
  *"raw JSON string; caller must parse and validate (may be malformed on truncation)."* Keep it raw.
- `LLMMessage` provider-neutral (not the `openai` SDK's `ChatCompletionMessageParam`) — ovelia
  `api.py:73-80`. cc-server leaked the SDK's `ChatCompletionMessageParam` into its *public* `api.py`
  (`api.py:13`, `:118`), which is exactly the seam-leak ShaderBox should avoid: the impl translates
  `LLMMessage → wire dict` internally (ovelia `_to_wire_message`, `llm.py:70-104`). **Adopt ovelia's
  neutral message, not cc-server's SDK-typed one.**
- The streaming **tool-call-builder accumulation** (deltas arrive fragmented across SSE chunks, keyed
  by `index`, name+args concatenated) is non-trivial and is the part worth lifting verbatim — ovelia
  `llm.py:261-303` and marginalia `agent.ts:160-172` do the identical accumulation. §3.3.

### 1.1 What ShaderBox's seam drops vs cc-server's `ILLMService`

cc-server's interface (`api.py:113-174`) has eight methods: `completion_text`, `completion`,
`completion_stream`, `completion_with_tools`, `embed_texts`, `get_model_capabilities`, `count_tokens`,
plus every method threads `task: LLMTask, account_id`. ShaderBox needs **three** (`stream`,
`complete_text`, `complete_structured`), **no `task`/`account_id` params**, **no embeddings**, **no
`count_tokens`/`get_model_capabilities`** (those exist to plan a token budget against a tokenizer —
ShaderBox reads token counts back from the response instead; a budget cap is a simple int compare in
the loop, §2.3). This is the ruthless-cut the adversarial section (§7) demands.

---

## 2. Cost / pricing / usage tracking

### 2.1 How cc-server does it (and why most of it is server-only)

- **Extraction** — `client.py:45 _extract_usage_from_response()` reads `cost`, `prompt_tokens`,
  `completion_tokens`, `total_tokens` off the response's `usage` object and packs them into
  `LLMUsageInfo(input_tokens, output_tokens, cost_usd)` (`api.py:72-78`). **This is the reusable
  core.** Note `client.py:55`: `cost_usd = getattr(usage_data, "cost", None)` — **OpenRouter puts the
  dollar cost right on `usage.cost`**, cc-server just reads it. No local pricing table is consulted for
  cost — `tokens.py`'s pricing is only for *budget pre-estimation*, not for the actual charged cost.
- **Per-call recording** — `LLMService._record_usage()` (`llm.py:209`) forwards each call's usage to
  `IUsageService.record_llm_usage(account_id, usage, billable)` → `counter.py:195` which increments
  per-account `llm:<op>:input` / `:output` / `:cost` counters in **day+month+total buckets**, cost
  stored as **integer microdollars** (`counter.py:216 int(cost_usd * 1_000_000)`). **This entire path
  is multi-tenant metering — drop all of it.**
- **Per-run rollup** — `agent.py:43 AgentRunStats` accumulates across the loop's iterations:
  `total_input_tokens`, `total_output_tokens`, `total_cost_usd`, plus `llm_calls: list[LLMUsageInfo]`
  and `tool_calls`. `record_llm_call(usage, finish_reason)` (`agent.py:55`) is called once per LLM
  turn. **THIS is the part ShaderBox wants** — it's the in-memory per-conversation-turn rollup, with
  no DB, no account. The server then maps it to API response fields (`counter_models.py:63-65`
  `total_input_tokens/total_output_tokens/total_cost_usd`, microdollars `/1_000_000` at
  `counter_endpoint.py:212`) — that mapping is server cruft, but the rollup dataclass itself is exactly
  right for a desktop chat that wants to show per-turn + cumulative cost.

### 2.2 Does OpenRouter return cost? Yes.

Confirmed three ways in-tree: cc-server reads `usage.cost` (`client.py:55`); ovelia reads
`getattr(u, "cost", 0.0)` (`llm.py:124`); marginalia logs `usage` from the stream chunk
(`agent.ts:175`). OpenRouter returns a `usage` block on the chat-completion response, and **when you
pass `usage: {include: true}` (or the SDK's `stream_options={"include_usage": True}`) it includes
`cost` (USD) alongside `prompt_tokens`/`completion_tokens`.** So ShaderBox gets the real charged dollar
amount per call **for free, off the response** — no pricing table, no tiktoken. (Verify-at-impl in §6
that the streaming usage chunk carries `cost`; ovelia logs `llm_cost_missing` (`llm.py:313`) when a
streamed `usage` arrives with `cost=None`, so handle the None gracefully — show tokens, cost "n/a".)

### 2.3 The minimal ShaderBox version

One dataclass, in-memory, owned by the agent loop / chat session — a stripped `AgentRunStats`:

```python
# copilot/agent.py (or a tiny copilot/usage.py if shared with the UI)
@dataclass
class TurnUsage:                     # per user→assistant turn (may span N LLM iterations)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    def add(self, u: LLMUsage) -> None:
        self.input_tokens += u.input_tokens
        self.output_tokens += u.output_tokens
        self.cost_usd += u.cost_usd
```

- **Where cost logging lives:** the **client emits** `LLMUsage` (on `LLMDone`); the **loop
  accumulates** it into `TurnUsage` (cc-server's `record_llm_call`, `agent.py:55-60`). The client must
  NOT own a running total — it's stateless per call (cc-server's `LLMClient` is likewise pure,
  `client.py:106-109` "No billing or usage recording"). A **session-level** cumulative
  (`SessionUsage` = sum of all `TurnUsage`) lives wherever the chat session state lives, for the
  "total spent this session" line.
- **Where it surfaces:** the chat UI shows per-turn `↑{in} ↓{out} ~${cost:.4f}` under the assistant
  message and a session cumulative in the panel header. (UI placement is `05_chat_ui_ux.md`'s call;
  this report only fixes that the *data* is the `TurnUsage`/`SessionUsage` rollup, fed by `LLMUsage`
  off `LLMDone`.)
- **Budget cap** (optional): cc-server's `over_input_budget` (`agent.py:65`) is a one-line int compare
  the loop can keep, to stop a runaway tool loop from silently burning the user's money — but it's a
  loop concern, not a client concern. Cheap to keep, no tokenizer needed (compare accumulated
  `input_tokens`).

---

## 3. Tooled responses — the OpenRouter wire shape

### 3.0 Delta vs the Anthropic-shaped `04` report (the override, explicit)

| Concern | `04` (Anthropic, OVERRIDDEN) | This report (OpenRouter / OpenAI-shaped) | Source |
|---|---|---|---|
| Tool schema key | `input_schema` | **`parameters`** | ovelia `llm.py:107-115`, cc-server `agent.py:213` |
| Tool spec wrapper | top-level tool object | **`{type:"function", function:{name,description,parameters}}`** | ovelia `_to_wire_tool` `llm.py:107` |
| Model's tool request | `content` block `type:"tool_use"`, `.input` already parsed dict | **`message.tool_calls[].function.arguments` = JSON STRING (caller parses)** | cc-server `llm.py:163-167`, `agent.py:376` |
| Tool result back to model | a `tool_result` content block on a `user` message | **a message with `role:"tool"`, `tool_call_id`, string `content`** | cc-server `agent.py:345-351`, ovelia `_to_wire_message` `llm.py:79-88` |
| "stop because tools" signal | `stop_reason == "tool_use"` | **`finish_reason == "tool_calls"`** | cc-server `agent.py:308`, `api.py:104` |
| Args need re-parse | no (SDK gives parsed dict) | **YES — `json.loads(arguments)`, may fail, may be double-escaped** | cc-server `agent.py:373-381`, `:128` |
| SDK | `anthropic` | **`openai` SDK pointed at OpenRouter base_url, or raw httpx** | cc-server `client.py:12,124`, ovelia `llm.py:135` |
| Prompt caching | Anthropic `cache_control` blocks | **N/A at v1** (OpenRouter passes through provider caching; not a seam concern) | — |

### 3.1 Request shape (what the loop sends)

```jsonc
POST {base_url}/chat/completions          // base_url = https://openrouter.ai/api/v1
{
  "model": "<provider/model-id>",         // e.g. "openai/gpt-4.1-mini", "x-ai/grok-4.1-fast"
  "messages": [ ...wire messages... ],
  "tools": [
    { "type": "function",
      "function": { "name": "edit_shader", "description": "...", "parameters": { /* JSON Schema */ } } }
  ],
  "stream": true,
  "stream_options": { "include_usage": true },   // → usage (incl. cost) on the final chunk
  "max_completion_tokens": <int>
  // optional: "tool_choice", "usage": {"include": true} (raw-httpx equivalent of stream_options)
}
```
(cc-server `client.py:362-391 completion_with_tools` builds exactly this kwargs dict; ovelia
`_stream_impl` `llm.py:204-212` the streaming variant; marginalia `agent.ts:202-203` the raw body.)

### 3.2 Response shape (non-streamed, the model's tool turn)

```jsonc
{ "choices": [ { "finish_reason": "tool_calls",
    "message": { "role": "assistant", "content": null,
      "tool_calls": [
        { "id": "call_abc", "type": "function",
          "function": { "name": "edit_shader", "arguments": "{\"path\":\"x.frag\",\"src\":\"...\"}" } }
      ] } } ],
  "usage": { "prompt_tokens": 1234, "completion_tokens": 56, "cost": 0.0021 } }
```
`arguments` is a **JSON string**, not an object. cc-server pulls these into `ToolCallData(id, name,
arguments=tc.function.arguments)` (`llm.py:163-167`) and the loop `json.loads` them (`agent.py:376`).

### 3.3 Streamed tool calls (the accumulation)

Tool-call deltas arrive fragmented across SSE chunks, keyed by `index`; `id`+`name` appear once, then
`function.arguments` arrives in pieces that must be **concatenated**. The reference accumulation:
ovelia `llm.py:261-303` (`_ToolCallBuilder` per index, emit `LLMToolCallStarted` on first id+name,
`LLMToolCallCompleted` at stream end) and marginalia `agent.ts:160-172` (the `while toolCalls.length <=
tc.index` slot-grow + `slot.function.arguments += ...`). **Lift ovelia's builder loop verbatim** — it's
the one genuinely fiddly bit of the client impl and it's already correct.

### 3.4 The double-escape repair (cc-server's `_unescape_double_escaped_strings`)

`agent.py:128-157`. Some providers (cc-server's comment: *"observed on Grok via OpenRouter"*,
`agent.py:131`) double-escape JSON inside the tool-call arguments — after `json.loads`, a string holds
literal `\n` (two chars) instead of a newline. `_maybe_unescape` (`agent.py:151`) detects the marker
(`\n`,`\t`,`\r`,`\"` present but no real whitespace) and re-interprets through escape repair. **This
matters for ShaderBox** because the copilot's biggest tool argument is *GLSL source text* — full of
newlines — and the maintainer floated Grok as a candidate model. **This belongs in the
argument-parsing helper in the agent loop (or a tiny `copilot/llm/_args.py`), NOT in the client** — the
client returns the raw string per the seam contract; the loop owns parse+repair. Lift `_parse_arguments`
+ `_unescape_double_escaped_strings` + `_maybe_unescape` as a unit (cc-server `agent.py:372-381` +
`:128-157`).

### 3.5 The clean seam: client exposes the call, loop owns the cycle

cc-server's split is the right one: **`LLMClient.stream/completion_with_tools` does ONE call and
returns tool_calls + finish_reason + usage; the `AgentLoop` (`agent.py:237-355`) owns the cycle** —
append assistant-with-tool_calls message, execute each tool, append `role:"tool"` result, re-call,
until `finish_reason != "tool_calls"`. The loop also owns: status events, the max-iterations guard
(`agent.py:273,353`), the input-budget guard (`agent.py:277`), and old-tool-result compression
(`agent.py:171 _compress_old_tool_results`, mirrored in marginalia `agent.ts:23`). **None of that is
the client's business.** The assistant-message builder (`agent.py:357-370 _build_assistant_message`)
and the `role:"tool"` result append (`agent.py:345-351`) are the two wire-shaping helpers the loop owns
— small, lift directly.

---

## 4. Structured output

### 4.1 How the two Python projects do it

- **cc-server** — `completion(prompt, response_clazz, ...)` (`client.py:290-329`): if the model
  `supports_structured_output`, calls `client.beta.chat.completions.parse(response_format=response_clazz, ...)`
  (the `openai` SDK's pydantic-native parse), then `response_clazz.model_validate_json(text)`. If the
  model can't do structured output, it falls back to plain text wrapped into the schema's first string
  field (`_wrap_text_in_schema`, `client.py:550`).
- **ovelia** — `oneshot_structured(prompt, *, response_model, ...)` (`llm.py:427-517`): same
  `beta.chat.completions.parse(response_format=response_model, ...)`, reads `completion.choices[0].message.parsed`,
  maps a `ValidationError` → a typed `LLMParseFailed` (`llm.py:453-466`), maps a content-filter trip →
  `LLMContentFiltered` (`api.py:34-50`) with a model-fallback chain.

Both go through OpenRouter via the `openai` SDK's `response_format=<pydantic model>` (OpenAI
structured-output / `json_schema` response format). **Tool-forcing is NOT used** by either for
structured output — they use `response_format`. (cc-server *does* use `tool_choice` to force a
particular tool, `llm.py:383`, but that's tool-use, not the structured-output path.)

### 4.2 When ShaderBox's copilot wants structured vs free-form-chat-with-tools

- **Free-form chat-with-tools (`stream`)** is the default and covers ~all of v1: the user chats, the
  model writes/edits GLSL via tools, streams prose back. This is the workhorse.
- **Structured (`complete_structured`)** is for an internal, non-conversational step where the loop
  needs a *machine-readable* answer it will branch on — e.g. a **planning/triage step** ("classify this
  request: explain | edit | new-shader") or the **"action-required" confirmation message** the
  maintainer mentioned (a typed `{action, summary, risky: bool}` the UI renders as a confirm gate).
  These want a pydantic model back, not prose. **But do not over-design it:** add `complete_structured`
  to the Protocol as a thin seam (ovelia's `oneshot_structured` signature, made sync) and only wire it
  when a v1 capability actually needs it. If v1 ships without a planning step, the method is an
  unimplemented seam — that's fine, it documents the intent without cost.

### 4.3 The seam

`complete_structured(prompt, *, schema: type[T], max_tokens) -> T` (T bound to `pydantic.BaseModel`).
Impl: `openai` SDK `beta.chat.completions.parse(response_format=schema)` against OpenRouter, return
`.parsed`, raise a typed parse-failure on `ValidationError` (ovelia `llm.py:446-469`). **Verify-at-impl
(§6):** not every OpenRouter model supports `response_format: json_schema` — cc-server gates on
`supports_structured_output` (`client.py:299`). ShaderBox's chosen cheap model must be checked; if it
can't, fall back to tool-forcing or a JSON-in-prose parse. Flag, don't pre-solve.

---

## 5. The module skeleton (with import directions)

The maintainer asked for **clean encapsulated modules with robust seams, NOT capabilities/tool
catalogs.** Here is the LLM-layer-and-around module boundary. (The tool *catalog* and the
threading/bridge are `02`/`01`'s angles; this fixes only how the LLM layer slots in.)

```
copilot/
├── llm/
│   ├── api.py          # THE SEAM (leaf). The frozen dataclasses (§1) + the LLMClient Protocol (§0.1)
│   │                   #   + typed errors (LLMUpstreamError / LLMRateLimited / LLMParseFailed /
│   │                   #   LLMContentFiltered — ovelia api.py:14-50). Imports: stdlib + pydantic only.
│   │                   #   Imports NOTHING from copilot. Everyone imports FROM here.
│   ├── openrouter.py   # THE IMPL. OpenRouterLLMClient(LLMClient). Owns the `openai`/httpx client,
│   │                   #   base_url, api_key, the IPv4-pin decision (§6), retry/backoff, the
│   │                   #   wire-translation (_to_wire_message/_to_wire_tool — ovelia llm.py:70-115),
│   │                   #   the stream-event accumulation (§3.3), usage extraction (§2.1).
│   │                   #   Imports: copilot.llm.api + openai/httpx + settings (for key+model).
│   └── _args.py        # (optional) parse_tool_arguments + _unescape_double_escaped_strings (§3.4).
│                       #   Pure function, no deps. Could live in agent.py instead — small.
├── prompt.py           # PROMPT ASSEMBLY. system prompt text + (history, shader context) → list[LLMMessage].
│                       #   Pure data transform. Imports: copilot.llm.api (for LLMMessage). NOTHING else
│                       #   from copilot — it does NOT know about tools or the client. (ovelia's
│                       #   copilot/prompt.py is the model — see 00_grounding.)
├── tools/              # TOOL REGISTRY (02's angle). Exposes get_tool_specs() -> list[LLMToolSpec] and
│                       #   execute(name, args) -> str. Imports: copilot.llm.api (for LLMToolSpec) +
│                       #   the CopilotCapabilities seam (02). Does NOT import the client or the loop.
└── agent.py            # THE LOOP (the orchestrator). Owns: TurnUsage rollup (§2.3), the tool-use
                        #   cycle (§3.5), arg parse+repair, status/text/usage events to the UI,
                        #   max-iter + budget guards, old-result compression.
                        #   Imports: copilot.llm.api, copilot.llm.openrouter (or gets the client
                        #   injected), copilot.prompt, copilot.tools. Sits at the TOP of the DAG.
```

**Import DAG (strict, acyclic, leaf-first):**

```
            llm/api.py   ← (leaf; imported by all)
           /    |     \
 openrouter  prompt   tools
      \         |      /
        \       |     /
            agent.py        ← (root; imports all four)
```

- **No sideways edges:** `prompt` ⊥ `tools` ⊥ `openrouter` (none imports another). `agent.py` is the
  only module that knows all of them — it's the composition root for the LLM layer.
- **Cycle-free by construction** → satisfies ShaderBox's "no `if TYPE_CHECKING`, a circular import is a
  design bug" rule (CLAUDE.md). The seam types in `api.py` are the shared vocabulary that lets
  `prompt`/`tools`/`openrouter` all depend on a common leaf without depending on each other.
- **Testability seam:** because `agent.py` depends on the **`LLMClient` Protocol** (not the concrete
  `OpenRouterLLMClient`), a headless test injects a fake client that yields scripted stream events —
  exactly cc-server's `LLMService(client=...)` and ovelia's `OpenRouterLLMService(client=...)`
  constructor injection (ovelia `llm.py:129`). This is how the loop gets tested without network (and
  ShaderBox can't screenshot the GLFW app — headless testability is load-bearing per
  `no-screenshot-driven-dev`).
- **Where settings/key enter:** `openrouter.py`'s constructor reads `api_key` + `model` from the
  existing `IntegrationsStore` (`exporters/integrations.py:51` — a pydantic `BaseModel` persisted to
  `integrations.json` at `app_data_dir()`, already holds the Telegram token + YouTube creds; add an
  `openrouter_key: str` + `copilot_model: str` field, same cleartext posture, same existing deferral).
  `agent.py`/`prompt.py`/`tools/` never see the key — it's encapsulated in the impl.

---

## 6. OpenRouter verify-at-impl checklist

Confirm each against OpenRouter's live docs/API at impl time (do NOT trust this report's recall):

1. **base_url** = `https://openrouter.ai/api/v1` (marginalia `OPENROUTER_URL` constant; cc-server reads
   it from `CONFIG.llm.base_url`). The `openai` SDK works against it as a drop-in
   (`OpenAI(base_url=..., api_key=...)` — cc-server `client.py:124`, ovelia `llm.py:135`).
2. **API key** — env `OPENROUTER_API_KEY` is OpenRouter's convention, but ShaderBox should store it in
   `IntegrationsStore`/`integrations.json` (user enters it in settings — same posture as the TG token,
   §5). Confirm the header is `Authorization: Bearer <key>` (marginalia `_apiHeaders` `agent.ts:78`) —
   the `openai` SDK handles this.
3. **usage + cost on the response** — confirm passing `usage:{include:true}` (raw) /
   `stream_options:{include_usage:true}` (SDK) returns `usage.cost` (USD) plus
   `prompt_tokens`/`completion_tokens`. cc-server `client.py:55`, ovelia `llm.py:124` both read
   `usage.cost`. Handle `cost == None` gracefully (ovelia logs `llm_cost_missing` `llm.py:313`).
   Confirm whether the **streamed** usage chunk carries `cost` for the chosen model (provider-dependent).
4. **streaming SSE shape** — `data: {json}\n` lines, `data: [DONE]` terminator, `chunk.choices[0].delta`
   carries `content` and/or `tool_calls[]` (fragmented, index-keyed, args concatenated). marginalia
   `agent.ts:143-177` parses it raw; the `openai` SDK parses it for you (ovelia `llm.py:220-303`). An
   **error mid-stream arrives as `chunk.error`** (marginalia `agent.ts:149`) — handle it.
5. **model id is a string** — yes, OpenRouter routes by `provider/model` string id. Confirm the
   maintainer's candidates exist: `openai/gpt-…` (codex-family), `x-ai/grok-…`, `google/gemini-…`
   (ovelia's `_ROLE_MODELS` `llm.py:49-53` shows `openai/gpt-4.1-mini` + `x-ai/grok-4.1-fast` live).
   Model choice is deferred; the impl just takes the string from settings.
6. **`response_format: json_schema` support** is per-model (§4.3) — confirm the chosen cheap model
   supports it before relying on `complete_structured`; gate or fall back if not (cc-server
   `supports_structured_output` gate, `client.py:299`).
7. **tool-use support** is also per-model — confirm the chosen model supports function calling
   (cc-server gates on `supports_tool_use`, `llm.py:151`). A model that can't call tools breaks the
   whole copilot; pick one that can.
8. **`max_completion_tokens` vs `max_tokens`** — cc-server/ovelia use `max_completion_tokens`
   (`client.py:265`, ovelia `llm.py:210`). Confirm OpenRouter accepts it (it proxies OpenAI's newer
   field); `max_tokens` is the older name marginalia omits entirely.
9. **`extra_body={"reasoning": {"effort": "minimal"}}`** — cc-server (`client.py:33`) and ovelia
   (`llm.py:55`) both pass this to suppress reasoning tokens on reasoning-capable models (cost +
   latency). Confirm it's a no-op / accepted for the chosen model; harmless if ignored.

### 6.1 The IPv4 / dead-IPv6 egress concern — do NOT overfit to the dev box

The maintainer's box has a **dead IPv6 route** (`conventions.md` lines 184-193 — the Telegram
`_ipv4_request` story: AAAA resolves but the route is dead, so a dual-stack httpx client dials v6 and
the TLS handshake hangs/`ConnectError`). ovelia's OpenRouter client **pins egress to IPv4**
(`llm.py:143-148`: `http_client=httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(local_address="0.0.0.0"))`,
with the explicit comment that it forces `AF_INET`). **The trap:** ShaderBox ships to *users*, most of
whom have working IPv6 — hard-pinning v4 for everyone is overfitting to the maintainer's broken box and
could *break* a v6-only user. **Recommendation:** the `openai`-SDK / httpx client in `openrouter.py`
should default to **normal dual-stack** (works for users), and expose the IPv4 pin as an **opt-in**
(a settings flag, or auto-detect: if the first connect times out on a dead-v6 symptom, retry v4-pinned).
The mechanism (inject a custom httpx client into the SDK, ovelia `llm.py:145`) is identical; only the
*default* differs — users get dual-stack, the maintainer flips the flag. Flag this clearly in the spec;
it's the one place the reference impl (ovelia, a server on a controlled box) made a box-specific choice
ShaderBox must NOT inherit blindly.

---

## 7. Adversarial — don't cargo-cult the server design

cc-server's LLM layer is engineered for a **multi-tenant SaaS with per-account billing, plan limits,
model A/B toggles, and a fallback provider**. ShaderBox is a **single-user desktop app where the user
pays their own OpenRouter bill.** The genuinely-reusable core is small; most of cc-server's surface is
server scaffolding. Ruthless partition:

**DO NOT PORT (server cruft):**

- **`LLMService` wrapper + the whole `core/billing/usage/` stack** — `record_llm_usage`, `CounterEngine`
  (`counter.py`), day/month/total buckets, microdollar integer storage, `LimitChecker`,
  `count_toward_limit`. This is metering for billing a customer. ShaderBox has no customer.
- **`LLMTask` / `LLMOperation` (13 ops) / `billable` flag / `_OPERATION_ROLES` map** (`api.py:24-60`,
  `llm.py:42-56`) — "which billable server operation, mapped to which account-toggled model." ShaderBox
  has one operation and one user-chosen model. Every method param `task`/`account_id` vanishes.
- **`_resolve_model` / `IFeatureToggleService` / `get_models_config`** (`llm.py:71-80`) — per-account
  model A/B toggles. ShaderBox reads one model string from settings.
- **Fallback-provider + GigaChat multi-client routing** (`client.py:128-158`, `_get_client`,
  `_get_fallback`, `_fallback_model_map`, `gigachat_client`) — server resilience across paid providers.
  ShaderBox has one provider; a single retry/backoff is plenty (keep the *idea* of retry, drop the
  multi-client routing). (ovelia keeps a *content-filter* fallback chain `llm.py:347-409` — even that
  is more than a desktop tool needs at v1; a single model is fine.)
- **Embeddings** (`embed_texts`, batching, the npz fixture loader `client.py:93-102`) — not a copilot
  concern at all.
- **`tokens.py` entirely** — tiktoken encoding resolution + the pricing/budget pre-estimation. ShaderBox
  reads token *counts + cost* off the response (§2.2). A budget cap is one int compare; no local
  tokenizer needed. (cc-server needs it to *plan* context windows before calling — server-scale concern.)
- **`get_model_capabilities` / `count_tokens` public methods** — both exist to plan against the
  tokenizer. Drop.
- **`ChatCompletionMessageParam` (the SDK type) in the public `api.py`** (cc-server `api.py:13`) — a
  seam leak. Use the neutral `LLMMessage` (ovelia's choice). The SDK type stays *inside* `openrouter.py`.

**GENUINELY REUSABLE (the core):**

- The **typed seam** (the frozen dataclasses + the small Protocol) — but ovelia's neutral version, not
  cc-server's SDK-leaking one.
- **`_extract_usage_from_response`** logic (`client.py:45`) — read `cost`/`prompt_tokens`/
  `completion_tokens` off the response. The single most reusable function.
- **`AgentRunStats`-style in-memory rollup** (`agent.py:43-66`) minus the DB persistence — exactly the
  desktop per-turn/session cost display.
- **The tool-use loop mechanics** (`agent.py:237-400`): the cycle, `_build_assistant_message`, the
  `role:"tool"` result append, the `finish_reason=="tool_calls"` branch, max-iter guard, old-result
  compression (`_compress_old_tool_results`).
- **`_unescape_double_escaped_strings` + `_parse_arguments`** (`agent.py:128-157,372-381`) — the
  Grok-via-OpenRouter double-escape repair. Directly relevant (GLSL source args).
- **The streaming tool-call accumulation** (ovelia `llm.py:261-303`) and the **IPv4-pin mechanism**
  (ovelia `llm.py:143-148`) — but default dual-stack (§6.1), not hard-pinned.
- **Constructor injection of the client** (ovelia `llm.py:129`) — for headless loop tests.

**The one-sentence test:** if a piece exists *because there are many paying accounts and many server
operations*, it's cruft; if it exists *because talking to an OpenAI-shaped LLM over HTTP is fiddly*
(usage extraction, streamed tool-call assembly, arg-repair, retry), it's the reusable core.

---

## 8. Open questions (for the maintainer)

1. **Sync or async client?** This report assumes **sync** (the copilot loop runs on the existing worker
   thread, like the exporters — `01_threading_architecture.md`). ovelia is async because it's an ASGI
   server. Confirm the loop is worker-thread-sync; if so, drop the `async`/`AsyncIterator` from the
   ovelia seam. (Strong recommendation: sync. The `openai` SDK has a sync client.)
2. **`openai` SDK or raw httpx against OpenRouter?** cc-server/ovelia use the `openai` SDK (it owns
   stream parsing, tool-call delta accumulation, `beta.chat.completions.parse` for structured output,
   typed errors, retry). marginalia hand-rolls raw httpx+SSE (because it's a browser/Svelte app with no
   SDK option). **Recommendation: the `openai` SDK** — one `uv add openai`, and it cleanly accepts the
   custom IPv4-pinnable httpx client (ovelia proves this). Raw httpx means reimplementing §3.3
   accumulation + §4 structured parse + retry. Confirm.
3. **Is `complete_structured` in v1 scope?** Only if a planning/triage step or the typed
   "action-required" message ships in v1. If v1 is pure chat-with-tools, leave it an unimplemented seam
   (§4.2). Maintainer's call on whether the action-required-message is v1.
4. **Single retry/backoff vs none at v1?** A desktop tool can arguably surface a transient error to the
   user and let them retry by hand. cc-server's 5-retry + rate-limit-cooldown is server-grade. Suggest
   a **light** retry (2-3, exponential, honor `Retry-After` on 429 — cc-server `client.py:504-521`),
   no fallback-provider. Confirm appetite.
5. **Where does `SessionUsage` live** — on the chat-session state object, the `agent.py`, or a tiny
   `copilot/usage.py`? Depends on `05_chat_ui_ux.md`'s session-state design. Non-blocking; the *shape*
   (`TurnUsage`/`SessionUsage` fed by `LLMUsage`) is fixed here.

---

## 9. Provenance (file:line index)

- **cc-server** `core/ai/llm/api.py` — `ILLMService` (113-174), `ToolCallData` (90-95), `LLMUsageInfo`
  (72-78), `LLMTask`/`LLMOperation`/`billable` (24-60), SDK-type leak (13,118).
- **cc-server** `core/ai/llm/llm.py` — `LLMService` billing wrapper (59-220), `_OPERATION_ROLES`
  (42-56), `_resolve_model` (71-80), `completion_with_tools` shaping (133-178), `_record_usage` (209).
- **cc-server** `core/ai/llm/client.py` — pure client (105-109), `_extract_usage_from_response` (45-90,
  `usage.cost` at 55), `completion_with_tools` request (362-391), `completion`/structured (290-329),
  `completion_stream` (333-358), multi-client routing (128-158), retry/429 (500-547), structured gate
  (299), `_wrap_text_in_schema` (550).
- **cc-server** `core/ai/llm/models.py` — `ModelMetadata`/`supports_tool_use`/`supports_structured_output`
  (73-98), OpenRouter models JSON loader (107-137).
- **cc-server** `core/ai/llm/tokens.py` — tiktoken pricing/budget machinery (whole file; DROP).
- **cc-server** `core/ai/chat/agent.py` — `AgentRunStats` rollup (43-66), `_unescape_double_escaped_strings`
  (128-157), the loop (237-355), `_build_assistant_message` (357-370), `_parse_arguments` (372-381),
  `role:"tool"` append (345-351), `finish_reason=="tool_calls"` (308), `_compress_old_tool_results`
  (171-234), budget guard (277), max-iter (273,353).
- **cc-server** `core/billing/usage/counter.py` — `record_llm_usage` buckets + microdollars (195-216; DROP).
- **ovelia** `llm/api.py` — neutral seam: `LLMMessage` (73-80), `LLMToolSpec`/`parameters` (83-87),
  `LLMToolCall`/raw args (66-71), `LLMUsage` (90-94), stream events (97-123), `ILLMService` Protocol
  (126-160), typed errors (14-50).
- **ovelia** `llm/llm.py` — `_to_wire_message`/`_to_wire_tool` (70-115), `_to_usage`/`usage.cost`
  (118-125), IPv4 pin (143-148), stream-event accumulation `_ToolCallBuilder` (63-67, 261-303), usage
  capture (305-337), `oneshot_structured` (427-517), client injection (129), `_ROLE_MODELS` (49-53),
  `_MINIMAL_REASONING` (55).
- **marginalia** `agent.ts` — raw OpenRouter body (202-203), SSE parse (143-177, `chunk.error` 149),
  tool-call accumulation (160-172), usage off chunk (175), `_apiHeaders` Bearer (78), the loop
  (276-370), tool-result append (350-354), token rollup (284,309-311).
- **ShaderBox** `exporters/integrations.py` — `IntegrationsStore` pydantic+JSON (51-87); `conventions.md`
  184-193 (the dead-IPv6 / `_ipv4_request` story).
