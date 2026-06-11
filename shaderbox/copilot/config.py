from dataclasses import dataclass


@dataclass
class CopilotConfig:
    # Agent-loop limits. The seven user-facing ones (caps, retry budgets, nudge
    # thresholds) are Settings-tunable: persisted on `CopilotIntegration`
    # (integrations.json) and applied onto the shared COPILOT_CONFIG instance via
    # `apply_user_limits` (startup + Settings edit) — every consumer holds this
    # instance, so a change takes effect immediately. The rest stay constants.
    max_iterations: int = 16
    max_input_tokens: int = 150_000
    # Reasoning models bill hidden thinking into the output budget; creative
    # generations need headroom beyond the visible reply + tool args.
    max_tokens_per_turn: int = 12_000
    # Soft per-edit compile-fix retry budget, distinct from max_iterations (§I2).
    max_edit_retries: int = 3
    # Consecutive broken-compile edits on ONE file before the engine force-restores it to
    # its last clean-compiling state (033; 0 = off). The engine streak is per-node and
    # session-persistent, the nudge counter is per-turn — nudge-before-restore ordering is
    # best-effort within a turn, not guaranteed across turns/interleavings.
    auto_revert_after_failed_edits: int = 6
    # Consecutive applies-but-compiles-with-errors edits before a one-time "rewrite the whole
    # block in one edit" nudge (0 = off). Distinct from max_edit_retries (which counts edits that FAIL to
    # apply); an edit that applies returns ok=True, so it never trips that cap — this catches the
    # apply-but-broken thrash separately. Not a giveup: the model usually recovers.
    max_compile_failures: int = 5
    # Consecutive CLEAN source edits in one turn (applied, zero compile errors) before a
    # one-time "stop and let the user look" nudge (0 = off). The model is render-blind, so nothing
    # else brakes an unbounded aesthetic-tweak spree of individually-clean edits (live
    # case: 16 edits / $0.51 in one turn). Not a giveup; per-turn cumulative, no reset.
    max_clean_edit_streak: int = 6
    # Headroom the history trim withholds for the per-turn working-set scratchpad, which is
    # spliced AFTER the trim runs and is otherwise invisible to it (feature 020·29 D10).
    scratchpad_reserve_tokens: int = 50_000
    # Feature 033 enriched results: probe-render facts after clean mutations
    # (ink/bbox/luma off a tiny offscreen render). Size is the square probe edge.
    render_facts_enabled: bool = True
    render_facts_size: int = 64
    # A list-arg above this count trips a BULK-policy gate (§2.3 / §F4).
    bulk_gate_threshold: int = 5
    # Worker join() timeout at shutdown; a blocking network read may outlive it
    # (then the thread is abandoned — daemon=False, warn-and-leave, like the exporters).
    worker_join_timeout_s: float = 5.0
    # Per-op wait on a worker->main bridge round-trip (UI busy / shutting down).
    bridge_op_timeout_s: float = 5.0
    # Render ops get a longer wait — a video encode freezes the frame loop far past the
    # 5s default (R3/§5). Used via bridge.run_on_main(fn, timeout=…) by the render tools.
    render_op_timeout_s: float = 60.0
    # Publish-await (feature 020·18): the copilot worker polls the exporter's terminal
    # progress through trivial bridge ops; this bounds the total wait (a never-terminal
    # stuck upload) and the per-poll sleep between them.
    publish_await_timeout_s: float = 300.0
    publish_poll_interval_s: float = 0.2
    # Telegram connect-await (feature 020·19): bounds the poll for auth_state to leave LINKING.
    telegram_connect_timeout_s: float = 30.0


COPILOT_CONFIG = CopilotConfig()


def apply_user_limits(
    *,
    max_iterations: int,
    max_input_tokens: int,
    max_tokens_per_turn: int,
    max_edit_retries: int,
    max_compile_failures: int,
    max_clean_edit_streak: int,
    auto_revert_after_failed_edits: int,
) -> None:
    # The Settings -> live-config seam. Values arrive pre-clamped by the Settings UI;
    # a hand-edited integrations.json gets a floor here so a 0 cap can't wedge the loop.
    COPILOT_CONFIG.max_iterations = max(1, max_iterations)
    COPILOT_CONFIG.max_input_tokens = max(10_000, max_input_tokens)
    COPILOT_CONFIG.max_tokens_per_turn = max(1_000, max_tokens_per_turn)
    COPILOT_CONFIG.max_edit_retries = max(1, max_edit_retries)
    COPILOT_CONFIG.max_compile_failures = max(0, max_compile_failures)  # 0 = off
    COPILOT_CONFIG.max_clean_edit_streak = max(0, max_clean_edit_streak)  # 0 = off
    COPILOT_CONFIG.auto_revert_after_failed_edits = max(
        0, auto_revert_after_failed_edits
    )  # 0 = off
