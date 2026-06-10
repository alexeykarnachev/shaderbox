from dataclasses import dataclass


@dataclass(frozen=True)
class CopilotConfig:
    # Agent-loop limits. Not user-tuned — constants, so they avoid the UIAppState
    # migration discipline (they live here, not on app_state).
    max_iterations: int = 16
    max_input_tokens: int = 150_000
    max_tokens_per_turn: int = 8_000
    # Soft per-edit compile-fix retry budget, distinct from max_iterations (§I2).
    max_edit_retries: int = 3
    # Consecutive applies-but-compiles-with-errors edits before a one-time "rewrite the whole
    # block in one edit" nudge. Distinct from max_edit_retries (which counts edits that FAIL to
    # apply); an edit that applies returns ok=True, so it never trips that cap — this catches the
    # apply-but-broken thrash separately. Not a giveup: the model usually recovers.
    max_compile_failures: int = 5
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
