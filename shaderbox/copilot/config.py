from dataclasses import dataclass


@dataclass(frozen=True)
class CopilotConfig:
    # Agent-loop limits. Not user-tuned — constants, so they avoid the UIAppState
    # migration discipline (they live here, not on app_state).
    max_iterations: int = 12
    max_input_tokens: int = 150_000
    max_tokens_per_turn: int = 8_000
    # Soft per-edit compile-fix retry budget, distinct from max_iterations (§I2).
    max_edit_retries: int = 3
    # Context lines on each side of a changed region in the edit apply-feedback excerpt.
    edit_feedback_context: int = 2
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
