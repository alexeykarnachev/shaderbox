from dataclasses import dataclass


@dataclass(frozen=True)
class CopilotConfig:
    # Agent-loop limits. Not user-tuned — constants, so they avoid the UIAppState
    # migration discipline (they live here, not on app_state).
    max_iterations: int = 12
    max_input_tokens: int = 150_000
    max_tokens_per_turn: int = 8_000
    # Worker join() timeout at shutdown; a blocking network read may outlive it
    # (then the thread is abandoned — daemon=False, warn-and-leave, like the exporters).
    worker_join_timeout_s: float = 5.0
    # Per-op wait on a worker->main bridge round-trip (UI busy / shutting down).
    bridge_op_timeout_s: float = 5.0


COPILOT_CONFIG = CopilotConfig()
