class CopilotError(Exception):
    """Base for all copilot-layer errors."""


class CopilotCancelled(CopilotError):
    """A main-thread op or turn was cancelled (app shutting down / Stop pressed)."""


class CopilotToolError(CopilotError):
    """A tool DELIBERATELY rejected the call (bad args, a failed main-thread op, a timeout,
    a guard). Its message is authored FOR the model — the registry surfaces it verbatim."""


class CopilotConfigError(CopilotError):
    """A turn can't start because the copilot isn't configured (no key / model)."""
