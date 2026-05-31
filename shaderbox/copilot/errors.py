class CopilotError(Exception):
    """Base for all copilot-layer errors."""


class CopilotCancelled(CopilotError):
    """A main-thread op or turn was cancelled (app shutting down / Stop pressed)."""


class CopilotToolError(CopilotError):
    """A tool could not run (bad args, a failed main-thread op, a timeout)."""


class CopilotConfigError(CopilotError):
    """A turn can't start because the copilot isn't configured (no key / model)."""
