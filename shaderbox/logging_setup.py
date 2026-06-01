import sys
from dataclasses import dataclass

from loguru import logger

from shaderbox.paths import log_dir

# Central logging setup — called ONCE at startup (ui.py::main, scripts/smoke.py).
# Two sinks: a terse INFO+ console for at-a-glance flow, and a rotated DEBUG+ file
# that gets a strict SUPERSET of the console (everything the console sees, plus the
# lifecycle/diagnostic detail). No module configures sinks itself — call sites only
# do `from loguru import logger; logger.info(...)`. See conventions.md ## Design
# decisions (logging is configured once, never per-module).

_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <7}</level> | "
    "<level>{message}</level>"
)
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | "
    "{name}:{function}:{line} - {message}"
)


@dataclass(frozen=True)
class LoggingConfig:
    # Application-internal constants (not user-tuned, not on UIAppState — mirrors
    # copilot/config.py). console_level keeps the terminal terse; file_level captures
    # the DEBUG lifecycle detail. trace_retention bounds the copilot transcript files
    # (copilot/trace.py), which are large debug ephemera.
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    rotation: str = "5 MB"
    retention: int = 5
    trace_retention: int = 20


LOGGING_CONFIG = LoggingConfig()


def configure_logging() -> None:
    logger.remove()  # drop loguru's default stderr sink (DEBUG, everything)
    logger.add(
        sys.stderr,
        level=LOGGING_CONFIG.console_level,
        format=_CONSOLE_FORMAT,
        # The worker thread (copilot) logs too; enqueue serializes writes off the
        # logging path so a slow sink never blocks it.
        enqueue=True,
        # Same secret-hygiene reason as the file sink: no variable-value dumps in
        # crash tracebacks (could echo the OpenRouter key / Telegram token).
        diagnose=False,
    )
    logger.add(
        log_dir() / "shaderbox_{time}.log",
        level=LOGGING_CONFIG.file_level,
        format=_FILE_FORMAT,
        rotation=LOGGING_CONFIG.rotation,
        retention=LOGGING_CONFIG.retention,
        enqueue=True,
        # No variable-value dumps in tracebacks — they would leak the OpenRouter key
        # / Telegram token into the log file on a crash. The traceback frames remain.
        diagnose=False,
    )
