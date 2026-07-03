from enum import Enum, auto
from threading import Thread

from shaderbox.copilot.llm.openrouter import fetch_model_image_support


class VisionProbeStatus(Enum):
    IDLE = auto()
    CHECKING = auto()
    READY = auto()
    ERROR = auto()


class VisionVerdict(Enum):
    CHECKING = auto()  # a fetch is in flight
    SUPPORTED = auto()  # the model is listed and accepts image input
    UNSUPPORTED = auto()  # the model is listed but is text-only
    UNKNOWN = auto()  # the fetch succeeded but the id isn't in the catalogue (typo)
    UNVERIFIED = auto()  # no key yet, or a transient fetch failure (offline / 5xx)


class VisionModelProbe:
    """Async capability check for the Settings vision-model field (feature 053).

    A single daemon thread fetches OpenRouter's per-model image-input support; the draw thread reads
    the result each frame for a live badge, so a bad/typo/text-only model is caught at set-time instead
    of silently failing on the first probe. NO lock is needed: the single-threaded imgui draw solely
    owns the IDLE->CHECKING transition (so at most one fetch runs at a time), and the worker assigns
    `status` LAST (atomic reference store under the GIL), so a reader that observes READY also observes
    the fully-populated `support`. Daemon is acceptable here (unlike the exporters' joined workers): the
    op is an idempotent read-only GET writing only in-memory state, so an interpreter-shutdown kill
    leaks nothing.
    """

    def __init__(self) -> None:
        self.status: VisionProbeStatus = VisionProbeStatus.IDLE
        self.support: dict[str, bool] = {}
        self.checked_key: str = ""

    def ensure_checked(self, api_key: str) -> None:
        # Called from the Settings draw each frame. Kicks ONE fetch when stale (never checked, or the
        # key changed) and not already in flight. The `== CHECKING` short-circuit is what stops a
        # per-frame re-kick storm while a fetch for a just-changed key is running: exactly one more
        # kick fires after it completes if the key changed again mid-fetch, not one per frame.
        if not api_key:
            return
        if self.status == VisionProbeStatus.CHECKING:
            return
        if self.status != VisionProbeStatus.IDLE and self.checked_key == api_key:
            return
        self.status = (
            VisionProbeStatus.CHECKING
        )  # draw-thread-only transition -> "one thread max"
        Thread(target=self._fetch, args=(api_key,), daemon=True).start()

    def _fetch(self, api_key: str) -> None:
        support = fetch_model_image_support(api_key)
        self.checked_key = (
            api_key  # set BEFORE status so a READY reader sees the matching key
        )
        if support is None:
            self.support = {}
            self.status = VisionProbeStatus.ERROR
        else:
            self.support = support
            self.status = VisionProbeStatus.READY  # assigned LAST (see class docstring)

    def verdict(self, model: str) -> VisionVerdict:
        if self.status == VisionProbeStatus.CHECKING:
            return VisionVerdict.CHECKING
        if self.status in (VisionProbeStatus.IDLE, VisionProbeStatus.ERROR):
            return VisionVerdict.UNVERIFIED
        has_image = self.support.get(model)
        if has_image is None:
            return VisionVerdict.UNKNOWN
        return VisionVerdict.SUPPORTED if has_image else VisionVerdict.UNSUPPORTED
