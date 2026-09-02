"""The (pass, uniform) key the script engine and the persisted stop set share (069 D3).

It lives in the engine's own package rather than beside `UIDocumentState`, because the engine takes
a `frozenset[StoppedKey]` per tick and `ui_models` imports the concrete `Document` — the engine may
not. The persisted field imports it from here, so the on-disk shape and the engine's key are one
type rather than two that must be kept in step.
"""

from pydantic import BaseModel


class StoppedKey(BaseModel, frozen=True):
    # One (pass, uniform) the user has STOPPED. A pair, not a name: the same uniform name on two
    # passes is two independently stoppable rows (069 D3). `frozen=True` in the CLASS ARGS, not in a
    # `model_config`, because the engine holds these in a set and only this form makes the generated
    # `__hash__` visible to the type checker. `pass_name`, not `pass`, on disk as in code — `pass` is
    # a Python keyword and cannot be an attribute.
    pass_name: str
    name: str
