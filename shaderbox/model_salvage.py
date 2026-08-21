"""Load a persisted pydantic model without letting one bad key cost the user everything.

Every on-disk model here is fail-soft: an unreadable or incompatible file degrades to
defaults rather than crashing. Done naively that is a data-loss bug, because the app writes
the loaded state back on quit — so a single retired key or wrong-typed value round-trips as
"reset every setting you had". `IntegrationsStore` learned this the expensive way (a retired
key silently wiped every stored credential), and its rule is the one this module generalises:
**a retired or malformed field must cost the user that setting, never the rest of the file.**

`drop_unknown` prunes keys no field claims, recursing into nested models. `drop_invalid`
then drops the keys whose values their own field rejects. What survives is constructed
normally, so defaults fill the holes and everything else is preserved.
"""

import json
from pathlib import Path
from typing import Any, get_args

from loguru import logger
from pydantic import BaseModel, ValidationError


def drop_unknown(model: type[BaseModel], data: dict[str, Any], path: str) -> None:
    """Prune keys no field claims, recursing into nested models and lists of models.

    A non-dict `data` is left alone rather than raising: callers that parse a file themselves
    can hand us `null` or a list, and a TypeError here escapes into whatever loads the store
    — for the credential store that is ProjectSession.__init__, i.e. the app failing to start.
    """
    if not isinstance(data, dict):
        return
    for key in [k for k in data if k not in model.model_fields]:
        logger.warning(f"Ignoring unknown {path} key: {key}")
        data.pop(key)
    for key, field in model.model_fields.items():
        value = data.get(key)
        # `or (annotation,)` covers a bare nested model; get_args covers list[Model],
        # whose elements are their own models.
        for nested in get_args(field.annotation) or (field.annotation,):
            if not (isinstance(nested, type) and issubclass(nested, BaseModel)):
                continue
            if isinstance(value, dict):
                drop_unknown(nested, value, f"{path}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        drop_unknown(nested, item, f"{path}.{key}[{index}]")


def drop_invalid(model: type[BaseModel], data: dict[str, Any], path: str) -> None:
    """Drop keys whose value their own field rejects, so the rest of the file survives.

    Each candidate is validated ALONE against its field, so one malformed value can't
    implicate its siblings. A dropped key falls back to the field's default.

    Descends into nested models FIRST, mirroring `drop_unknown`: a nested block is validated
    as a whole, so without the descent one bad row inside it takes the entire block with it —
    a malformed pack entry would cost the user the Telegram token sitting beside it.
    """
    if not isinstance(data, dict):
        return
    for key, field in model.model_fields.items():
        value = data.get(key)
        for nested in get_args(field.annotation) or (field.annotation,):
            if not (isinstance(nested, type) and issubclass(nested, BaseModel)):
                continue
            if isinstance(value, dict):
                drop_invalid(nested, value, f"{path}.{key}")
            elif isinstance(value, list):
                # A list element that stays invalid after its own salvage is dropped
                # whole — keeping it would fail the list's validation and cost every sibling.
                kept: list[Any] = []
                for index, item in enumerate(value):
                    if not isinstance(item, dict):
                        kept.append(item)
                        continue
                    drop_invalid(nested, item, f"{path}.{key}[{index}]")
                    try:
                        nested(**item)
                    except ValidationError:
                        logger.warning(f"Ignoring invalid {path}.{key}[{index}]")
                        continue
                    kept.append(item)
                data[key] = kept

    for key in list(data):
        field = model.model_fields.get(key)
        if field is None:
            continue
        try:
            model.__pydantic_validator__.validate_assignment(
                model.model_construct(), key, data[key]
            )
        except ValidationError as e:
            logger.warning(
                f"Ignoring invalid {path}.{key} ({e.error_count()} error(s))"
            )
            data.pop(key)


def load_model[Model: BaseModel](
    model: type[Model], file_path: str | Path, path: str
) -> Model:
    """Read `file_path` into `model`, salvaging every key that is still valid."""
    try:
        with Path(file_path).open("r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Unreadable {path} ({e}); falling back to defaults")
        return model()
    if not isinstance(data, dict):
        logger.warning(f"Malformed {path} (not an object); falling back to defaults")
        return model()

    drop_unknown(model, data, path)
    drop_invalid(model, data, path)
    try:
        return model(**data)
    except ValidationError as e:
        logger.warning(f"Incompatible {path} ({e}); falling back to defaults")
        return model()
