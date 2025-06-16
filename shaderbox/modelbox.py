from pathlib import Path
from typing import Any, TypeVar

import httpx

from shaderbox.core import Image, Video
from shaderbox.settings import settings


def _post_file(file_path: Path | str) -> str:
    file_path = Path(file_path)
    with file_path.open("rb") as file, httpx.Client() as client:
        response = client.post(
            settings.modelbox_file_url,
            files={"file": (file_path.name, file)},
            timeout=10.0,
        )
        response.raise_for_status()
    return str(response.json())


def _get_file(file_name: str, output_path: Path) -> None:
    with httpx.Client() as client:
        response = client.get(
            f"{settings.modelbox_file_url}/{file_name}",
            timeout=10.0,
        )
        response.raise_for_status()

    with output_path.open("wb") as file:
        file.write(response.content)


def fetch_modelbox_info() -> dict[str, Any]:
    with httpx.Client() as client:
        response = client.get(
            settings.modelbox_info_url,
            timeout=1.0,
        )
        response.raise_for_status()
    return response.json()  # type: ignore


T = TypeVar("T", Image, Video)


def infer_media_model(media: T, model_name: str, output_dir: Path) -> T:
    input_file_name = _post_file(media.details.file_details.path)

    with httpx.Client() as client:
        response = client.post(
            settings.modelbox_media_model_url,
            json={
                "file_name": input_file_name,
                "model_name": model_name,
            },
            timeout=600.0,
        )
        response.raise_for_status()

    output_file_name = response.json()["file_name"]
    output_file_path = Path(output_dir / output_file_name)
    _get_file(output_file_name, output_file_path)

    output_media = media.__class__(output_file_path)
    return output_media
