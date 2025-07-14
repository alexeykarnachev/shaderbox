from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urljoin

import httpx

from shaderbox.core import Image, Video


def _post_file(modelbox_url: str, file_path: Path | str) -> str:
    url = urljoin(modelbox_url, "file")
    file_path = Path(file_path)

    with file_path.open("rb") as file, httpx.Client() as client:
        response = client.post(
            url, files={"file": (file_path.name, file)}, timeout=10.0
        )
        response.raise_for_status()
    return str(response.json())


def _get_file(modelbox_url: str, file_name: str, output_path: Path) -> None:
    url = urljoin(modelbox_url, "file")

    with httpx.Client() as client:
        response = client.get(f"{url}/{file_name}", timeout=10.0)
        response.raise_for_status()

    with output_path.open("wb") as file:
        file.write(response.content)


def fetch_modelbox_info(modelbox_url: str) -> dict[str, Any]:
    url = urljoin(modelbox_url, "info")

    with httpx.Client() as client:
        response = client.get(url, timeout=1.0)
        response.raise_for_status()
    return response.json()  # type: ignore


T = TypeVar("T", Image, Video)


def infer_media_model(
    modelbox_url: str,
    media: T,
    model_name: str,
    output_dir: Path,
) -> T:
    input_file_name = _post_file(modelbox_url, media.details.file_details.path)

    url = urljoin(modelbox_url, "media_model")
    with httpx.Client() as client:
        response = client.post(
            url,
            json={
                "file_name": input_file_name,
                "model_name": model_name,
            },
            timeout=600.0,
        )
        response.raise_for_status()

    output_file_name = response.json()["file_name"]
    output_file_path = Path(output_dir / output_file_name)
    _get_file(modelbox_url, output_file_name, output_file_path)

    output_media = media.__class__(output_file_path)
    return output_media
