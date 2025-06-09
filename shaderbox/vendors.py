import base64
import tempfile
import time
from io import BytesIO
from pathlib import Path

import httpx

from shaderbox.core import Image, Video


def _post_image_to_endpoint(url: str, image: Image) -> str:
    buffer = BytesIO()
    image._image.save(buffer, format="PNG")
    buffer.seek(0)

    with httpx.Client() as client:
        response = client.post(
            url,
            files={"image": ("image.png", buffer, "image/png")},
            timeout=30.0,
        )
        response.raise_for_status()
    return str(response.json()["image"])


def get_modelbox_depthmap(image: Image) -> Image:
    url = "http://0.0.0.0:8228/infer_depth_pro"
    base64_image = _post_image_to_endpoint(url, image)
    image_data = base64.b64decode(base64_image)
    return Image(BytesIO(image_data))


def get_modelbox_bg_removal(image: Image) -> Image:
    url = "http://0.0.0.0:8228/infer_bg_removal"
    base64_image = _post_image_to_endpoint(url, image)
    image_data = base64.b64decode(base64_image)
    return Image(BytesIO(image_data))


def get_modelbox_depthmap_video(video: Video) -> Video:
    url = "http://0.0.0.0:8228/infer_depth_pro_video"
    file_path = video.details.file_details.path
    with Path(file_path).open("rb") as video_file:
        video_data = video_file.read()

    with httpx.Client() as client:
        response = client.post(
            url,
            files={
                "video": (f"video{Path(file_path).suffix}", video_data, "video/mp4")
            },
            timeout=300.0,
        )
        response.raise_for_status()

    response_data = response.json()
    base64_video = response_data["video"]

    temp_dir = Path(tempfile.gettempdir())
    temp_file_path = temp_dir / f"depthmap_video_{int(time.time() * 1000)}.mp4"

    video_data = base64.b64decode(base64_video)
    with temp_file_path.open("wb") as f:
        f.write(video_data)

    return Video(temp_file_path)


def get_modelbox_bg_removal_video(video: Video) -> Video:
    url = "http://0.0.0.0:8228/infer_bg_removal_video"
    file_path = video.details.file_details.path
    with Path(file_path).open("rb") as video_file:
        video_data = video_file.read()

    with httpx.Client() as client:
        response = client.post(
            url,
            files={
                "video": (f"video{Path(file_path).suffix}", video_data, "video/mp4")
            },
            timeout=300.0,
        )
        response.raise_for_status()

    response_data = response.json()
    base64_video = response_data["video"]

    temp_dir = Path(tempfile.gettempdir())
    temp_file_path = temp_dir / f"bg_removal_video_{int(time.time() * 1000)}.mp4"

    video_data = base64.b64decode(base64_video)
    with temp_file_path.open("wb") as f:
        f.write(video_data)

    return Video(temp_file_path)
