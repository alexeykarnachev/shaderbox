import base64
from io import BytesIO

import httpx

from shaderbox.core import Image


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
