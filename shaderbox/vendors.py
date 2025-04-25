import base64
from io import BytesIO

import httpx
from PIL import Image


def _image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _base64_to_image(base64_str: str) -> Image.Image:
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))


def _post_image_to_endpoint(url: str, base64_image: str) -> str:
    with httpx.Client() as client:
        response = client.post(
            url,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
            },
            json={"image": base64_image},
            timeout=30.0,
        )
        response.raise_for_status()
    return str(response.json()["image"])


def get_modelbox_depthmap(image: Image.Image) -> Image.Image:
    url = "http://localhost:8228/infer_depth_pro"
    base64_image = _image_to_base64(image)
    depthmap_base64 = _post_image_to_endpoint(url, base64_image)
    return _base64_to_image(depthmap_base64)


def get_modelbox_bg_removal(image: Image.Image) -> Image.Image:
    url = "http://localhost:8228/infer_bg_removal"
    base64_image = _image_to_base64(image)
    no_bg_base64 = _post_image_to_endpoint(url, base64_image)
    return _base64_to_image(no_bg_base64)
