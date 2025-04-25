import base64
from io import BytesIO

import httpx
from PIL import Image


def get_modelbox_depthmap(image: Image.Image) -> Image.Image:
    url = "http://localhost:8228/infer_depth_pro"

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

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

    depthmap_base64 = response.json()["image"]
    depthmap_image_data = base64.b64decode(depthmap_base64)
    depthmap_image = Image.open(BytesIO(depthmap_image_data))

    return depthmap_image


def get_modelbox_bg_removal(image: Image.Image) -> Image.Image:
    url = "http://localhost:8228/infer_bg_removal"

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

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

    no_bg_base64 = response.json()["image"]
    no_bg_image_data = base64.b64decode(no_bg_base64)
    no_bg_image = Image.open(BytesIO(no_bg_image_data))

    return no_bg_image
