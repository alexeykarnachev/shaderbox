import base64
from io import BytesIO

import httpx

from shaderbox.core import Image
from shaderbox.settings import settings


def get_image_to_image_model_names() -> list[str]:
    with httpx.Client() as client:
        response = client.get(
            settings.modelbox_image_to_image_url,
            timeout=5.0,
        )
        response.raise_for_status()
    return response.json()  # type: ignore


def infer_image_to_image(image: Image, model_name: str) -> Image:
    buffer = BytesIO()
    image._image.save(buffer, format="PNG")
    buffer.seek(0)

    inp_image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    with httpx.Client() as client:
        response = client.post(
            settings.modelbox_image_to_image_url,
            json={"image_base64": inp_image_base64, "model_name": model_name},
            timeout=30.0,
        )
        response.raise_for_status()

    out_image_base64 = str(response.json()["image_base64"])
    out_image = Image(BytesIO(base64.b64decode(out_image_base64)))
    return out_image


# def get_modelbox_depthmap_video(video: Video) -> Video:
#     url = "http://0.0.0.0:8228/infer_depth_pro_video"
#     file_path = video.details.file_details.path
#     with Path(file_path).open("rb") as video_file:
#         video_data = video_file.read()
#
#     with httpx.Client() as client:
#         response = client.post(
#             url,
#             files={
#                 "video": (f"video{Path(file_path).suffix}", video_data, "video/mp4")
#             },
#             timeout=300.0,
#         )
#         response.raise_for_status()
#
#     response_data = response.json()
#     base64_video = response_data["video"]
#
#     temp_dir = Path(tempfile.gettempdir())
#     temp_file_path = temp_dir / f"depthmap_video_{int(time.time() * 1000)}.mp4"
#
#     video_data = base64.b64decode(base64_video)
#     with temp_file_path.open("wb") as f:
#         f.write(video_data)
#
#     return Video(temp_file_path)
#
#
# def get_modelbox_bg_removal_video(video: Video) -> Video:
#     url = "http://0.0.0.0:8228/infer_bg_removal_video"
#     file_path = video.details.file_details.path
#     with Path(file_path).open("rb") as video_file:
#         video_data = video_file.read()
#
#     with httpx.Client() as client:
#         response = client.post(
#             url,
#             files={
#                 "video": (f"video{Path(file_path).suffix}", video_data, "video/mp4")
#             },
#             timeout=300.0,
#         )
#         response.raise_for_status()
#
#     response_data = response.json()
#     base64_video = response_data["video"]
#
#     temp_dir = Path(tempfile.gettempdir())
#     temp_file_path = temp_dir / f"bg_removal_video_{int(time.time() * 1000)}.mp4"
#
#     video_data = base64.b64decode(base64_video)
#     with temp_file_path.open("wb") as f:
#         f.write(video_data)
#
#     return Video(temp_file_path)
