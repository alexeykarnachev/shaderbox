import base64
import hashlib
import sys
import time
from io import BytesIO
from pathlib import Path

import moderngl
from litestar import Litestar, post
from litestar.datastructures import UploadFile
from litestar.enums import MediaType
from litestar.params import Body
from loguru import logger
from PIL import Image
from platformdirs import user_data_dir
from pydantic import BaseModel

from shaderbox.core import Node, image_to_texture
from shaderbox.vendors import get_modelbox_depthmap

logger.remove()
logger.add(sys.stderr, level="DEBUG", backtrace=True, diagnose=True)

_APP_DIR = Path(user_data_dir("shaderbox"))
_NODES_DIR = _APP_DIR / "nodes"
_VIDEOS_DIR = Path("/app/videos")

_VIDEOS_DIR.mkdir(exist_ok=True)

gl_params = {"standalone": True, "backend": "egl", "major": 4, "minor": 6}
gl = moderngl.create_context(**gl_params)

_NODES = {
    "green_scan": Node.load_from_dir(_NODES_DIR / "2aecfbe7", gl)[0],
}


class ApplyNodeRequest(BaseModel):
    name: str
    image: UploadFile

    class Config:
        arbitrary_types_allowed = True


class ApplyNodeResult(BaseModel):
    video: str


_body_multipart = Body(media_type="multipart/form-data")


@post("/apply_node", media_type=MediaType.JSON)
async def apply_node(data: ApplyNodeRequest = _body_multipart) -> ApplyNodeResult:
    try:
        node = _NODES[data.name]
        hash = hashlib.md5(f"{id(data)}{time.time()}".encode()).hexdigest()[:8]
        output_path = _VIDEOS_DIR / f"{data.name}_{hash}.webm"

        image_bytes = await data.image.read()
        image = Image.open(BytesIO(image_bytes))
        depthmap = get_modelbox_depthmap(image)

        u_image = image_to_texture(image)
        u_depthmap = image_to_texture(depthmap)

        node.set_uniform_value("u_photo", u_image)
        node.set_uniform_value("u_depth", u_depthmap)
        node.reset_output_texture_size(u_image.size)

        node.render_to_video(
            output_path=output_path,
            duration=2.0,
            fps=30,
        )
        logger.debug(f"Video rendered: {output_path}")
        with output_path.open("rb") as f:
            video_data = base64.b64encode(f.read()).decode("utf-8")
        return ApplyNodeResult(video=video_data)
    except Exception as e:
        logger.exception(f"Failed to apply node: {e}")
        raise


app = Litestar(route_handlers=[apply_node], request_max_body_size=100 * 1024 * 1024)
