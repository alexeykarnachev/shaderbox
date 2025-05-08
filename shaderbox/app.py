import hashlib
import sys
import time
from pathlib import Path

import moderngl
from litestar import Litestar, post
from litestar.datastructures import UploadFile
from litestar.enums import MediaType
from litestar.params import Body
from litestar.response import File
from loguru import logger
from platformdirs import user_data_dir
from pydantic import BaseModel

from shaderbox.core import Node

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


_body_multipart = Body(media_type="multipart/form-data")


@post("/apply_node", media_type=MediaType.JSON)
async def apply_node(data: ApplyNodeRequest = _body_multipart) -> File:
    try:
        node = _NODES[data.name]

        hash = hashlib.md5(f"{id(data)}{time.time()}".encode()).hexdigest()[:8]
        output_path = _VIDEOS_DIR / f"{data.name}_{hash}.webm"

        node.render_to_video(
            output_path=output_path,
            duration=1.0,
            fps=10,
        )

        logger.debug(f"Video rendered: {output_path}")
        return File(
            path=output_path,
            filename="output.webm",
            media_type="video/webm",
            content_disposition_type="attachment",
        )
    except Exception as e:
        logger.exception(f"Failed to apply node: {e}")
        raise


app = Litestar(route_handlers=[apply_node], request_max_body_size=100 * 1024 * 1024)
