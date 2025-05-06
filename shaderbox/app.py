from litestar import Litestar, post
from litestar.datastructures import UploadFile
from litestar.params import Body
from loguru import logger
from pydantic import BaseModel


class ApplyNodeResult(BaseModel):
    image: str


class ApplyNodeRequest(BaseModel):
    image: UploadFile

    class Config:
        arbitrary_types_allowed = True


_body_multipart = Body(media_type="multipart/form-data")


@post("/infer_depth_pro", media_type="application/json")
async def apply_node(
    data: ApplyNodeRequest = _body_multipart,
) -> ApplyNodeResult:
    logger.debug("Received image file")
    try:
        return ApplyNodeResult(image="")
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        raise


app = Litestar(
    route_handlers=[apply_node],
    request_max_body_size=100 * 1024 * 1024,
)
