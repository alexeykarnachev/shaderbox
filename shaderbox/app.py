from litestar import Litestar, post
from litestar.datastructures import UploadFile
from litestar.params import Body
from loguru import logger
from pydantic import BaseModel


class ApplyNodeRequest(BaseModel):
    image: UploadFile
    node_name: str

    class Config:
        arbitrary_types_allowed = True


class ApplyNodeResult(BaseModel):
    image: str


_body_multipart = Body(media_type="multipart/form-data")


@post("/apply_node", media_type="application/json")
async def apply_node(
    data: ApplyNodeRequest = _body_multipart,
) -> ApplyNodeResult:
    try:
        return ApplyNodeResult(image="")
    except Exception as e:
        logger.exception(f"Failed to apply node: {e}")
        raise


app = Litestar(
    route_handlers=[apply_node],
    request_max_body_size=100 * 1024 * 1024,
)
