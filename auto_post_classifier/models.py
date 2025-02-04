from pydantic import BaseModel
from typing import Optional


# TODO: change names to image_url, text, content_url, platform
class Post(BaseModel):
    image: Optional[str] = None
    text: Optional[str] = None
    content_url: str
