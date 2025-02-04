import os

from aiohttp import PAYLOAD_REGISTRY
import dotenv
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

import auto_post_classifier.api as api
from auto_post_classifier.models import Post
from auto_post_classifier.gpt_handler import RequestPayload

dotenv.load_dotenv()
api_manager = api.ApiManager()

if not os.path.exists("logs"):
    os.mkdir("logs")

# TODO: connect logger to cloud storage
logger.add(os.path.join("logs", "file_{time}.log"))

app = FastAPI(
    title="auto post classifier",
    description="Description of my app.",
    version="1.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    redoc_url="/redoc",
)


# TODO: move to a class responsible for building requests
def construct_single_payload(post: Post) -> RequestPayload:
    if post.text:
        api_manager.request_builder.add_text_support(post.text)
    if post.image:
        api_manager.request_builder.add_image_support(post.image)

    return RequestPayload(request_config=api_manager.request_builder.build())


# TODO: move to a class responsible for building requests
def construct_request_payload(posts: dict[str, Post]) -> list[RequestPayload]:
    return [construct_single_payload(post) for post in posts.values()]


@app.post("/rank")
async def process_posts(posts: dict[str, Post]):

    # get json -> validate -> construct RequestPayloads(config + posts) -> send requests -> return responses
    # TODO: construct RequestPayloads

    payloads = construct_request_payload(posts)
    return await api_manager.send_requests(payloads)
