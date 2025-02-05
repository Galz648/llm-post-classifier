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


@app.post("/rank")
async def process_posts(posts: dict[str, Post]):

    # get json -> validate -> construct RequestPayloads(config + posts) -> send requests -> return responses
    # TODO: construct RequestPayloads

    payloads = api_manager.construct_request_payload(posts)
    return await api_manager.send_requests(payloads)
