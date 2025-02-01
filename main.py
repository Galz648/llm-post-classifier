import os

import dotenv
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

import auto_post_classifier.api as api

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
async def process_posts(posts: dict[str, api.Post]):
    return await api_manager.process_posts(posts)
