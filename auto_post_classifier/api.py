"""
API Manager Module

This module provides the main interface for processing posts through the GPT classifier.
It handles request validation, GPT interactions, and response persistence.

Classes:
    PreRequestValidator: Validates posts before processing
    ApiManager: Main manager for API operations
"""

import json
import os
from typing import Dict, List

from loguru import logger
from pydantic import ValidationError

import auto_post_classifier.gpt_handler as gpt_handler
from auto_post_classifier.request_builder import RequestBuilder
from auto_post_classifier.models import Post
from auto_post_classifier.gpt_handler import RequestPayload


class ApiManager:
    """Manages API operations including post processing and response handling"""

    def __init__(self) -> None:
        # TODO: move the constants to a separate file
        """Initialize API manager with necessary components"""

        self.gpt_handler = gpt_handler.GptHandler(
            api_key=os.environ["OPENAI_API_KEY"],
        )

        self.request_builder = RequestBuilder()

    def __str__(self) -> str:
        """String representation of the API manager"""
        return json.dumps(self.__dict__, indent=2)

    async def send_requests(
        self, requests: list[gpt_handler.RequestPayload]
    ) -> List[gpt_handler.Response]:
        return [await self.gpt_handler.send_request(request) for request in requests]

    def construct_single_payload(self, post: Post) -> RequestPayload:
        if post.text:
            self.request_builder.add_text_support(post.text)
        if post.image:
            self.request_builder.add_image_support(post.image)

        return RequestPayload(request_config=self.request_builder.build())

    def construct_request_payload(self, posts: dict[str, Post]) -> list[RequestPayload]:
        return [self.construct_single_payload(post) for post in posts.values()]
