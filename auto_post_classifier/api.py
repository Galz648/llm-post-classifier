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
import pathlib
from typing import Dict

from loguru import logger

import auto_post_classifier.consts as consts
import auto_post_classifier.gpt_handler as gpt_handler
from auto_post_classifier.request_builder import RequestBuilder, RequestConfigAndModel

from .models import Post
from .utils import get_var_from_env


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

    @logger.catch
    async def process_posts(self, json_posts: dict[str, Post]) -> Dict:
        """
        Process multiple posts through GPT classifier

        Args:
            json_posts: Dictionary of posts with UUIDs as keys

        Returns:
            dict: Classification results for each post
        """
        responses = {}
        try:
            for uuid, post in json_posts.items():
                # TODO: validate posts in the route itself - not here, separation of concerns
                self.request_builder.add_text_support(post.text)
                request_config = self.request_builder.build()
                response = await self.gpt_handler.send_request(
                    RequestConfigAndModel(
                        request_config=request_config, model="gpt-4o"
                    ),
                    text=post.text,
                )

            responses[uuid] = response
        except Exception as e:
            logger.error(f"Error processing post {uuid}: {e}")
            responses[uuid] = {"error": str(e)}

        if len(json_posts) != len(responses):
            logger.warning(
                f"Number of requests and responses not equal. "
                f"Sent {len(json_posts)} posts and got {len(responses)} responses"
            )

        return responses
