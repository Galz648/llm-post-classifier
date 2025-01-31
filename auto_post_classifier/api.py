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

from .models import Post
from .utils import get_var_from_env


class PreRequestValidator:
    """Validates posts before they are sent to GPT"""
    
    def validate_length(self, post: Post) -> bool:
        """Check if post meets minimum length requirement"""
        return len(post.text) > 10

    def validate(self, post: Post) -> bool:
        """Main validation method for posts"""
        return True


class ApiManager:
    """Manages API operations including post processing and response handling"""

    def get_config(self) -> Dict:
        """Get current configuration of the API manager"""
        return {
            "gpt": str(self.gpt_handler),
        }

    def __init__(self) -> None:
        # TODO: move the constants to a separate file
        """Initialize API manager with necessary components"""
        self.pre_request_validator = PreRequestValidator()
        self.gpt_handler = gpt_handler.GptHandler(
            responses_path=pathlib.Path("responses.txt"),
            api_key=os.environ["OPENAI_API_KEY"],
            request_config_path=pathlib.Path("config/request_config.json"),
            mock_file=get_var_from_env("MOCK_FILE"),
        )

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
        
        for uuid, post in json_posts.items():
            # TODO: validate posts in the route itself - not here, separation of concerns
            if self.pre_request_validator.validate(post):
                response = await self.gpt_handler.send_request(post.text)
                responses[uuid] = response

        if len(json_posts) != len(responses):
            logger.warning(
                f"Number of requests and responses not equal. "
                f"Sent {len(json_posts)} posts and got {len(responses)} responses"
            )

        
        return responses
