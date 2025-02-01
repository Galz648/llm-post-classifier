"""
GPT Handler Module

This module provides functionality to interact with OpenAI's GPT models for text classification.
It supports configurable request schemas and response validation through external configuration files.

Classes:
    GPT_MODEL: Available GPT models
    GPT_ERROR_REASONS: Possible error types
    ResponseValidator: Validates GPT responses
    GptHandler: Main handler for GPT API interactions
"""

import json

from typing import Dict, Any
from loguru import logger
from openai import OpenAI

from auto_post_classifier.request_builder import (
    RequestConfigModel,
    Message,
    MessageContent,
)

# import openai


class GPT_MODEL:
    """Available GPT models for classification"""

    GPT_4_MINI = "gpt-4o-mini"


# TODO: consider transforming each error reason into a class that has a message and a code, while inheriting from some base class Error
# TODO: replace with existing fastapi errors
class GPT_ERROR_REASONS:
    """Possible error reasons during API interaction"""

    TO_MANY_REQUESTS = "many_requests"
    JSON_VALIDATION = "json_validation"
    CANT_OPEN_RESPONSE_JSON = "cant_open_response_json"


class GptHandler:
    """Handles interactions with OpenAI's GPT API"""

    def __init__(
        self,
        api_key: str,
    ) -> None:
        """
        Initialize GPT handler

        Args:
            responses_path: Path to store responses
            api_key: OpenAI API key
            request_config_path: Path to request configuration JSON
        """
        self.client = OpenAI(api_key=api_key)

    async def send_request(
        self, text: str, request_config: RequestConfigModel, model: Any
    ) -> Dict[str, Any]:
        """
        Send a classification request to GPT

        Args:
            text: Text to classify
            model: GPT model to use

        Returns:
            dict: Classification result with category and reason
        """
        try:
            # Create messages with user content
            messages = request_config.messages.copy()
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": text}]}
            )

            # Send request using configuration
            response = self.client.chat.completions.create(
                **request_config.model_dump()
            )

            response_content = json.loads(response.choices[0].message.content)

            return response_content
        except Exception as e:
            # TODO: fix this - code not reached
            raise e

    # Remove or modify other methods that are no longer needed...
