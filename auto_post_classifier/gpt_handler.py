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

from re import A
from typing import Dict, Any
from openai import OpenAI
from pydantic import BaseModel

from auto_post_classifier.models import Post
from auto_post_classifier.request_builder import (
    RequestConfigModel,
    Message,
    MessageContent,
)


# import openai
class Response(BaseModel):
    category: str
    reason: str


class RequestPayload(BaseModel):
    request_config: RequestConfigModel


# def create_request_payload(config: RequestConfigModel, post: Post) -> RequestPayload:
#     """
#     Combine configuration with a model to create a request payload

#     Args:
#         config: Request configuration model
#         model: GPT model to use

#     Returns:
#         dict: Request payload
#     """
#     request_builder = RequestBuilder()
#     request_builder.add_text_support(post.text)
#     request_builder.add_image_support(post.image)

#     return RequestPayload(request_config=request_builder.build(), post=post)


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
        self,
        request_payload: RequestPayload,
    ) -> Response:
        """
        Send a classification request to GPT

        Args:
            text: Text to classify
            model: GPT model to use

        Returns:
            dict: Classification result with category and reason
        """
        try:
            print(request_payload.request_config.model_dump())
            # Send request using configuration
            response = self.client.chat.completions.create(
                **request_payload.request_config.model_dump()
            )

            response_content = json.loads(response.choices[0].message.content)

            return response_content
        except Exception as e:
            # TODO: fix this - code not reached
            raise e

    # Remove or modify other methods that are no longer needed...
