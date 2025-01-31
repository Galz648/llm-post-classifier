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
import os
from pathlib import Path
from typing import Optional, Dict, Any

import jinja2
from loguru import logger
from openai import OpenAI
# import openai


class GPT_MODEL:
    """Available GPT models for classification"""
    GPT_4_MINI = "gpt-4o-mini"

# TODO: consider transforming each error reason into a class that has a message and a code, while inheriting from some base class Error
class GPT_ERROR_REASONS:
    """Possible error reasons during API interaction"""
    TO_MANY_REQUESTS = "many_requests"
    JSON_VALIDATION = "json_validation"
    CANT_OPEN_RESPONSE_JSON = "cant_open_response_json"

# TODO: replace code with pydantic model
class ResponseValidator:
    # TODO: validate json schema with pydantic
    """Validates responses from GPT API against expected schema"""


    def __init__(self, valid_categories: list[str]) -> None:
        """
        Initialize validator with configuration
        
        Args:
            valid_categories: List of valid category values
        """
        self.validators = [self.validate_json_schema]
        self.valid_categories = valid_categories

    def validate_json_schema(self, response_dict: dict) -> bool:
        # TODO: validate json schema with pydantic
        """
        Validates response against expected schema
        
        Args:
            response_dict: Response dictionary from GPT
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_fields = ["category", "reason"]
        
        try:
            # Check required fields
            if not all(field in response_dict for field in required_fields):
                return False
            
            # Validate category
            if response_dict["category"] not in self.valid_categories:
                return False
                
            # Validate reason is string and not empty
            if not isinstance(response_dict["reason"], str) or not response_dict["reason"]:
                return False
                
            return True
        except Exception as e:
            logger.warning(f"JSON validation error: {e}")
            return False

    def validate(self, response: dict) -> tuple[bool, str, dict]:
        # TODO: validate json schema with pydantic
        """
        Main validation method
        
        Args:
            response: Response dictionary to validate
            
        Returns:
            tuple: (is_valid, error_message, response_dict)
        """
        try:
            if not self.validate_json_schema(response):
                return (False, "validate_json_schema", response)
            return (True, "", response)
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return (False, "validation_error", {})


class GptHandler:
    """Handles interactions with OpenAI's GPT API"""

    def __init__(
        self,
        responses_path: Path,
        api_key: str,
        request_config_path: Optional[Path] = None,
    ) -> None:
        """
        Initialize GPT handler
        
        Args:
            responses_path: Path to store responses
            api_key: OpenAI API key
            request_config_path: Path to request configuration JSON
        """
        self.client = OpenAI(api_key=api_key)
        self.responses_path = responses_path
        
        # Load request configuration
        self.request_config = self._load_request_config(request_config_path)
        
        # Initialize validator with categories from config
        valid_categories = self.request_config["response_format"]["json_schema"]["schema"][
            "properties"]["category"]["enum"]
        self.response_validator = ResponseValidator(valid_categories)

    def _load_request_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        # TODO: validate json schema with pydantic
        """
        Load request configuration from file
        
        Args:
            config_path: Path to configuration JSON file
            
        Returns:
            dict: Request configuration
        
        Raises:
            ValueError: If no config file is provided or file doesn't exist
        """
        if not config_path:
            raise ValueError(
                "Request configuration file path is required. "
                "Please provide a JSON file containing the OpenAI API request schema. "
                "You can export this configuration from the OpenAI playground. "
                "Expected location: <project_root>/config/request_config.json"
            )
        
        if not config_path.exists():
            raise ValueError(
                f"Request configuration file not found at: {config_path}\n"
                "Please ensure the file exists and contains a valid OpenAI API request schema. "
            )
        
        with open(config_path) as f:
            return json.load(f)

    async def send_request(self, text: str, model: str = GPT_MODEL.GPT_4_MINI) -> Dict[str, Any]:
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
            messages = self.request_config["messages"].copy()
            messages.append({"role": "user", "content": text})
            
            # Send request using configuration
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=self.request_config["response_format"],
                temperature=self.request_config["temperature"],
                max_completion_tokens=self.request_config["max_completion_tokens"],
                top_p=self.request_config["top_p"],
                frequency_penalty=self.request_config["frequency_penalty"],
                presence_penalty=self.request_config["presence_penalty"]
            )
            
            response_content = json.loads(response.choices[0].message.content)
            validation_result = self.response_validator.validate(response_content)
            
            if validation_result[0]:
                # TODO: cleaner with fastAPI
                return {
                    "text": text,
                    "category": response_content["category"],
                    "reason": response_content["reason"],
                    "error": ""
                }
            else:
                return {
                    "text": text,
                    "error": validation_result[1],
                    "raw_response": response_content
                }
                
        except Exception as e:
            # TODO: fix this - code not reached
            raise e
            logger.error(f"Error processing request: {e}")
            return {
                "text": text,
                "error": str(e),
                "raw_response": None
            }

    # Remove or modify other methods that are no longer needed...
