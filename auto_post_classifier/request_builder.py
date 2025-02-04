from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict, Any
from pydantic import ValidationError

# TODO: extract pydantic models to config/request_config.py
from enum import Enum

EnumCategories = [
    "ClassicAntisemitism",
    "ProPalestine",
    "Holocaust",
    "AntiIsraeli",
    "ProHamas",
    "NonHarmful",
]


class Reason(BaseModel):

    reason: str


class MessageContent(BaseModel):
    type: str
    text: str


class Message(BaseModel):
    role: str
    content: List[MessageContent]


class JsonSchemaProperties(BaseModel):
    # TODO: change the categories to an enum
    category: List[str]
    # TODO: change the reason to a string or enum of reasons
    reason: str


class JsonSchema(BaseModel):
    type: str
    required: List[str]
    additionalProperties: bool


class Schema(BaseModel):
    type: str
    # properties: JsonSchemaProperties
    properties: Dict[str, Any]
    required: List[str]
    additionalProperties: bool


class ResponseFormat(BaseModel):
    type: str
    json_schema: JsonSchema


class TuningParameters(BaseModel):
    temperature: float
    max_completion_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float


class RequestConfigModel(BaseModel):
    model: str
    messages: List[Message]
    response_format: Dict[str, Any]
    temperature: float
    max_completion_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float


# End Generation Here
class RequestBuilder:
    def __init__(self) -> None:
        self.client = OpenAI()

        # TODO: not sure if this is the best way to initialize the config with pydantic
        self.config: RequestConfigModel = RequestConfigModel(
            model="",
            messages=[],
            response_format={},
            temperature=0.0,
            max_completion_tokens=1000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

    def build(self):

        try:
            # populate
            # validate the config
            # TODO: move validations to the config model
            RequestConfigModel.model_validate(self.config)
            if not self.config.messages:
                raise ValueError("Messages must be provided")
            if not self.config.response_format:
                raise ValueError("Response format must be provided")

            if not self.config.model:
                raise ValueError("Model must be provided")

        # TODO: add error handling for other validation errors
        except ValidationError as e:
            print("Validation error:", e.json())
            raise e

        return self.config

    def add_text_support(self, text: str):
        # TODO: consider changing method name to "add_post_text",

        if not text:
            raise ValueError("Text must be provided")

        self.config.messages.append(
            Message(role="user", content=[MessageContent(type="text", text=text)])
        )
        return self

    def add_image_support(self, image: str):
        if not image:
            raise ValueError("Image must be provided")

    def add_response_format(self, response_format: Dict[str, Any]):
        if not response_format:
            raise ValueError("Response format must be provided")

        self.config.response_format = response_format
        return self

    def add_model(self, model: str):
        if not model:
            raise ValueError("Model must be provided")

        self.config.model = model
        return self
