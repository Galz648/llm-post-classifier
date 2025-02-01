from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict, Any
from pydantic import ValidationError

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
        self.config: RequestConfigModel = RequestConfigModel.model_construct(
            model="gpt-4o",
            messages=[],
            response_format={},
        )

    def build(self):
        try:
            # validate the config
            RequestConfigModel.model_validate(self.config)
            if self.config.messages == []:
                raise ValueError("Messages must be provided")
        except ValidationError as e:
            print("Validation error:", e.json())
            raise e

        def send_request():
            return self.client.chat.completions.create(**self.config.model_dump())

        return send_request

    def add_text_support(self, text: str):
        # TODO: consider changing method name to "add_post_text",

        if not text:
            raise ValueError("Text must be provided")

        self.config.messages.append(
            Message(role="user", content=[MessageContent(type="text", text=text)])
        )
        return self
