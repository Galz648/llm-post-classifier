from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ResponseFormat(BaseModel):
    type: str
    json_schema: Dict[str, Any]

class Message(BaseModel):
    role: str
    content: str

class JsonSchema(BaseModel):
    type: str
    properties: Dict[str, Any]
    required: List[str]

class CategoryProperties(BaseModel):
    type: str
    enum: List[str]

class ReasonProperties(BaseModel):
    type: str

class ScoreProperties(BaseModel):
    type: str

class RequestConfig(BaseModel):
    model: str
    temperature: float
    max_completion_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    response_format: ResponseFormat
    messages: List[Message]

# Define individual components
category_properties = CategoryProperties(
    type="string",
    enum=[
        "antisemitism",
        "graphic_violence",
        "antiZionism_extremist",
        "pro_palestine",
        "endorsement_of_terrorism",
        "antiIsrael_extremist",
        "holocaust",
        "misinformation"
    ]
)


reason_properties = ReasonProperties(type="string")
score_properties = ScoreProperties(type="number") # TODO: mot sure this is the right type for score
json_schema = JsonSchema(
    type="object",
    properties={
        "category": category_properties.model_dump(),
        "explanation": reason_properties.model_dump(),
        "score": score_properties.model_dump()
    },
    required=["category", "explanation", "score"]
)

response_format = ResponseFormat(
    type="json",
    json_schema=json_schema.model_dump()
)

system_message = Message(
    role="system",
    content=(
        "You are helping to keep the internet safe.\n"
        "You're tasked with evaluating a post enclosed by three backticks to rank it in the following dimensions:\n"
        "- antisemitism\n"
        "- anti-Israel extremism\n"
        "- graphic violence\n"
        "- weapons\n"
        "- calls for violence\n"
        "- endorsement of terrorism\n"
        "- misinformation\n\n"
        "Use the ranking system:\n"
        "-1: The dimension is not present in the post.\n"
        "0: It's unclear if the dimension is in the post.\n"
        "1: The dimension is highly likely to be in the post.\n\n"
        "Provide an explanation for each ranking."
    )
)

# Assemble the request configuration
request_config = RequestConfig(
    model="gpt-4o-mini",
    temperature=0.2,
    max_completion_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format=response_format,
    messages=[system_message]
)

def load_request_config():
    return RequestConfig.model_validate(request_config)
