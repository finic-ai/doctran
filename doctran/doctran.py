import os
import json
import openai
import uuid
from enum import Enum
from jsonschema import Draft7Validator
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel
import requests
from prompts import Prompts

import pdb

class ContentType(Enum):
    text = "text"
    html = "html"
    pdf = "pdf"
    mbox = "mbox"

class Transformation(Enum):
    compress = "compress"
    denoise = "denoise"
    extract = "extract"
    distill = "distill"
    interrogate = "interrogate"
    redact = "redact"
    translate = "translate"


class Document(BaseModel):
    uri: str
    id: str
    content_type: ContentType
    content: str
    transformations: Optional[Dict[Transformation, str]]
    parent: Optional["Document"] = None
    children: Optional[List["Document"]] = None
    metadata: Optional[Dict[str, Any]] = None

    def get_token_size(self):
        pass

    def get_char_size(self):
        pass

class OpenAIFunctionCall(BaseModel):
    model: str = "gpt-3.5-turbo-0613"
    messages: List[Dict[str, str]]
    functions: List[Dict[str, Any]]
    function_call: str | Dict[str, Any]

class ExtractProperty(BaseModel):
    name: str
    description: str
    type: Literal["string", "number", "boolean", "array", "object"]
    items: Optional[List | Dict[str, Any]]
    required: bool = True

class Doctran:
    def __init__(self, openai_api_key: str, openai_model: str = "gpt-3.5-turbo-0613"):
        self.openai_api_key = openai_api_key
        self.openai_completions_url = "https://api.openai.com/v1/completions"
        self.openai_model = openai_model
        self.openai = openai
        self.openai.api_key = os.environ["OPENAI_API_KEY"]

    def parse(self, *, content: str, content_type: ContentType, uri: str = None, metadata: dict = None) -> Document:
        '''
        Parse raw text and apply different chunking schemes based on the content type.

        Returns:
            Document: the parsed content represented as a Doctran Document
        '''
        if not uri:
            uri = str(uuid.uuid4())
        if content_type == ContentType.text.value:
            # TODO: Recursively chunk the document
            document = Document(id=str(uuid.uuid4()), content_type=content_type, content=content, uri=uri, metadata=metadata)
            return document

    async def extract(self, *, document: Document, properties: List[ExtractProperty], recursive: bool = False) -> dict[str, any]:
        '''
        Use OpenAI function calling to extract structured data from the document.

        Returns:
            values: the extracted values for each parameter
        '''
        function_parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }
        try:
            for property in properties:
                # pdb.set_trace()
                function_parameters["properties"][property.name] = {
                    "type": property.type,
                    "description": property.description,
                }
                if property.items:
                    function_parameters["properties"][property.name]["items"] = property.items
                if property.required:
                    function_parameters["required"].append(property.name)
                # if not Draft7Validator.check_schema(function_parameters["properties"][property.name]):
                #     raise Exception(f"Property {property.name} is not a valid JSON schema.")
        except Exception as e:
            raise Exception(f"Invalid properties provided: {e}")

        try:
            function_call = OpenAIFunctionCall(
                model=self.openai_model, 
                messages=[{"role": "user", "content": document.content}], 
                functions=[{
                    "name": "extract_information",
                    "description": "Extract structured data from a raw text document.",
                    "parameters": function_parameters,
                }],
                function_call={"name": "extract_information"}
            )
            # pdb.set_trace()
            completion = self.openai.ChatCompletion.create(**function_call.dict())
            arguments = completion.choices[0].message["function_call"]["arguments"]
            return json.loads(arguments)
        except Exception as e:
            raise Exception(f"OpenAI function call failed: {e}")

    def compress(self, *, document: Document, token_limit: int, recursive: bool = False):
        pass

    def denoise(self, *, document: Document, recursive: bool = False):
        pass