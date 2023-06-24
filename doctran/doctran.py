from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import requests
from prompts import Prompts

class ContentType(Enum):
    text = "text"
    html = "html"
    pdf = "pdf"

class Transformation(Enum):
    compress = "compress"
    denoise = "denoise"
    extract = "extract"
    distill = "distill"
    interrogate = "interrogate"
    redact = "redact"
    translate = "translate"


class Block(BaseModel):
    id: str
    content_type: ContentType
    content: str
    token_size: int
    char_size: int
    transformations: Optional[Dict[Transformation, str]]
    parent: Optional["Block"] = None
    children: Optional[List["Block"]] = None
    metadata: Optional[Dict[str, Any]] = None

class Document(BaseModel):
    root_bloc: Block
    content_type: ContentType
    uri: str
    metadata: Optional[Dict[str, Any]] = None

class Doctran:
    def __init__(self, openai_api_key: str, openai_model: str = "gpt-3.5-turbo"):
        self.openai_api_key = openai_api_key
        self.openai_completions_url = "https://api.openai.com/v1/completions"
        self.openai_model = openai_model

    def parse(self, *, content: str, content_type: ContentType, metadata: dict = None) -> Document:
        '''
        Parse raw text and apply different chunking schemes based on the content type.

        Returns:
        Document: the parsed content represented as a tree structure of blocks
        '''
        pass

    def compress(self, *, block: Block, token_limit: int, recursive: bool = False):
        pass

    def denoise(self, *, block: Block, recursive: bool = False):
        pass