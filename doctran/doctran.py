import os
import importlib
import openai
import uuid
from enum import Enum
import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel
import requests
from doctran.models import ContentType, DoctranConfig, Transformation, ExtractProperty, DenoiseProperty, TranslateProperty, RecognizerEntity

import pdb

class Document(BaseModel):
    uri: str
    id: str
    content_type: ContentType
    raw_content: str
    transformed_content: str
    config: DoctranConfig
    transformation_builder: 'DocumentTransformationBuilder' = None
    extracted_properties: Optional[Dict[str, Any]] = None
    applied_transformations: Optional[List[Transformation]] = []
    metadata: Optional[Dict[str, Any]] = None

    def extract(self, *, properties: List[ExtractProperty]) -> 'DocumentTransformationBuilder':
        if not self.transformation_builder:
            self.transformation_builder = DocumentTransformationBuilder(self)
        self.transformation_builder.extract(properties=properties)
        return self.transformation_builder
    
    def summarize(self, token_limit: int) -> 'DocumentTransformationBuilder':
        if not self.transformation_builder:
            self.transformation_builder = DocumentTransformationBuilder(self)
        self.transformation_builder.summarize(token_limit=token_limit)
        return self.transformation_builder

    def redact(self, *, entities: List[RecognizerEntity | str]) -> 'DocumentTransformationBuilder':
        if not self.transformation_builder:
            self.transformation_builder = DocumentTransformationBuilder(self)
        self.transformation_builder.redact(entities=entities)
        return self.transformation_builder

    def denoise(self, *, topics: List[str] = None) -> 'DocumentTransformationBuilder':
        if not self.transformation_builder:
            self.transformation_builder = DocumentTransformationBuilder(self)
        self.transformation_builder.denoise(topics=topics)
        return self.transformation_builder

    def translate(self) -> 'DocumentTransformationBuilder':
        pass

    def interrogate(self) -> 'DocumentTransformationBuilder':
        pass

class DocumentTransformationBuilder:
    '''
    A builder for Document transformations to enable chaining of transformations.
    '''
    def __init__(self, document: Document) -> None:
        self.document = document
        self.transformations = []
    
    async def execute(self) -> Document:
        module_name = "doctran.transformers"
        module = importlib.import_module(module_name)
        try:
            for transformation in self.transformations:
                transformer = getattr(module, transformation[0].value)(config=self.document.config, **transformation[1])
                self.document = await transformer.transform(self.document)
                self.document.applied_transformations.append(transformation[0])
            return self.document
        except Exception as e:
            raise Exception(f"Error executing transformation {transformation}: {e}")

    def extract(self, *, properties: List[ExtractProperty]) -> 'DocumentTransformationBuilder':
        self.transformations.append((Transformation.extract, {"properties": properties}))
        return self

    def summarize(self, token_limit: int) -> 'DocumentTransformationBuilder':
        self.transformations.append((Transformation.summarize, {"token_limit": token_limit}))
        return self

    def redact(self, *, entities: List[RecognizerEntity | str]) -> 'DocumentTransformationBuilder':
        self.transformations.append((Transformation.redact, {"entities": entities}))
        return self

    def denoise(self, *, topics: List[str] = None) -> 'DocumentTransformationBuilder':
        self.transformations.append((Transformation.denoise, {"topics": topics}))
        return self

    def translate(self, *, language: str) -> 'DocumentTransformationBuilder':
        self.transformations.append((Transformation.translate, {"language": language}))
        return self

    def interrogate(self) -> 'DocumentTransformationBuilder':
        self.transformations.append((Transformation.interrogate, {}))
        return self


class Doctran:
    def __init__(self, openai_api_key: str = None, openai_model: str = "gpt-3.5-turbo-0613", openai_token_limit: int = 4000):
        self.config = DoctranConfig(
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            openai_completions_url="https://api.openai.com/v1/completions",
            openai=openai,
            openai_token_limit=openai_token_limit
        )
        if openai_api_key:
            self.config.openai.api_key = openai_api_key
        elif os.environ["OPENAI_API_KEY"]:
            self.config.openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            raise Exception("No OpenAI API Key provided")

    def parse(self, *, content: str, content_type: ContentType, uri: str = None, metadata: dict = None) -> Document:
        '''
        Parse raw text and apply different chunking schemes based on the content type.

        Returns:
            Document: the parsed content represented as a Doctran Document
        '''
        if not uri:
            uri = str(uuid.uuid4())
        if content_type == ContentType.text.value:
            # TODO: Optional chunking for documents that are too large
            document = Document(id=str(uuid.uuid4()), content_type=content_type, raw_content=content, transformed_content=content, config=self.config, uri=uri, metadata=metadata)
            return document
    
    # TODO: Use OpenAI function call to convert documents to question and answer format
    def interrogate(self, *, document: Document) -> List[dict[str, str]]:
        pass

    async def translate(self, *, document: Document, property: TranslateProperty) -> Document:
        pass
        # '''
        # Use OpenAI function calling to translate the document to another language.

        # Returns:
        # Document: the translated document represented as a Doctran Document
        # '''

        # function_parameters = {
        #     "type": "object",
        #     "properties": {},
        #     "required": [],
        # }

        # function_parameters["properties"][property.name] = {
        #     "type": property.type,
        #     "description": property.description,
        #     "properties": property.properties
        # }
        # if property.required:
        #     function_parameters["required"].append(property.name)

        # try:
        #     function_call = OpenAIFunctionCall(
        #         model=self.openai_model, 
        #         messages=[{"role": "user", "content": document.transformed_content}], 
        #         functions=[{
        #             "name": "translate_text",
        #             "description": "Re-write the whole text content in an other language",
        #             "parameters": function_parameters,
        #         }],
        #         function_call={"name": "translate_text"}
        #     )

        #     completion = self.openai.ChatCompletion.create(**function_call.dict())
        #     arguments = completion.choices[0].message["function_call"]["arguments"]
        #     arguments_dict = json.loads(arguments)
        #     document.transformed_content = arguments_dict["translated_text"]
        #     return document
        # except Exception as e:
        #     raise Exception(f"OpenAI function call failed: {e}")
