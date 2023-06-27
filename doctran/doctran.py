import os
import json
import openai
import uuid
from enum import Enum
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
    interrogate = "interrogate"
    redact = "redact"
    translate = "translate"

# Not easily retrievalble from the presidio library so it should be kept up to date manually based on
# https://microsoft.github.io/presidio/supported_entities/
class RecognizerEntity(Enum):
    CREDIT_CARD = "CREDIT_CARD"
    CRYPTO = "CRYPTO"
    DATE_TIME = "DATE_TIME"
    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    IBAN_CODE = "IBAN_CODE"
    IP_ADDRESS = "IP_ADDRESS"
    NRP = "NRP"
    PHONE_NUMBER = "PHONE_NUMBER"
    URL = "URL"
    LOCATION = "LOCATION"
    PERSON = "PERSON"
    MEDICAL_LICENSE = "MEDICAL_LICENSE"
    US_BANK_NUMBER = "US_BANK_NUMBER"
    US_DRIVER_LICENSE = "US_DRIVER_LICENSE"
    US_ITIN = "US_ITIN"
    US_PASSPORT = "US_PASSPORT"
    US_SSN = "US_SSN"
    UK_NHS = "UK_NHS"
    ES_NIF = "ES_NIF"
    IT_FISCAL_CODE = "IT_FISCAL_CODE"
    IT_DRIVER_LICENSE = "IT_DRIVER_LICENSE"
    IT_VAT_CODE = "IT_VAT_CODE"
    IT_PASSPORT = "IT_PASSPORT"
    IT_IDENTITY_CARD = "IT_IDENTITY_CARD"
    SG_NRIC_FIN = "SG_NRIC_FIN"
    AU_ABN = "AU_ABN"
    AU_ACN = "AU_ACN"
    AU_TFN = "AU_TFN"
    AU_MEDICARE_NUMBER = "AU_MEDICARE_NUMBER"

class Document(BaseModel):
    uri: str
    id: str
    content_type: ContentType
    raw_content: str
    transformed_content: str
    interrogation: Optional[Dict[str, Any]] = None
    extracted_properties: Optional[Dict[str, Any]] = None
    applied_transformations: Optional[List[Transformation]] = None
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
    enum: Optional[List[str]]
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
            document = Document(id=str(uuid.uuid4()), content_type=content_type, raw_content=content, transformed_content=content, uri=uri, metadata=metadata)
            return document

    async def extract(self, *, document: Document, properties: List[ExtractProperty]) -> Document:
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
        for prop in properties:
            function_parameters["properties"][prop.name] = {
                "type": prop.type,
                "description": prop.description,
                **({"items": prop.items} if prop.items else {}),
                **({"enum": prop.enum} if prop.enum else {}),
            }
            if prop.required:
                function_parameters["required"].append(prop.name)

        try:
            function_call = OpenAIFunctionCall(
                model=self.openai_model, 
                messages=[{"role": "user", "content": document.transformed_content}], 
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
            try:
                document.extracted_properties = json.loads(arguments)
            except Exception as e:
                raise Exception(f"OpenAI returned malformatted json: {e} {arguments}")
            return document
        except Exception as e:
            raise Exception(f"OpenAI function call failed: {e}")

    # TODO: Use OpenAI function call to compress a document to under a certain token limit
    def compress(self, *, document: Document, token_limit: int) -> Document:
        pass
    
    # TODO: Use OpenAI function call to remove irrelevant information from a document
    def denoise(self, *, document: Document, topics: List[str]) -> Document:
        pass
    
    # TODO: Use OpenAI function call to convert documents to question and answer format
    def interrogate(self, *, document: Document) -> List[dict[str, str]]:
        pass
    
    # TODO: Use presidio or similar libraries to redact sensitive information. Cannot use a hosted 3rd party
    # service for this because of privacy concerns.
    def redact(self, *, document: Document, entities: List[RecognizerEntity | str] = None) -> Document:
        '''
        Use presidio to redact sensitive information from the document.

        Returns:
            document: the document with content redacted from document.transformed_content
        '''
        for entity, i in entities:
            if entity in RecognizerEntity.__members__:
                entities[i] = RecognizerEntity[entity]
            elif entity not in RecognizerEntity.__members__.values():
                raise Exception(f"Invalid entity type: {entity}")

        import spacy
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        try:
            spacy.load("en_core_web_lg")
        except OSError:
            while True:
                response = input("en_core_web_lg model not found, but is required to run presidio-anonymizer. Download it now? (Y/n)")
                if response.lower() in ["n", "no"]:
                    raise Exception("Cannot run presidio-anonymizer without en_core_web_lg model.")
                elif response.lower() in ["y", "yes", ""]:
                    print("Downloading...")
                    from spacy.cli.download import download
                    download(model="en_core_web_lg")
                    break
                else:
                    print("Invalid response.")
        text = document.transformed_content
        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()
        results = analyzer.analyze(text=text,
                                   entities=[entity.value for entity in entities] if entities else None,
                                   language='en')
        anonymized_text = anonymizer.anonymize(text=text, analyzer_results=results)
        print(anonymized_text)
        document.transformed_content = str(anonymized_text)


    # TODO: Use OpenAI function call to translate this document to another language
    def translate(self, *, document: Document, language: str) -> Document:
        pass