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

class ExtractProperty(BaseModel):
    name: str
    description: str
    type: Literal["string", "number", "boolean", "array", "object"]
    items: Optional[List | Dict[str, Any]]
    enum: Optional[List[str]]
    required: bool = True

class DoctranConfig(BaseModel):
    openai_api_key: str
    openai_model: str
    openai_completions_url: str
    openai: Any
    openai_token_limit: int

class ContentType(Enum):
    text = "text"
    html = "html"
    pdf = "pdf"
    mbox = "mbox"

class Transformation(Enum):
    summarize = "DocumentSummarizer"
    refine = "DocumentRefiner"
    extract = "DocumentExtractor"
    interrogate = "DocumentInterrogator"
    redact = "DocumentRedactor"
    translate = "DocumentTranslator"

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
    config: DoctranConfig
    extracted_properties: Optional[Dict[str, Any]] = {}
    metadata: Optional[Dict[str, Any]] = None

    def extract(self, *, properties: List[ExtractProperty]) -> 'DocumentTransformationBuilder':
        transformation_builder = DocumentTransformationBuilder(self)
        transformation_builder.extract(properties=properties)
        return transformation_builder
    
    def summarize(self, token_limit: int) -> 'DocumentTransformationBuilder':
        transformation_builder = DocumentTransformationBuilder(self)
        transformation_builder.summarize(token_limit=token_limit)
        return transformation_builder

    def redact(self, *, entities: List[RecognizerEntity | str], spacy_model: str = "en_core_web_md", interactive: bool = True) -> 'DocumentTransformationBuilder':
        transformation_builder = DocumentTransformationBuilder(self)
        transformation_builder.redact(entities=entities, spacy_model=spacy_model, interactive=interactive)
        return transformation_builder

    def refine(self, *, topics: List[str] = None) -> 'DocumentTransformationBuilder':
        transformation_builder = DocumentTransformationBuilder(self)
        transformation_builder.refine(topics=topics)
        return transformation_builder

    def translate(self, language: str) -> 'DocumentTransformationBuilder':
        transformation_builder = DocumentTransformationBuilder(self)
        transformation_builder.translate(language=language)
        return transformation_builder

    def interrogate(self) -> 'DocumentTransformationBuilder':
        transformation_builder = DocumentTransformationBuilder(self)
        transformation_builder.interrogate()
        return transformation_builder

class DocumentTransformationBuilder:
    '''
    A builder to enable chaining of document transformations.
    '''
    def __init__(self, document: Document) -> None:
        self.document = document
        self.transformations = []
    
    async def execute(self) -> Document:
        module_name = "doctran.transformers"
        module = importlib.import_module(module_name)
        try:
            transformed_document = self.document.copy()
            for transformation in self.transformations:
                transformer = getattr(module, transformation[0].value)(config=transformed_document.config, **transformation[1])
                transformed_document = await transformer.transform(transformed_document)
            self.transformations = []
            return transformed_document
        except Exception as e:
            raise Exception(f"Error executing transformation {transformation}: {e}")

    def extract(self, *, properties: List[ExtractProperty]) -> 'DocumentTransformationBuilder':
        self.transformations.append((Transformation.extract, {"properties": properties}))
        return self

    def summarize(self, token_limit: int = 100) -> 'DocumentTransformationBuilder':
        self.transformations.append((Transformation.summarize, {"token_limit": token_limit}))
        return self

    def redact(self, *, entities: List[RecognizerEntity | str], spacy_model: str, interactive: bool) -> 'DocumentTransformationBuilder':
        self.transformations.append((Transformation.redact, {"entities": entities, "spacy_model": spacy_model, "interactive": interactive}))
        return self

    def refine(self, *, topics: List[str]) -> 'DocumentTransformationBuilder':
        self.transformations.append((Transformation.refine, {"topics": topics}))
        return self

    def translate(self, *, language: str) -> 'DocumentTransformationBuilder':
        self.transformations.append((Transformation.translate, {"language": language}))
        return self

    def interrogate(self) -> 'DocumentTransformationBuilder':
        self.transformations.append((Transformation.interrogate, {}))
        return self


class Doctran:
    def __init__(self, openai_api_key: str = None, openai_model: str = "gpt-4", openai_token_limit: int = 8000):
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

    def parse(self, *, content: str, content_type: ContentType = "text", uri: str = None, metadata: dict = None) -> Document:
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
