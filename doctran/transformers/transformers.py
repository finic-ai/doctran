from enum import Enum
import json
from abc import ABC, abstractmethod
import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel
import requests
import tiktoken
from doctran import Document, DoctranConfig, ExtractProperty, RecognizerEntity

class OpenAIChatCompletionCall(BaseModel):
    model: str = "gpt-3.5-turbo-0613"
    messages: List[Dict[str, str]]
    temperature: int = 0
    max_tokens: Optional[int] = None

class OpenAIFunctionCall(OpenAIChatCompletionCall):
    functions: List[Dict[str, Any]]
    function_call: str | Dict[str, Any]

class DocumentTransformer(ABC):
    config: DoctranConfig
    
    def __init__(self, config: DoctranConfig) -> None:
        self.config = config

    @abstractmethod
    async def transform(self, document: Document) -> Document:
        pass

class OpenAIDocumentTransformer(DocumentTransformer):
    function_parameters: Dict[str, Any]
    
    def __init__(self, config: DoctranConfig) -> None:
        super().__init__(config)
        self.function_parameters = {
            "type": "object",
            "properties": {},
            "required": [],
        }
    
    @abstractmethod
    async def transform(self, document: Document) -> Document:
        pass

class DocumentExtractor(OpenAIDocumentTransformer):
    properties: List[ExtractProperty]

    def __init__(self, *, config: DoctranConfig, properties: List[ExtractProperty]) -> None:
        super().__init__(config)
        self.properties = properties
        for prop in self.properties:
            self.function_parameters["properties"][prop.name] = {
                "type": prop.type,
                "description": prop.description,
                **({"items": prop.items} if prop.items else {}),
                **({"enum": prop.enum} if prop.enum else {}),
            }
            if prop.required:
                self.function_parameters["required"].append(prop.name)
    
    async def transform(self, document: Document) -> Document:
        '''
        Use OpenAI function calling to extract structured data from the document.

        Returns:
            document: the original document, with the extracted properties added to the extracted_properties field
        '''
        try:
            function_call = OpenAIFunctionCall(
                model=self.config.openai_model, 
                temperature=0,
                messages=[{"role": "user", "content": document.transformed_content}], 
                functions=[{
                    "name": "extract_information",
                    "description": "Extract structured data from a raw text document.",
                    "parameters": self.function_parameters,
                }],
                function_call={"name": "extract_information"}
            )
            # pdb.set_trace()
            completion = self.config.openai.ChatCompletion.create(**function_call.dict())
            arguments = completion.choices[0].message["function_call"]["arguments"]
            try:
                document.extracted_properties = json.loads(arguments)
            except Exception as e:
                raise Exception(f"OpenAI returned malformatted json: {e} {arguments}")
            return document
        except Exception as e:
            raise Exception(f"OpenAI function call failed: {e}")

class DocumentCompressor(OpenAIDocumentTransformer):
    token_limit: int

    def __init__(self, *, config: DoctranConfig, token_limit: int) -> None:
        super().__init__(config)
        self.token_limit = token_limit
        self.function_parameters["properties"]["summary"] = {
            "type": "string",
            "description": "The summary of the document.",
        }
        self.function_parameters["required"].append("summary")
    
    async def transform(self, document: Document) -> Document:
        '''
        Use OpenAI function calling to summarize the document to under a certain token limit.

        Returns:
            document: the original document, with the compressed content added to the transformed_content field
        '''
        encoding = tiktoken.encoding_for_model(self.config.openai_model)
        content_token_size = len(encoding.encode(document.transformed_content))
        if content_token_size + self.token_limit > 4000:
            raise Exception(f"Cannot compress document to {self.token_limit} tokens, as the document already takes up {content_token_size} tokens.")
        try:
            function_call = OpenAIFunctionCall(
                model=self.config.openai_model,
                temperature=0,
                messages=[
                    {"role": "user", "content": document.transformed_content}
                ], 
                functions=[{
                    "name": "summarize",
                    "description": f"Summarize the document in under {self.token_limit} tokens.",
                    "parameters": self.function_parameters,
                }],
                function_call={"name": "summarize"}
            )
            completion = self.config.openai.ChatCompletion.create(**function_call.dict())
            arguments = completion.choices[0].message["function_call"]["arguments"]
            try:
                # TODO: retry if the summary is longer than the token limit
                document.transformed_content = json.loads(arguments).get("summary")
            except Exception as e:
                raise Exception("OpenAI returned malformatted JSON" +
                                "This is likely due to the completion running out of tokens. " +
                                f"Setting a higher token limit may fix this error. JSON returned: {arguments}")
            return document
        except Exception as e:
            raise Exception(f"OpenAI function call failed: {e}")

class DocumentRedactor(DocumentTransformer):
    entities: List[str]

    def __init__(self, *, config: DoctranConfig, entities: List[RecognizerEntity | str] = None) -> None:
        super().__init__(config)
        # TODO: support multiple NER models and sizes
        # Entities can be provided as either a string or enum, so convert to string in a all cases
        for i, entity in enumerate(entities):
            if entity in RecognizerEntity.__members__:
                entities[i] = RecognizerEntity[entity].value
            else:
                raise Exception(f"Invalid entity type: {entity}")
        self.entities = entities
    
    async def transform(self, document: Document) -> Document:
        '''
        Use presidio to redact sensitive information from the document.

        Returns:
            document: the document with content redacted from document.transformed_content
        '''
        try:
            spacy.load("en_core_web_md")
        except OSError:
            while True:
                response = input("en_core_web_md model not found, but is required to run presidio-anonymizer. Download it now? (~40MB) (Y/n)")
                if response.lower() in ["n", "no"]:
                    raise Exception("Cannot run presidio-anonymizer without en_core_web_md model.")
                elif response.lower() in ["y", "yes", ""]:
                    print("Downloading...")
                    from spacy.cli.download import download
                    download(model="en_core_web_md")
                    break
                else:
                    print("Invalid response.")
        text = document.transformed_content
        nlp_engine_provider = NlpEngineProvider(nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en",
                        "model_name": "en_core_web_md"
                        }]
        })
        nlp_engine = nlp_engine_provider.create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        anonymizer = AnonymizerEngine()
        results = analyzer.analyze(text=text,
                                   entities=self.entities if self.entities else None,
                                   language='en')
        anonymized_data = anonymizer.anonymize(text=text, analyzer_results=results)
        # Extract just the anonymized text, discarding items metadata
        anonymized_text = anonymized_data.text
        document.transformed_content = anonymized_text
        return document