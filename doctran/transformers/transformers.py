from enum import Enum
import json
from abc import ABC, abstractmethod
import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel
import tiktoken
from doctran import Document, DoctranConfig, ExtractProperty, RecognizerEntity

class TooManyTokensException(Exception):
    def __init__(self, content_token_size: int, token_limit: int):
        super().__init__(f"OpenAI document transformation failed. The document is {content_token_size} tokens long, which exceeds the token limit of {token_limit}.")

class OpenAIChatCompletionCall(BaseModel):
    deployment_id: Optional[str] = None
    model: str = "gpt-3.5-turbo-0613"
    messages: List[Dict[str, str]]
    temperature: int = 0
    max_tokens: Optional[int] = None

class OpenAIFunctionCall(OpenAIChatCompletionCall):
    functions: List[Dict[str, Any]]
    function_call: Union[str, Dict[str, Any]]

class DocumentTransformer(ABC):
    config: DoctranConfig
    
    def __init__(self, config: DoctranConfig) -> None:
        self.config = config

    @abstractmethod
    def transform(self, document: Document) -> Document:
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
    
    def transform(self, document: Document) -> Document:
        encoding = tiktoken.encoding_for_model(self.config.openai_model)
        content_token_size = len(encoding.encode(document.transformed_content))
        try:
            if content_token_size > self.config.openai_token_limit:
                raise TooManyTokensException(content_token_size, self.config.openai_token_limit)
        except Exception as e:
            print(e)
            return document
        return self.executeOpenAICall(document)

    def executeOpenAICall(self, document: Document) -> Document:
        try:
            function_call = OpenAIFunctionCall(
                deployment_id=self.config.openai_deployment_id,
                model=self.config.openai_model, 
                messages=[{"role": "user", "content": document.transformed_content}], 
                functions=[{
                    "name": self.function_name,
                    "description": self.function_description,
                    "parameters": self.function_parameters,
                }],
                function_call={"name": self.function_name}
            )
            completion = self.config.openai.ChatCompletion.create(**function_call.dict())
            arguments = completion.choices[0].message["function_call"]["arguments"]
            try:
                arguments = json.loads(arguments)
            except Exception as e:
                raise Exception("OpenAI returned malformatted JSON" +
                                "This is likely due to the completion running out of tokens. " +
                                f"Setting a higher token limit may fix this error. JSON returned: {arguments}")
            first_value = next(iter(arguments.values()))
            if len(arguments) > 1 or not isinstance(first_value, str):
                # If multiple arguments or a dict/list is returned, treat arguments as extracted values
                document.extracted_properties = document.extracted_properties or arguments
            else:
                # If there is only one argument and it's a string, treat arguments as transformed content
                document.transformed_content = first_value
            return document
        except Exception as e:
            raise Exception(f"OpenAI function call failed: {e}")

class DocumentExtractor(OpenAIDocumentTransformer):
    '''
    Use OpenAI function calling to extract structured data from the document.

    Returns:
        document: the original document, with the extracted properties added to the extracted_properties field
    '''
    properties: List[ExtractProperty]

    def __init__(self, *, config: DoctranConfig, properties: List[ExtractProperty]) -> None:
        super().__init__(config)
        self.properties = properties
        self.function_name = "extract_information"
        self.function_description = "Extract structured data from a raw text document."
        for prop in self.properties:
            self.function_parameters["properties"][prop.name] = {
                "type": prop.type,
                "description": prop.description,
                **({"items": prop.items} if prop.items else {}),
                **({"enum": prop.enum} if prop.enum else {}),
            }
            if prop.required:
                self.function_parameters["required"].append(prop.name)

class DocumentSummarizer(OpenAIDocumentTransformer):
    '''
    Use OpenAI function calling to summarize the document to under a certain token limit.

    Returns:
        document: the original document, with the summarized content added to the transformed_content field
    '''
    token_limit: int

    def __init__(self, *, config: DoctranConfig, token_limit: int) -> None:
        super().__init__(config)
        self.token_limit = token_limit
        self.function_name = "summarize"
        self.function_description = f"Summarize a document in under {self.token_limit} tokens."
        self.function_parameters["properties"]["summary"] = {
            "type": "string",
            "description": "The summary of the document.",
        }
        self.function_parameters["required"].append("summary")

class DocumentRedactor(DocumentTransformer):
    '''
    Use presidio to redact sensitive information from the document.

    Returns:
        document: the document with content redacted from document.transformed_content
    '''
    entities: List[str]
    spacy_model: str
    interactive: bool

    def __init__(self, *, config: DoctranConfig, entities: List[Union[RecognizerEntity, str]] = None, spacy_model: str = "en_core_web_md", interactive: bool = True) -> None:
        super().__init__(config)
        # TODO: support multiple NER models and sizes
        # Entities can be provided as either a string or enum, so convert to string in a all cases
        if spacy_model not in ["en_core_web_sm", "en_core_web_md", "en_core_web_lg", "en_core_web_trf"]:
            raise Exception(f"Invalid spacy english language model: {spacy_model}")
        self.spacy_model = spacy_model
        self.interactive = interactive
        for i, entity in enumerate(entities):
            if entity in RecognizerEntity.__members__:
                entities[i] = RecognizerEntity[entity].value
            else:
                raise Exception(f"Invalid entity type: {entity}")
        self.entities = entities
    
    def transform(self, document: Document) -> Document:
        try:
            spacy.load(self.spacy_model)
        except OSError:
            from spacy.cli.download import download
            if not self.interactive:
                download(model="en_core_web_md")
            else:
                while True:
                    response = input(f"{self.spacy_model} model not found, but is required to run presidio-anonymizer. Download it now? (~40MB) (Y/n)")
                    if response.lower() in ["n", "no"]:
                        raise Exception(f"Cannot run presidio-anonymizer without {self.spacy_model} model.")
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
                        "model_name": self.spacy_model
                        }]
        })
        nlp_engine = nlp_engine_provider.create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        anonymizer = AnonymizerEngine()
        results = analyzer.analyze(text=text,
                                   entities=self.entities if self.entities else None,
                                   language='en')
        # TODO: Define customer operator to replace data types with numbered placeholders to differentiate between different PERSONs, EMAILs, etc
        anonymized_data = anonymizer.anonymize(text=text, 
                                               analyzer_results=results,
                                               operators={"DEFAULT": OperatorConfig("replace")})
        
        # Extract just the anonymized text, discarding items metadata
        anonymized_text = anonymized_data.text
        document.transformed_content = anonymized_text
        return document

class DocumentRefiner(OpenAIDocumentTransformer):
    '''
    Use OpenAI function calling to remove irrelevant information from the document.

    Returns:
        Document: the refined content represented as a Doctran Document
    '''
    topics: List[str]

    def __init__(self, *, config: DoctranConfig, topics: List[str] = None) -> None:
        super().__init__(config)
        self.topics = topics
        self.function_name = "refine"
        if topics:
            self.function_description = f"Remove all information from a document that is not relevant to the following topics: -{' -'.join(self.topics)}"
        else:
            self.function_description = "Remove all irrelevant information from a document."
        self.function_parameters["properties"]["refined_document"] = {
            "type": "string",
            "description": "The document with irrelevant information removed.",
        }
        self.function_parameters["required"].append("refined_document")

class DocumentTranslator(OpenAIDocumentTransformer):
    '''
    Use OpenAI function calling to translate the document to another language.

    Returns:
        Document: the translated document represented as a Doctran Document
    '''
    language: str

    def __init__(self, *, config: DoctranConfig, language: str) -> None:
        super().__init__(config)
        self.function_name = "translate"
        self.function_description = f"Translate a document into {language}"
        self.function_parameters["properties"]["translated_document"] = {
            "type": "string",
            "description": f"The document translated into {language}."
        }
        self.function_parameters["required"].append("translated_document")

class DocumentInterrogator(OpenAIDocumentTransformer):
    '''
    Use OpenAI function calling to convert the document to a series of questions and answers.

    Returns:
        Document: the interrogated document represented as a Doctran Document
    '''
    def __init__(self, *, config: DoctranConfig) -> None:
        super().__init__(config)
        self.function_name = "interrogate"
        self.function_description = "Convert a text document into a series of questions and answers."
        self.function_parameters["properties"]["questions_and_answers"] = {
            "type": "array",
            "description": "The list of questions and answers.",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question.",
                    },
                    "answer": {
                        "type": "string",
                        "description": "The answer.",
                    },
                },
                "required": ["question", "answer"],
            },
        }
        self.function_parameters["required"].append("questions_and_answers")