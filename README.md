<h2 align="center">
🐛 Doctran
</h2>

<p align="center">
  <p align="center"><b>Doc</b>ument <b>tran</b>sformation library for AI knowledge</p>
</p>
<p align="center">
<a href="https://join.slack.com/t/psychicapi/shared_invite/zt-1ty1wz6w0-8jkmdvBpM5kj_Fh30EiCcg" target="_blank">
    <img src="https://img.shields.io/badge/slack-join-blue.svg?logo=slack" alt="Slack">
</a>
</a>
<a href="https://github.com/psychic-api/doctran/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/static/v1?label=license&message=GPL-3.0&color=blue" alt="License">
</a>
<a href="https://github.com/psychic-api/doctran/issues?q=is%3Aissue+is%3Aclosed" target="_blank">
    <img src="https://img.shields.io/github/issues-closed/psychic-api/doctran?color=blue" alt="Issues">
</a>
  <a href="https://twitter.com/psychicapi" target="_blank">
    <img src="https://img.shields.io/twitter/follow/psychicapi?style=social" alt="Twitter">
</a>
</p>

Vector databases are useful for retrieving context for LLMs, however they struggle to find relevant information if the source documents are indexed hapharzardly and information is sparse. Doctran is an open-source library that uses LLMs and open source NLP libraries to transform raw text into clean, structured, information-dense documents that are optimized for vector space retrieval.

Doctran is maintained by [Psychic](https://github.com/psychic-api/psychic), the data integration layer for LLMs.

## Getting Started
`pip install doctran`

```python
from doctran import Doctran

doctran = Doctran(openai_api_key=OPENAI_API_KEY)
document = doctran.parse(content="your_content_as_string")
```
Clone or download the notebook [here](/examples/doctran_examples.ipynb) for interactive examples.

## Chaining transformations
Doctran is designed to make chaining document transformations easy. For example, you may want to first redact all PII from a document before sending it over to OpenAI to be summarized.

Ordering is important when chaining transformations - transformations that are invoked first will be executed first, and its result will be passed to the next transformation.

```python
document = await document.redact(entities=["EMAIL_ADDRESS", "PHONE_NUMBER"]).extract(properties).summarize().execute()
```

## Doctransformers

### Extract
Given any valid JSON schema, yses OpenAI function calling to extract structured data from a document.

```python
from doctran import ExtractProperty

properties = ExtractProperty(
    name="millenial_or_boomer", 
    description="A prediction of whether this document was written by a millenial or boomer",
    type="string",
    enum=["millenial", "boomer"],
    required=True
)
document = await document.extract(properties=properties).execute()
```

### Redact
Uses a spaCy model to remove names, emails, phone numbers and other sensitive information from a document. Runs locally to avoid sending sensitive data to third party APIs.

```python
document = await document.redact(entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"]).execute()
```

### Summarize
Summarize the information in a document. `token_limit` may be passed to configure the size of the summary, however it may not be respected by OpenAI.

```python
document = await document.summarize().execute()
```

### Refine
Remove all information from a document unless it's related to a specific set of topics.

```python
document = await document.refine(topics=['marketing', 'meetings']).execute()
```

### Translate
Translates text into another language

```python
document = await document.translate(language="spanish").execute()
```

### Interrogate
Convert information in a document into question and answer format. End user queries often take the form of a question, so converting information into questions and creating indexes from these questions often yields better results when using a vector database for context retrieval.

```python
document = await document.interrogate().execute()
```

## Contributing
Doctran is open to contributions! The best way to get started is to contribute a new document transformer. Transformers that don't rely on API calls (e.g. OpenAI) are especially valuable since they can run fast and don't require any external dependencies.

### Adding a new doctransformer
Contributing new transformers is straightforward.

1. Add a new class that extends `DocumentTransformer` or `OpenAIDocumentTransformer`
2. Implement the `__init__` and `transform` methods
3. Add corresponding methods to the `DocumentTransformationBuilder` and `Document` classes to enable chaining

