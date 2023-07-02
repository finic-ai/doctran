<h2 align="center">
🐛 Doctran
</h2>

<p align="center">
  <p align="center"><b>Doc</b>ument <b>tran</b>sformation framework to improve vector search results</p>
</p>
<p align="center">
<a href="https://join.slack.com/t/psychicapi/shared_invite/zt-1ty1wz6w0-8jkmdvBpM5kj_Fh30EiCcg" target="_blank">
    <img src="https://img.shields.io/badge/slack-join-blue.svg?logo=slack" alt="Slack">
</a>
</a>
<a href="https://github.com/psychic-api/doctran/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/static/v1?label=license&message=MIT&color=blue" alt="License">
</a>
<a href="https://github.com/psychic-api/doctran/issues" target="_blank">
    <img src="https://img.shields.io/github/issues/psychic-api/doctran?color=blue" alt="Issues">
</a>
  <a href="https://twitter.com/psychicapi" target="_blank">
    <img src="https://img.shields.io/twitter/follow/psychicapi?style=social" alt="Twitter">
</a>
</p>

Vector databases are useful for retrieving context for LLMs, however they struggle to find relevant information if the source documents are indexed hapharzardly and information is sparse. Doctran uses LLMs and open source NLP libraries to transform raw text into clean, structured, information-dense documents that are optimized for vector space retrieval.

Doctran is maintained by [Psychic](https://github.com/psychic-api/psychic), the data integration layer for LLMs.

## Examples
Clone or download [`examples.ipynb`](/examples.ipynb) for interactive demos.

#### Doctran converts messy, unstructured text
```
<doc type="Confidential Document - For Internal Use Only">
<metadata>
<date> &#x004A; &#x0075; &#x006C; &#x0079; &#x0020; &#x0031; , &#x0032; &#x0030; &#x0032; &#x0033; </date>
<subject> Updates and Discussions on Various Topics; </subject>
</metadata>
<body>
<p>Dear Team,</p>
<p>I hope this email finds you well. In this document, I would like to provide you with some important updates and discuss various topics that require our attention. Please treat the information contained herein as highly confidential.</p>
<section>
<header>Security and Privacy Measures</header>
<p>As part of our ongoing commitment to ensure the security and privacy of our customers' data, we have implemented robust measures across all our systems. We would like to commend John Doe (email: john.doe&#64;example.com) from the IT department for his diligent work in enhancing our network security. Moving forward, we kindly remind everyone to strictly adhere to our data protection policies and guidelines. Additionally, if you come across any potential security risks or incidents, please report them immediately to our dedicated team at security&#64;example.com.</p>
</section>
<section>
<header>HR Updates and Employee Benefits</header>
<p>Recently, we welcomed several new team members who have made significant contributions to their respective departments. I would like to recognize Jane Smith (SSN: &#x0030; &#x0034; &#x0039; - &#x0034; &#x0035; - &#x0035; &#x0039; &#x0032; &#x0038;) for her outstanding performance in customer service. Jane has consistently received positive feedback from our clients. Furthermore, please remember that the open enrollment period for our employee benefits program is fast approaching. Should you have any questions or require assistance, please contact our HR representative, Michael Johnson (phone: &#x0034; &#x0031; 
...
```

#### Into structured or semi-structured documents that are optimized for vector search.

```
{
  "topics": ["Security and Privacy", "HR Updates", "Marketing", "R&D"],
  "summary": "The document discusses updates on security measures, HR, marketing initiatives, and R&D projects. It commends John Doe for enhancing network security, welcomes new team members, and recognizes Jane Smith for her customer service. It also mentions the open enrollment period for employee benefits, thanks Sarah Thompson for her social media efforts, and announces a product launch event on July 15th. David Rodriguez is acknowledged for his contributions to R&D. The document emphasizes the importance of confidentiality.",
  "contact_info": [
    {
      "name": "John Doe",
      "contact_info": {
        "phone": "",
        "email": "john.doe@example.com"
      }
    },
    {
      "name": "Michael Johnson",
      "contact_info": {
        "phone": "418-492-3850",
        "email": "michael.johnson@example.com"
      }
    },
    {
      "name": "Sarah Thompson",
      "contact_info": {
        "phone": "415-555-1234",
        "email": ""
      }
    }
  ],
  "questions_and_answers": [
    {
      "question": "What is the purpose of this document?",
      "answer": "The purpose of this document is to provide important updates and discuss various topics that require the team's attention."
    },
    {
      "question": "Who is commended for enhancing the company's network security?",
      "answer": "John Doe from the IT department is commended for enhancing the company's network security."
    }
  ]
}
```

## Getting Started
`pip install doctran`

```python
from doctran import Doctran

doctran = Doctran(openai_api_key=OPENAI_API_KEY)
document = doctran.parse(content="your_content_as_string")
```

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

