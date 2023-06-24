from enum import Enum

class Prompts(Enum):
    compress = """Summarize this document in fewer than {token_limit} tokens.

Document:
{document}"""

    denoise = """Summarize this document but exclude all information that's not related to one of the topics provided.

Topics:
{topics}

Document:
{document}"""