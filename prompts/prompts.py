from enum import Enum

COMPRESS_PROMPT = """Summarize this document, preserving as much information as possible

Document:
{document}"""

DENOISE_PROMPT = """Summarize this document but exclude all information that's not related to one of the topics provided.

Topics:
{topics}

Document:
{document}"""