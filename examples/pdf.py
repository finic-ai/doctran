import asyncio
import json
import os
import io
import pdfplumber
import requests
from doctran import Doctran, Document
from doctran.doctran import DenoiseProperty

## Test with pdf file

def parse_pdf_to_text(file_url):
    response = requests.get(file_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Convert the content bytes into a file-like object
        file_bytes = io.BytesIO(response.content)

        # Open the PDF from the content of the response
        with pdfplumber.open(file_bytes) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
            return text
    else:
        raise Exception(f"Failed to retrieve the file {file_url}")


async def run():
    doctran = Doctran(openai_api_key=os.environ['OPENAI_API_KEY'], openai_model="gpt-3.5-turbo-16k-0613")

    # Parse PDF to text
    pdf_text = ""
    # TODO: Parse with Document parse
    try:
        file_url = "https://drive.google.com/uc?export=download&id=111LlWDw6Rzht_DRMbiVGNIYhF_dEWniS"
        pdf_text = parse_pdf_to_text(file_url)
    except Exception as e:
        print(f"An error occurred when trying to read file: {e}")

    # Create Doctran Document
    document = doctran.parse(content=pdf_text, content_type="text")

    property = DenoiseProperty(
            name="only_relevant_data", 
            description="Only include text related to the given relevant topics",
            type="string",
            properties={
                "Investment": {
                    "type": "string",
                    "description": "A relevant topic in this text",
                },
                "Acquisitions": {
                    "type": "string",
                    "description": "A relevant topic in this text",
                },
            },
            required=True
        )
    
    
    document = await doctran.denoise(document=document, property=property)

    print("\nDocument Content: " + pdf_text[:250] + "...")
    print(f"\n👴 Denoised Content:\n \033[1m {document.transformed_content} \033[0m")

asyncio.run(run())