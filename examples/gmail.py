import os
import json
import asyncio
import mailbox
import bs4
from email import message_from_string, message_from_bytes
from email.message import Message
from doctran import Doctran, Document, ExtractProperty

## Test with gmail mbox file

def get_html_text(html):
    try:
        return bs4.BeautifulSoup(html, 'lxml').body.get_text(' ', strip=True)
    except AttributeError: # message contents empty
        return None
    
class GmailMboxMessage():
    def __init__(self, email_data):
        if not isinstance(email_data, mailbox.mboxMessage):
            raise TypeError('Variable must be type mailbox.mboxMessage')
        self.email_data = email_data

    def parse_email(self):
        email_date = self.email_data['Date']
        email_from = self.email_data['From']
        email_to = self.email_data['To']
        email_cc = self.email_data['Cc']
        email_subject = self.email_data['Subject']
        email_text = self.read_email_payload()
        return {
            'date': email_date,
            'from': email_from,
            'to': email_to,
            'cc': email_cc,
            'subject': email_subject,
            'text': email_text
        }

    def read_email_payload(self):
        if self.email_data.is_multipart():
            email_payload = self.email_data.get_payload()
            email_messages = list(self._get_email_messages(email_payload))
        else:
            email_messages = [self.email_data.get_payload(decode=True)]
        return [self._read_email_text(msg) for msg in email_messages]

    def _get_email_messages(self, email_payload: list[mailbox.mboxMessage]):
        for msg in email_payload:
            if isinstance(msg, (list,tuple)):
                for submsg in self._get_email_messages(msg):
                    yield submsg
            elif msg.is_multipart():
                for submsg in self._get_email_messages(msg.get_payload()):
                    yield submsg
            elif msg.get_content_type() in ('text/html'):
                yield msg

    def _read_email_text(self, msg: mailbox.mboxMessage):
        if isinstance(msg, str):
            msg = message_from_string(msg)
        elif isinstance(msg, bytes):
            msg = message_from_bytes(msg)
        content_type = msg.get_content_type()
        encoding = msg.get('Content-Transfer-Encoding', 'NA')
        if 'text/plain' in content_type and 'base64' not in encoding:
            msg_text = msg.get_payload(decode=True)
        elif 'text/html' in content_type and 'base64' not in encoding:
            msg_text = get_html_text(msg.get_payload(decode=True))
        elif content_type == 'NA':
            msg_text = get_html_text(msg)
        else:
            msg_text = None
        return (content_type, encoding, msg_text)

async def test_extract(doctran: Doctran, document: Document, parsed_email: dict):
    properties = [
        ExtractProperty(
            name="millenial_or_boomer", 
            description="A prediction of whether this email was written by a millenial or boomer",
            type="string",
            enum=["millenial", "boomer"],
            required=True
        ),
        ExtractProperty(
            name="as_gen_z", 
            description="The email rewritten and summarized as if it were from a a Gen Z person",
            type="string",
            required=True
        )
    ]
    document = await doctran.extract(document=document, properties=properties)
    print("\nEmail Subject: " + parsed_email.get('subject'))
    print("Email Body: " + parsed_email.get('text')[0][2][:250] + "...")
    print("\n👴 Written by boomer or millenial:\n" + '\033[1m' + json.dumps(document.extracted_properties["millenial_or_boomer"], ensure_ascii=False) + '\033[0m')
    print("\n✨ Rewritten as Gen Z:\n" + '\033[1m' + json.dumps(document.extracted_properties["as_gen_z"], ensure_ascii=False) + '\033[0m')

async def run():
    mbox_obj = mailbox.mbox('/Users/jasonfan/Documents/code/Takeout/Mail/All mail Including Spam and Trash.mbox')

    num_entries = len(mbox_obj)
    print("Loaded {num_entries} entries from mbox file".format(num_entries=num_entries))

    doctran = Doctran(openai_api_key=os.environ['OPENAI_API_KEY'], openai_model="gpt-4-0613")

    for i in range(700, 710):
        email_obj = mbox_obj[i]
        email_data = GmailMboxMessage(email_obj)
        parsed_email = email_data.parse_email()
        # Approximate check to ensure we're not sending emails > 8000 tokens
        if len(str(parsed_email)) > 25000:
            continue
        document = doctran.parse(content=str(parsed_email), content_type="text")

        # Extract parameters
        # await test_extract(doctran=doctran, document=document, parsed_email=parsed_email)

        # Redact PII
        # document = doctran.redact(document=document, entities=["PERSON", "EMAIL_ADDRESS", "LOCATION"])
        # print(document.transformed_content)

        # Summarize context
        document = await document.redact(entities=["PERSON", "EMAIL_ADDRESS", "LOCATION"]).summarize(token_limit=100).execute()
        print(document.transformed_content)

asyncio.run(run())