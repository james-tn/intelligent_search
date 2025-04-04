import logging  
import os  
import json  
import uuid  
import tempfile  
import azure.functions as func  

#!/usr/bin/env python  
import os  
import extract_msg  
import re  
import uuid  
from openai import AzureOpenAI  
from pydantic import BaseModel  
from dotenv import load_dotenv  
  
  
class ParsedEmail(BaseModel):  
    summary: str  
    category: str  
  
# Read OpenAI settings from environment  
azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT")  
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")  
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")  
chat_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")  
  
embedding_client = AzureOpenAI(  
    azure_deployment=azure_openai_embedding_deployment,  
    api_version=azure_openai_api_version,  
    azure_endpoint=azure_openai_endpoint,  
    api_key=azure_openai_key,  
)  
  
chat_completion_client = AzureOpenAI(  
    api_key=azure_openai_key,  
    azure_endpoint=azure_openai_endpoint,  
    api_version=azure_openai_api_version,  
)  
  
def get_openai_chat_response(messages, json_output=False):  
    try:  
        if json_output:  
            response = chat_completion_client.beta.chat.completions.parse(  
                model=chat_model,  
                messages=messages,  
                max_tokens=500,  
                response_format=ParsedEmail,  
            )  
            return response.choices[0].message.parsed  
        else:  
            response = chat_completion_client.chat.completions.create(  
                model=chat_model,  
                messages=messages,  
                max_tokens=500,  
            )  
            return response.choices[0].message.content  
    except Exception as e:  
        print(f"Error getting OpenAI chat response: {e}")  
        return None  
  
def extract_email(text):  
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'  
    match = re.search(email_pattern, text)  
    return match.group() if match else ""  
  
def format_datetime(dt):  
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'  
  
def process_msg_file(msg_file_path, email_id):  
    try:  
        msg = extract_msg.Message(msg_file_path)  
        msg_data = {}  
        msg_data["id"] = email_id  
        msg_data["from"] = extract_email(msg.sender) if msg.sender else ""  
        msg_data["to_list"] = ",".join([r.email for r in msg.recipients if hasattr(r, 'email')]) if msg.recipients else ""  
        msg_data["cc_list"] = str(msg.cc) if msg.cc else ""  
        msg_data["subject"] = msg.subject or ""  
        msg_data["important"] = msg.importance  
  
        # Use OpenAI chat to summarize and categorize the email.  
        messages = [  
            {  
                "role": "user",  
                "content": (  
                    "Summarize the email content and categorize the email into one of the following "  
                    "['Urgent', 'Projects', 'Meetings', 'Internal', 'External', 'Admin']. "  
                    "The output should be in JSON format with 'summary' and 'category' as keys:\n"  
                    f"{msg.htmlBody}"  
                )  
            }  
        ]  
        output = get_openai_chat_response(messages, json_output=True)  
        if output:  
            msg_data["body"] = output.summary  
            msg_data["category"] = output.category  
        else:  
            msg_data["body"] = ""  
            msg_data["category"] = ""  
  
        # Process attachments  
        attachment_names = []  
        if msg.attachments:  
            for attachment in msg.attachments:  
                name = attachment.longFilename or attachment.shortFilename or "Unknown"  
                attachment_names.append(name)  
        msg_data["attachment_names"] = ",".join(attachment_names)  
  
        if msg.date:  
            msg_data["received_time"] = format_datetime(msg.date)  
            msg_data["sent_time"] = format_datetime(msg.date)  
        else:  
            msg_data["received_time"] = None  
            msg_data["sent_time"] = None  
  
        msg_data["size"] = os.path.getsize(msg_file_path)  
        msg.close()  
        return msg_data  
    except Exception as e:  
        print(f"Error processing file {msg_file_path}: {e}")  
        return None  

def main(myblob: func.InputStream, outputBlob: func.Out[str]) -> None:  
    logging.info(f"Processing blob: {myblob.name}, Size: {myblob.length} bytes")  
    temp_file = None  
    try:  
        # Write the incoming blob stream to a temporary file  
        with tempfile.NamedTemporaryFile(delete=False, suffix=".msg") as tmp:  
            tmp.write(myblob.read())  
            temp_file = tmp.name  
  
        email_id = str(uuid.uuid4())  
        msg_data = process_msg_file(temp_file, email_id)  
        if msg_data:  
            # Write the processed data as JSON to the output binding  
            outputBlob.set(json.dumps(msg_data))  
            logging.info(f"Processed email from blob: {myblob.name}")  
        else:  
            logging.error("Failed to process the email.")  
    except Exception as e:  
        logging.error(f"Exception in processing blob: {e}")  
    finally:  
        if temp_file and os.path.exists(temp_file):  
            os.remove(temp_file)  