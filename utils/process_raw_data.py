import os  
import extract_msg  
import json  
from openai import AzureOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import re
import uuid
load_dotenv()
class ParsedEmail(BaseModel):
    summary: str
    category: str


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

def get_openai_embedding(text):

    """Get the OpenAI embedding for the given text."""
    try:
        response = embedding_client.embeddings.create(
            model=azure_openai_embedding_deployment,
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting OpenAI embedding: {e}")
        return None
def get_openai_chat_response(messages, json_output=False):
    """Get the OpenAI chat response for the given messages."""
    try:
        if json_output:

            response = chat_completion_client.beta.chat.completions.parse(
                model=chat_model,
                messages=messages,
                max_tokens=500,
                response_format=ParsedEmail if json_output else None,

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
    # Regular expression to match an email pattern  
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'  
    match = re.search(email_pattern, text)  
    if match:  
        return match.group()  
    return None  

def format_datetime(dt):  
    """Format datetime to the OData V4 format."""  
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'  

def process_msg_file(msg_file_path, email_id):  
    try:  
        # Open the .msg file  
        msg = extract_msg.Message(msg_file_path)  
        msg_data = {}  
  
        # Extract basic email details  
        msg_data["id"] = email_id  
        msg_data["from"] = extract_email(msg.sender) or ""  
        msg_data["to_list"] = ",".join([recipient.email for recipient in msg.recipients]) if msg.recipients else ""
        msg_data["cc_list"] = str(msg.cc) if msg.cc else ""  
        msg_data["subject"] = msg.subject or ""  
        msg_data["important"] = msg.importance # Check if the email is marked as important  
  
        # Extract body & category 
        messages = [{"role": "user", "content": f"summarize the email content and categorize the email into one of the following ['Urgent: Emails that require immediate attention or action.Projects: Emails related to specific projects or tasks, including updates, progress reports, and deliverables.Meetings: Emails about scheduling, agendas, and minutes of meetings.Internal: Emails from within the organization, such as announcements, newsletters, and internal memos.External: Emails from clients, partners, suppliers, or other external parties.Admin: Emails related to administrative matters, human resources, policies, and compliance.']. The output should be in JSON format with 'summary' and 'category' as keys:\n{msg.htmlBody}"}]    
        output = get_openai_chat_response(messages, json_output=True)
        msg_data["body"] = output.summary
        msg_data["category"] =  output.category 
  
        # Extract attachments  
        attachment_names = []
        for attachment in msg.attachments:  
            attachment_names.append(attachment.longFilename or attachment.shortFilename or "Unknown")  
  
        msg_data["attachment_names"] = ",".join(attachment_names)
  
  
        # Time details  
        msg_data["received_time"] = format_datetime(msg.date) if msg.date else None  
        msg_data["sent_time"] = format_datetime(msg.date) if msg.date else None  
  
        # Size of the email (approximated from the file size)  
        msg_data["size"] = os.path.getsize(msg_file_path)  
  
        # Close the message after processing  
        msg.close()  
  
        return msg_data  
    except Exception as e:  
        print(f"Error processing file {msg_file_path}: {e}")  
        return None  
  
def extract_emails_from_folder(folder_path):  
    extracted_emails = []  
    email_counter = 100000  # Start numbering emails from 100000  
  
    # Iterate through files in the folder  
    for file_name in os.listdir(folder_path):  
        if file_name.endswith(".msg"):  
            email_counter += 1  
            msg_file_path = os.path.join(folder_path, file_name)  
            email_id = str(uuid.uuid4())
            msg_data = process_msg_file(msg_file_path, email_id)  
            if msg_data:  
                extracted_emails.append(msg_data)  
  
    return extracted_emails  
  
def save_to_json(data, output_file):  
    with open(output_file, "w", encoding="utf-8") as f:  
        json.dump(data, f, indent=4)  
  
def main():  
    folder_path = "./raw_data"  
    output_file = "extracted_emails.json"  
  
    # Extract emails and save to JSON  
    extracted_emails = extract_emails_from_folder(folder_path)  
    save_to_json(extracted_emails, output_file)  
    print(f"Extracted {len(extracted_emails)} emails and saved to {output_file}")  
  
if __name__ == "__main__":  
    main()  