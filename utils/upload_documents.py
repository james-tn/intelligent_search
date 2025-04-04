from azure.identity import DefaultAzureCredential  
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient  
import os  
import requests  
import json  
import dotenv  
  
dotenv.load_dotenv()  
  
# Configuration  
endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]  
credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY", "")) if len(os.getenv("AZURE_SEARCH_ADMIN_KEY", "")) > 0 else DefaultAzureCredential()  
index_name = os.getenv("AZURE_SEARCH_INDEX", "vectest")  
  
azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT")  
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")  
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")  
  
# Load data from extracted_emails.json  
with open("extracted_emails.json", "r") as file:  
    data = json.load(file)  
  
# Helper function to compute embeddings using Azure OpenAI  
def get_embedding(text):  
    url = f"{azure_openai_endpoint}/openai/deployments/{azure_openai_embedding_deployment}/embeddings?api-version={azure_openai_api_version}"  
    headers = {  
        "Content-Type": "application/json",  
        "api-key": azure_openai_key  
    }  
    response = requests.post(url, headers=headers, json={"input": text})  
    response.raise_for_status()  
    return response.json()["data"][0]["embedding"]  
  
# Compute vectors and add data to Azure Cognitive Search  
search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)  
  
for item in data:  
    item["subjectVector"] = get_embedding(item["subject"])  
    item["bodyVector"] = get_embedding(item["body"])  
  
search_client.upload_documents(documents=data)  
print("Documents uploaded successfully.")  