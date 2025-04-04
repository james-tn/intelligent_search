import logging  
import os  
import json  
import requests  
import azure.functions as func  
from azure.core.credentials import AzureKeyCredential  
from azure.identity import DefaultAzureCredential  
from azure.search.documents import SearchClient  
def get_embedding(text):  
    azure_openai_embedding_deployment = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT")  
    azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")  
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  
    azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")  
    url = f"{azure_openai_endpoint}/openai/deployments/{azure_openai_embedding_deployment}/embeddings?api-version={azure_openai_api_version}"  
    headers = {"Content-Type": "application/json", "api-key": azure_openai_key}  
    response = requests.post(url, headers=headers, json={"input": text})  
    response.raise_for_status()  
    return response.json()["data"][0]["embedding"]  
  
def main(myblob: func.InputStream) -> None:  
    logging.info(f"Triggered UploadDocuments for blob: {myblob.name}")  
    msg_data = json.loads(myblob.read())  
    # Compute vector embeddings for the email subject and body  
    msg_data["subjectVector"] = get_embedding(msg_data.get("subject", ""))  
    msg_data["bodyVector"] = get_embedding(msg_data.get("body", ""))  

    # Prepare Azure Cognitive Search settings  
    endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]  
    index_name = os.getenv("AZURE_SEARCH_INDEX", "vectest")  
    admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY", "")  

    # Choose credential (Admin Key or Managed Identity)  
    credential = AzureKeyCredential(admin_key) if admin_key else DefaultAzureCredential()  
    logging.info("endpoint: %s", endpoint)
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)  
    results = search_client.upload_documents(documents=[msg_data])  
    logging.info(f"Documents uploaded successfully: {results}")  
