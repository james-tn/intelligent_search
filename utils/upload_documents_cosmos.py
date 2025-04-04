# Import required modules  
import os  
import json  
import uuid  
import requests  
from dotenv import load_dotenv  
from azure.cosmos import CosmosClient, PartitionKey  
from azure.identity import DefaultAzureCredential  
  
# Load environment variables from .env file  
load_dotenv()  
  
# Set your AAD credentials for Cosmos DB authentication  
aad_client_id = os.getenv("AAD_CLIENT_ID")  
aad_client_secret = os.getenv("AAD_CLIENT_SECRET")  
aad_tenant_id = os.getenv("AAD_TENANT_ID")  
  
os.environ["AZURE_CLIENT_ID"] = aad_client_id  
os.environ["AZURE_CLIENT_SECRET"] = aad_client_secret  
os.environ["AZURE_TENANT_ID"] = aad_tenant_id  
  
# Cosmos DB connection settings  
# Configure your Cosmos DB connection settings  
cosmos_uri = os.getenv("COSMOS_URI")  
cosmos_db_name = os.getenv("COSMOS_DB_NAME", "vectordb")  
container_name = os.getenv("COSMOS_CONTAINER_NAME", "vectortest_hybridsearch")  
  
# Create the Cosmos DB client using DefaultAzureCredential  
credential = DefaultAzureCredential()  
cosmos_client = CosmosClient(cosmos_uri, credential=credential)  
cosmos_db_client = cosmos_client.get_database_client(cosmos_db_name)  
cosmos_container_client = cosmos_db_client.get_container_client(container_name)  
  
# Azure OpenAI configuration  
azure_openai_embedding_deployment = os.environ["AZURE_OPENAI_EMB_DEPLOYMENT"]  
azure_openai_key = os.environ["AZURE_OPENAI_API_KEY"]  
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]  
azure_openai_api_version = os.environ["AZURE_OPENAI_API_VERSION"]  
  
# Helper function to compute embeddings using Azure OpenAI  
def get_embedding(text):  
    url = (  
        f"{azure_openai_endpoint}/openai/deployments/"  
        f"{azure_openai_embedding_deployment}/embeddings?api-version="  
        f"{azure_openai_api_version}"  
    )  
    headers = {  
        "Content-Type": "application/json",  
        "api-key": azure_openai_key  
    }  
    response = requests.post(url, headers=headers, json={"input": text})  
    response.raise_for_status()  
    return response.json()["data"][0]["embedding"]  
  
# Load data from the extracted emails JSON file  
with open("extracted_emails.json", "r") as file:  
    emails = json.load(file)  
  
# Ingest each document into Cosmos DB without parallelization  
for email in emails:  
    # Compute the embeddings for the "subject" and "body" fields  
    email["subjectVector"] = get_embedding(email["subject"])  
    email["bodyVector"] = get_embedding(email["body"])  
  
    # Ensure each document has an "id" property (required by Cosmos DB)  
    if "id" not in email or not email["id"]:  
        email["id"] = str(uuid.uuid4())  
  
    # Upsert the item into Cosmos DB  
    cosmos_container_client.upsert_item(email)  
    print(f"Upserted document with id: {email['id']}")  
  
print("Documents uploaded successfully to Cosmos DB.")  