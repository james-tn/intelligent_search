# Import needed modules  
import os  
from azure.cosmos import CosmosClient, PartitionKey  
from azure.identity import DefaultAzureCredential  
from dotenv import load_dotenv  
  
# Load environment variables from .env file  
load_dotenv()  
  
# Set your AAD credentials for Cosmos DB authentication  
aad_client_id = os.getenv("AAD_CLIENT_ID")  
aad_client_secret = os.getenv("AAD_CLIENT_SECRET")  
aad_tenant_id = os.getenv("AAD_TENANT_ID")  
  
os.environ["AZURE_CLIENT_ID"] = aad_client_id  
os.environ["AZURE_CLIENT_SECRET"] = aad_client_secret  
os.environ["AZURE_TENANT_ID"] = aad_tenant_id  
  
# Configure your Cosmos DB connection settings  
cosmos_uri = os.getenv("COSMOS_URI")  
cosmos_db_name = os.getenv("COSMOS_DB_NAME", "vectordb")  
container_name = os.getenv("COSMOS_CONTAINER_NAME", "vectortest_hybridsearch")  
  
# Create the CosmosClient using DefaultAzureCredential  
credential = DefaultAzureCredential()  
cosmos_client = CosmosClient(cosmos_uri, credential=credential)  
  
# Create the database if it does not exist  
database = cosmos_client.create_database_if_not_exists(id=cosmos_db_name)  
  
# Check if the container exists and drop it if it does  
try:  
    container = database.get_container_client(container_name)  
    print(f"Container '{container_name}' exists. Dropping...")  
    container.delete_container()  
    print(f"Container '{container_name}' dropped successfully.")  
except Exception as e:  
    print(f"Container '{container_name}' does not exist or cannot be accessed. Proceeding to create...")  
  
# Define the vector embedding policy for the two vector fields  
vector_embedding_policy = {  
    "vectorEmbeddings": [  
        {  
            "path": "/subjectVector",  
            "dataType": "float32",  
            "distanceFunction": "cosine",  
            "dimensions": 1536  
        },  
        {  
            "path": "/bodyVector",  
            "dataType": "float32",  
            "distanceFunction": "cosine",  
            "dimensions": 1536  
        }  
    ]  
}  
  
# Define the indexing policy  
indexing_policy = {  
    "indexingMode": "consistent",  
    "automatic": True,  
    "includedPaths": [{"path": "/"}],  
    "excludedPaths": [  
        {"path": "/_etag/?"},  
        {"path": "/subjectVector/*"},  
        {"path": "/bodyVector/*"}  
    ],  
    "vectorIndexes": [  
        {"path": "/subjectVector", "type": "quantizedFlat"},  
        {"path": "/bodyVector", "type": "quantizedFlat"}  
    ]  
}  
  
# Define the full-text policy  
full_text_policy = {  
    "defaultLanguage": "en-US",  
    "fullTextPaths": [  
        {"path": "/subject", "language": "en-US"},  
        {"path": "/body", "language": "en-US"},  
        {"path": "/to_list", "language": "en-US"},  
        {"path": "/cc_list", "language": "en-US"},  
        {"path": "/category", "language": "en-US"},  
        {"path": "/attachment_names", "language": "en-US"}  
    ]  
}  
  
# Create the container  
container = database.create_container(  
    id=container_name,  
    partition_key=PartitionKey(path="/id"),  
    vector_embedding_policy=vector_embedding_policy,  
    indexing_policy=indexing_policy,  
    full_text_policy=full_text_policy,  
)  
  
print(f"Container '{container_name}' created with vector indexing on 'subjectVector' and 'bodyVector'.")  