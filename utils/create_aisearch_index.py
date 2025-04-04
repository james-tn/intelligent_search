from azure.identity import DefaultAzureCredential  
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.indexes.models import (  
    SimpleField,  
    SearchFieldDataType,  
    SearchableField,  
    SearchField,  
    VectorSearch,  
    HnswAlgorithmConfiguration,  
    VectorSearchProfile,  
    SemanticConfiguration,  
    SemanticPrioritizedFields,  
    SemanticField,  
    SemanticSearch,  
    SearchIndex,  
    AzureOpenAIVectorizer,  
    AzureOpenAIVectorizerParameters,  
)  
import os  
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
azure_openai_embedding_dimensions = 1536  # Set the embedding dimensions for Azure OpenAI embeddings  
  
# Create a search index client  
index_client = SearchIndexClient(endpoint=endpoint, credential=credential)  
  
# Define the fields for the search index  
fields = [  
    SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),  
    SimpleField(name="from", type=SearchFieldDataType.String, filterable=True, facetable=True),  # Sender  
    SearchableField(name="to_list", type=SearchFieldDataType.String),  # Recipient list  
    SearchableField(name="cc_list", type=SearchFieldDataType.String),  # CC list  
    SearchableField(name="subject", type=SearchFieldDataType.String),  # Subject  
    SimpleField(name="important", type=SearchFieldDataType.Int32, filterable=True, facetable=True),  # Importance  
    SearchableField(name="body", type=SearchFieldDataType.String),  # Body  
    SearchableField(name="category", type=SearchFieldDataType.String, filterable=True),  # Category  
    SearchableField(name="attachment_names", type=SearchFieldDataType.Collection(SearchFieldDataType.String)),  # Attachments  
    SimpleField(name="received_time", type=SearchFieldDataType.DateTimeOffset, filterable=True),  # Received time  
    SimpleField(name="sent_time", type=SearchFieldDataType.DateTimeOffset, filterable=True),  # Sent time  
    SimpleField(name="size", type=SearchFieldDataType.Int32, filterable=True, facetable=True),  # Size  
    SearchField(  
        name="subjectVector",  
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),  
        searchable=True,  
        vector_search_dimensions=azure_openai_embedding_dimensions,  
        vector_search_profile_name="myHnswProfile"  
    ),  
    SearchField(  
        name="bodyVector",  
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),  
        searchable=True,  
        vector_search_dimensions=azure_openai_embedding_dimensions,  
        vector_search_profile_name="myHnswProfile"  
    )  
]  
  
# Configure the vector search settings  
vector_search = VectorSearch(  
    algorithms=[  
        HnswAlgorithmConfiguration(name="myHnsw")  # HNSW algorithm configuration  
    ],  
    profiles=[  
        VectorSearchProfile(  
            name="myHnswProfile",  
            algorithm_configuration_name="myHnsw",  
            vectorizer_name="myVectorizer"  
        )  
    ],  
    vectorizers=[  
        AzureOpenAIVectorizer(  
            vectorizer_name="myVectorizer",  
            parameters=AzureOpenAIVectorizerParameters(  
                resource_url=azure_openai_endpoint,  
                deployment_name=azure_openai_embedding_deployment,  
                model_name="text-embedding-ada-002",  # Replace with your model name if different  
                api_key=azure_openai_key  
            )  
        )  
    ]  
)  
  
# Configure the semantic search settings  
semantic_config = SemanticConfiguration(  
    name="my-semantic-config",  
    prioritized_fields=SemanticPrioritizedFields(  
        title_field=SemanticField(field_name="subject"),  # Use "subject" as the title field  
        keywords_fields=[SemanticField(field_name="category")],  # Use "category" as keywords  
        content_fields=[SemanticField(field_name="body")]  # Use "body" as the main content field  
    )  
)  
  
semantic_search = SemanticSearch(configurations=[semantic_config])  
  
# Check if the index exists, drop it if it does  
try:  
    index_client.get_index(index_name)  
    print(f"Index '{index_name}' exists. Dropping it...")  
    index_client.delete_index(index_name)  
    print(f"Index '{index_name}' deleted.")  
except Exception as e:  
    print(f"Index '{index_name}' does not exist or could not be retrieved. Proceeding to create a new index.")  
  
# Create the search index with vector search and semantic search settings  
index = SearchIndex(  
    name=index_name,  
    fields=fields,  
    vector_search=vector_search,  
    semantic_search=semantic_search  
)  
  
result = index_client.create_or_update_index(index)  
print(f'{result.name} created')  