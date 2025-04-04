# Intelligent Email Search with Cosmos DB and Azure OpenAI  
  
This project demonstrates an intelligent search application that uses **Azure AI search**,  **Cosmos DB vector search** and **Azure OpenAI ** to enable advanced email search capabilities. The application provides a **Streamlit-based UI** for interacting with the search engine and includes an **Azure Function** for deploying server-side functionality.  
  
## Features  
- **Azure AI  Search**: Support Hybrid Search & Semantic Reranker to accurately retrieve result.  
- **Cosmos DB Vector Search**: Combines vector similarity (e.g., `VectorDistance`) and full-text search (`FullTextScore`) to rank results.  
- **Azure OpenAI Integration**: Computes embeddings for email content (`subject` and `body`) and generates search queries using OpenAI's chat completions.  
- **Streamlit UI**: Provides an interactive user interface for natural language queries and displaying email search results.  
- **Azure Function Deployment**: Supports server-side functionality for embedding computation and query execution.  
  
---  
  
## Prerequisites  
  
Before running the application, ensure you have the following installed:  
  
1. **Python** (version 3.8 or higher)  
2. **Azure CLI** (for managing Azure resources)  
3. **Streamlit** (installed via `pip`)  
4. **Azure Function Core Tools** (for deploying Azure Functions)  
  
You also need:  
  
- A **Cosmos DB account** with vector search enabled.  
- **Azure OpenAI access** (API key, endpoint, and deployment).  
- A `.env` file containing your environment variables (see below).  
  
---  
  
## Environment Variables  
  
Create a `.env` file in the project root directory with the following keys:  
  
```dotenv  
# Azure OpenAI Configuration  
AZURE_OPENAI_API_KEY=<your-openai-api-key>  
AZURE_OPENAI_ENDPOINT=https://<your-openai-endpoint>  
AZURE_OPENAI_API_VERSION=2023-03-15-preview  
AZURE_OPENAI_CHAT_DEPLOYMENT=<your-chat-deployment-name>  
AZURE_OPENAI_EMB_DEPLOYMENT=<your-embedding-deployment-name>  
  
# Cosmos DB Configuration  
COSMOS_URI=https://<your-cosmos-db-account>.documents.azure.com:443/  
COSMOS_DB_NAME=<your-database-name>  
COSMOS_CONTAINER_NAME=<your-container-name>  
  
# Azure AD Credentials (for Cosmos DB authentication)  
AZURE_CLIENT_ID=<your-client-id>  
AZURE_CLIENT_SECRET=<your-client-secret>  
AZURE_TENANT_ID=<your-tenant-id>  