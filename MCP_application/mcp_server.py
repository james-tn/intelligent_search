# mcp.py (Cosmos Email Search MCP Service)

from semantic_kernel.functions.kernel_function import KernelFunction
from semantic_kernel.functions.kernel_plugin import KernelPlugin
from fastmcp import FastMCP 
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from dotenv import load_dotenv
from datetime import datetime
import os
import requests
import json
import math
from azure.cosmos import CosmosClient
from azure.identity import ClientSecretCredential







# ─────────────────── Load ENV and Initialize ───────────────────
load_dotenv()



from datetime import datetime

def get_current_time():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

system_prompt = (
    f"Today is {get_current_time()}. You are an expert query translator for a Cosmos DB vector search engine. "
    "Convert the conversation of natural language email search queries into a complete query JSON object. "
    "The JSON object must contain exactly the following keys: 'search_text' and 'filter'.\n\n"
    "Additional guidelines:\n"
    "1. The 'search_text' key should include useful free-text search terms extracted from subject, body, and attachments.\n"
    "2. The 'filter' key should include any structured filter expressions. When filtering on dates, use ISO 8601 format. "
    "For example, if the user says 'before June 13 2025', output a filter like: c.sent_time < '2025-06-13T00:00:00Z'.\n"
    "For sender emails or other string properties, use equality with proper quoting.\n"
    "3. Output only the JSON object – do not include any additional text or commentary.\n\n"
    "Document schema:\n"
    " - id (string)\n"
    " - from (string): Sender Email\n"
    " - to_list (string): Recipient list\n"
    " - cc_list (string): CC list\n"
    " - subject (string)\n"
    " - important (int)\n"
    " - body (string)\n"
    " - category (string)\n"
    " - attachment_names (collection of string)\n"
    " - received_time (DateTimeOffset) e.g., '2025-06-13T00:00:00Z'\n"
    " - sent_time (DateTimeOffset) e.g., '2025-06-13T00:00:00Z'\n"
)

mcp = FastMCP(
    name="Email Search Tools",
    instructions=system_prompt,
)


# ─────────────────── Azure OpenAI & Cosmos Config ───────────────────
COSMOS_URI = os.getenv("COSMOS_URI")
COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME")
COSMOS_CONTAINER_NAME = os.getenv("COSMOS_CONTAINER_NAME")

AZURE_CLIENT_ID = os.getenv("AAD_CLIENT_ID")
AZURE_CLIENT_SECRET = os.getenv("AAD_CLIENT_SECRET")
AZURE_TENANT_ID = os.getenv("AAD_TENANT_ID")

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT")

credential = ClientSecretCredential(
    tenant_id=AZURE_TENANT_ID,
    client_id=AZURE_CLIENT_ID,
    client_secret=AZURE_CLIENT_SECRET,
)

cosmos_client = CosmosClient(COSMOS_URI, credential=credential)
container = cosmos_client.get_database_client(COSMOS_DB_NAME).get_container_client(COSMOS_CONTAINER_NAME)

# ─────────────────── Models ───────────────────
class SearchQuery(BaseModel):
    search_text: str
    filter: Optional[str] = ""

class EmailResult(BaseModel):
    id: str
    sender: str
    subject: str
    sent_time: str
    body_preview: str

# ─────────────────── Utility ───────────────────
def get_embedding(text: str) -> List[float]:
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{EMBEDDING_DEPLOYMENT}/embeddings?api-version={AZURE_OPENAI_API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}
    response = requests.post(url, headers=headers, json={"input": text})
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

# ─────────────────── Tool Endpoint ───────────────────
@mcp.tool(description="Run vector + full-text query over Cosmos DB email container")
def run_cosmos_query(params: SearchQuery) -> List[EmailResult]:
    embedding = get_embedding(params.search_text)
    embedding_literal = "[" + ",".join(str(x) for x in embedding) + "]"
    safe_text = params.search_text.replace("'", "\\'")
    where_clause = f"WHERE {params.filter}" if params.filter else ""

    query = f"""
    SELECT TOP 20 c.id, c["from"], c.subject, c.sent_time, c.body
    FROM c
    {where_clause}
    ORDER BY RANK RRF(
        VectorDistance(c.bodyVector, {embedding_literal}),
        FullTextScore(c.body, '{safe_text}')
    )
    """
    query = query.replace("c.from", "c[\"from\"]")  # keyword escaping

    results = list(container.query_items(query=query, enable_cross_partition_query=True))

    top_results = []
    for item in results:
        body = item.get("body", "")
        body_preview = body[:200] + ("..." if len(body) > 200 else "")
        top_results.append(
            EmailResult(
                id=item.get("id"),
                sender=item.get("from", "N/A"),
                subject=item.get("subject", ""),
                sent_time=item.get("sent_time", ""),
                body_preview=body_preview,
            )
        )
    return top_results

# ─────────────────── Run as SSE Server ───────────────────
if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_sse_async(host="0.0.0.0", port=8000))