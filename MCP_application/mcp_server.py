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
mcp = FastMCP(
    name="Email Search Tools",
    instructions=(
        "You are an intelligent agent that assists users in searching emails stored in Cosmos DB. "
        "You must first convert the natural language into a JSON query with keys 'search_text' and 'filter', "
        "then call the tool `run_cosmos_query` using that JSON to get the results."
    ),
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