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
import time
from azure.cosmos import CosmosClient
from azure.identity import ClientSecretCredential

# ─────────────────── Load ENV and Initialize ───────────────────
load_dotenv()

mcp = FastMCP(
    name="Email Search Tools",
    instructions=(
        "You are an intelligent agent that helps users search emails stored in Cosmos DB. "
        "When a user submits a natural language query, always first convert it into a JSON object with exactly two keys: 'search_text' (for semantic meaning) and 'filter' (for structured constraints) using the generate_search_query tool. "
        "Then, generate a vector embedding from the 'search_text' and use both the embedding and the filter to run a hybrid search in Cosmos DB. "
        "Return the most relevant emails based on both semantic similarity and filter criteria. "
        "Always follow this pipeline for every user query."
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
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

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

class ConversationMessage(BaseModel):
    role: str
    content: str

class ConversationHistory(BaseModel):
    messages: List[ConversationMessage]


# ─────────────────── Utility ───────────────────
def get_embedding(text: str) -> List[float]:
    print(f"[DEBUG] Generating embedding for search_text: '{text}'")
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{EMBEDDING_DEPLOYMENT}/embeddings?api-version={AZURE_OPENAI_API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}
    response = requests.post(url, headers=headers, json={"input": text})
    response.raise_for_status()
    print("[DEBUG] Embedding created successfully.")
    return response.json()["data"][0]["embedding"]

def call_azure_openai_chat(messages: List[Dict]) -> Optional[str]:
    # print(f"[DEBUG] Calling Azure OpenAI chat with messages: {messages}")
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{CHAT_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}
    body = {
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "stop": None,
    }
    try:
        resp = requests.post(url, headers=headers, json=body)
        resp.raise_for_status()
        completion = resp.json()
        return completion["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[ERROR] Azure OpenAI chat call failed: {e}")
        return None



def get_current_time() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


#@mcp.tool(description="Convert conversation to a JSON search query")
def generate_search_query(params: ConversationHistory) -> SearchQuery:
    system_prompt = f"""Today is {get_current_time()}. You are an expert query translator for a Cosmos DB vector search engine. 
Convert the conversation of natural language email search queries into a complete query JSON object. 
The JSON object must contain exactly the following keys: 'search_text' and 'filter'.

Additional guidelines:
1. The 'search_text' key should include useful free-text search terms extracted from subject, body, and attachments.
2. The 'filter' key should include any structured filter expressions. When filtering on dates, use ISO 8601 format. 
For example, if the user says 'before June 13 2025', output a filter like: c.sent_time < '2025-06-13T00:00:00Z'.
For sender emails or other string properties, use equality with proper quoting.
3. Output only the JSON object – do not include any additional text or commentary.

Document schema:
 - id (string)
 - from (string): Sender Email
 - to_list (string): Recipient list
 - cc_list (string): CC list
 - subject (string)
 - important (int)
 - body (string)
 - category (string)
 - attachment_names (collection of string)
 - received_time (DateTimeOffset) e.g., '2025-06-13T00:00:00Z'
 - sent_time (DateTimeOffset) e.g., '2025-06-13T00:00:00Z'
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend([{"role": m.role, "content": m.content} for m in params.messages])

    max_attempts = 3
    for attempt in range(max_attempts):
        print(f"[DEBUG] Attempt {attempt+1}: Sending messages to OpenAI for NLP-to-JSON conversion.")
        reply = call_azure_openai_chat(messages)
        if not reply:
            error_msg = "No response obtained from OpenAI."
        else:
            # print(f"[DEBUG] OpenAI raw reply: {reply}")
            try:
                query_json = json.loads(reply)
                print(f"[DEBUG] Parsed JSON: {query_json}")
                if "search_text" in query_json and "filter" in query_json:
                    print(f"[DEBUG] Extracted search_text: '{query_json['search_text']}', filter: '{query_json['filter']}'")
                    return SearchQuery(search_text=query_json["search_text"], filter=query_json["filter"])
                else:
                    error_msg = "The JSON is missing required keys 'search_text' or 'filter'."
            except Exception as e:
                error_msg = f"JSON parse error: {str(e)}. Full response was: {reply}"

        messages.append({"role": "assistant", "content": reply if reply else ""})
        messages.append({"role": "user", "content": f"Error: {error_msg} Please provide a valid JSON object with only the keys 'search_text' and 'filter'."})
        time.sleep(1)

    print("[ERROR] All attempts to generate search query failed.")
    return SearchQuery(search_text="", filter="")
    
#@mcp.tool(description="Run vector + full-text query over Cosmos DB email container")
def _run_cosmos_query_impl(params: ConversationHistory) -> List[EmailResult]:
    print(f"[DEBUG] User query received: {params.messages[-1].content if params.messages else '[No message]'}")
    search_query = generate_search_query(params)
    if not search_query:
        print("[ERROR] No search query generated.")
        return []

    # If search_text is empty but filter is present, do filter-only search
    if not search_query.search_text and search_query.filter:
        print("[DEBUG] No search_text found, running filter-only Cosmos DB query.")
        where_clause = f"WHERE {search_query.filter}" if search_query.filter else ""
        query = f"""
        SELECT TOP 20 c.id, c["from"], c.subject, c.sent_time, c.body
        FROM c
        {where_clause}
        ORDER BY c.sent_time DESC
        """
        query = query.replace("c.from", "c[\"from\"]")
        # print(f"[DEBUG] Final Cosmos DB SQL Query (filter-only):\n{query}")
        try:
            results = list(container.query_items(query=query, enable_cross_partition_query=True))
            print(f"[DEBUG] Cosmos DB returned {len(results)} results.")
        except Exception as e:
            print(f"[ERROR] Cosmos DB query failed: {e}")
            return []
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

    # If both are empty, return error
    if not search_query.search_text and not search_query.filter:
        print("[ERROR] No valid search_text or filter extracted from NLP-to-JSON step.")
        return []

    # Otherwise, proceed as before (hybrid search)
    print("[DEBUG] Proceeding with hybrid (vector + filter) search.")
    embedding = get_embedding(search_query.search_text)
    embedding_literal = "[" + ",".join(str(x) for x in embedding) + "]"
    safe_text = search_query.search_text.replace("'", "\\'")
    where_clause = f"WHERE {search_query.filter}" if search_query.filter else ""
    query = f"""
    SELECT TOP 20 c.id, c["from"], c.subject, c.sent_time, c.body
    FROM c
    {where_clause}
    ORDER BY RANK RRF(
        VectorDistance(c.bodyVector, {embedding_literal}),
        FullTextScore(c.body, '{safe_text}')
    )
    """
    query = query.replace("c.from", "c[\"from\"]")
    # print(f"[DEBUG] Final Cosmos DB SQL Query (hybrid):\n{query}")
    try:
        results = list(container.query_items(query=query, enable_cross_partition_query=True))
        print(f"[DEBUG] Cosmos DB returned {len(results)} results.")
    except Exception as e:
        print(f"[ERROR] Cosmos DB query failed: {e}")
        return []
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

@mcp.tool(description="Submit a natural language email search and get results from Cosmos DB")
def run_cosmos_query(params: ConversationHistory) -> List[EmailResult]:
    return _run_cosmos_query_impl(params)

# # ─────────────────── Test Block ───────────────────
# if __name__ == "__main__":
#     # Test block: Run a sample query directly
#     user_query = "show me emails before June 13 2025"
#     conversation = ConversationHistory(
#         messages=[ConversationMessage(role="user", content=user_query)]
#     )
#     print(f"[TEST] Submitting user query: {user_query}")
#     results = _run_cosmos_query_impl(conversation)
#     print("\n=== Search Results ===")
#     for email in results:
#         print(f"Subject: {email.subject}")
#         print(f"From: {email.sender}")
#         print(f"Sent: {email.sent_time}")
#         print(f"Preview: {email.body_preview}")
#         print("-" * 40)






# ─────────────────── Run as MCP Server ───────────────────
if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run_sse_async(host="0.0.0.0", port=8000))