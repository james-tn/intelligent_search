#!/usr/bin/env python3  
  
# Import required modules  
import os  
import json  
import time  
import streamlit as st  
import requests  
import uuid  
from azure.cosmos import CosmosClient, PartitionKey  
from azure.identity import DefaultAzureCredential  
from openai import AzureOpenAI  
from dotenv import load_dotenv  
  
# Load environment variables from .env file  
load_dotenv()  
  
# ───────────────────────── Azure OpenAI Configuration ─────────────────────────  
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")  
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")  
chat_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")  
  
# Create the Azure OpenAI client for chat completions  
chat_completion_client = AzureOpenAI(  
    api_key=azure_openai_key,  
    azure_endpoint=azure_openai_endpoint,  
    api_version=azure_openai_api_version,  
)  
  
def get_openai_chat_response(messages):  
    """Get the OpenAI chat response using the new Azure OpenAI syntax."""  
    try:  
        response = chat_completion_client.chat.completions.create(  
            model=chat_model,  
            messages=messages,  
            max_tokens=500,  
        )  
        return response.choices[0].message.content  
    except Exception as e:  
        print(f"Error getting OpenAI chat response: {e}")  
        return None  
  
def get_current_time():  
    """Return the current time in ISO 8601 format."""  
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())  
  
# ───────────────────────── Query Generation ─────────────────────────  
def generate_search_query(conversation_history: list) -> dict:  
    """  
    Translates the natural language conversation into a JSON object containing two keys:  
    'search_text' and 'filter'. The system prompt instructs OpenAI to output ONLY the JSON.  
    """  
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
  
    messages = [{"role": "system", "content": system_prompt}] + conversation_history  
    max_attempts = 3  
  
    for attempt in range(max_attempts):  
        reply = get_openai_chat_response(messages)  
        if not reply:  
            error_msg = "No response obtained from OpenAI."  
        else:  
            try:  
                query_json = json.loads(reply)  
                if "search_text" in query_json and "filter" in query_json:  
                    return query_json  
                else:  
                    error_msg = "The JSON is missing required keys 'search_text' or 'filter'."  
            except Exception as e:  
                error_msg = f"JSON parse error: {str(e)}. Full response was: {reply}"  
  
        messages.append({"role": "assistant", "content": reply if reply else ""})  
        messages.append({"role": "user", "content": f"Error: {error_msg} Please provide a valid JSON object with only the keys 'search_text' and 'filter'."})  
        time.sleep(1)  
  
    return None  
  
# ───────────────────────── Cosmos DB Configuration ─────────────────────────  
cosmos_uri = os.getenv("COSMOS_URI", "https://<your-account>.documents.azure.com:443/")  
cosmos_db_name = os.getenv("COSMOS_DB_NAME", "vectordb")  
container_name = os.getenv("COSMOS_CONTAINER_NAME", "vectortest_hybridsearch")  
  
# Set your AAD credentials for Cosmos DB authentication  
aad_client_id = os.getenv("AAD_CLIENT_ID")  
aad_client_secret = os.getenv("AAD_CLIENT_SECRET")  
aad_tenant_id = os.getenv("AAD_TENANT_ID")  
  
os.environ["AZURE_CLIENT_ID"] = aad_client_id  
os.environ["AZURE_CLIENT_SECRET"] = aad_client_secret  
os.environ["AZURE_TENANT_ID"] = aad_tenant_id  
  
credential = DefaultAzureCredential()  
cosmos_client = CosmosClient(cosmos_uri, credential=credential)  
cosmos_db_client = cosmos_client.get_database_client(cosmos_db_name)  
cosmos_container_client = cosmos_db_client.get_container_client(container_name)  
  
# ───────────────────────── Embedding Function ─────────────────────────  
def get_embedding(text):  
    """  
    Compute an embedding for the input text using Azure OpenAI.  
    Adjust the deployment name (AZURE_OPENAI_EMB_DEPLOYMENT) in your environment variables.  
    """  
    emb_deployment = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT")  
    url = f"{azure_openai_endpoint}/openai/deployments/{emb_deployment}/embeddings?api-version={azure_openai_api_version}"  
    headers = {"Content-Type": "application/json", "api-key": azure_openai_key}  
    response = requests.post(url, headers=headers, json={"input": text})  
    response.raise_for_status()  
    return response.json()["data"][0]["embedding"]  
  
# ───────────────────────── Query Execution ─────────────────────────  
def run_search_query(query_json: dict):  
    """  
    Execute a Cosmos DB vector search query.  
    The query uses vector similarity on the 'bodyVector' field and full-text scoring on the 'body' field.  
    """  
    search_text = query_json.get("search_text", "")  
    filter_str = query_json.get("filter", "").strip()  
  
    # Get embedding for the search text  
    search_embedding = get_embedding(search_text)  
    embedding_literal = "[" + ",".join(str(x) for x in search_embedding) + "]"  
  
    # Escape single quotes in search_text for SQL safety  
    safe_search_text = search_text.replace("'", "\\'")  
  
    # Build WHERE clause if a filter is provided  
    where_clause = f"WHERE {filter_str}" if filter_str else ""  
  
    # Build the query string  
    query_string = f"""  
    SELECT TOP 20 c.id, c.from, c.subject, c.sent_time, c.body  
    FROM c  
    {where_clause}  
    ORDER BY RANK RRF(  
        VectorDistance(c.bodyVector, {embedding_literal}),  
        FullTextScore(c.body, '{safe_search_text}')  
    )  
    """  
    print("Executing Cosmos DB Query:")  
    print(query_string)  
  
    items = list(cosmos_container_client.query_items(  
        query=query_string,  
        enable_cross_partition_query=True  
    ))  
    return items  
  
# ───────────────────────── Streamlit UI ─────────────────────────  
st.set_page_config(page_title="Intelligent Email Search (Cosmos DB)", layout="wide")  
st.title("Intelligent Email Search with Cosmos DB")  
st.markdown(  
    "Enter a natural language email search query. For example:\n\n"  
    "Show me emails about project management sent before June 13, 2025.\n\n"  
    "Then, after viewing the results, you can add follow-up criteria (e.g., only emails from a certain sender), "  
    "and a new query that combines all requests will be generated."  
)  
  
# Initialize conversation history in session state if not already present  
if "conversation_history" not in st.session_state:  
    st.session_state.conversation_history = []  
  
# Use a Streamlit form so that input text clears after submission  
with st.form("query_form", clear_on_submit=True):  
    col1, col2 = st.columns([4, 1])  
    new_query = col1.text_input("Search Query:")  
    col1, col2 = st.columns([1, 1])  
    with col1:  
        submitted = st.form_submit_button("Submit")  
    with col2:  
        clear_history = st.form_submit_button("Clear Conversation History")  
  
if submitted and new_query:  
    st.session_state.conversation_history.append({"role": "user", "content": new_query})  
  
if clear_history:  
    st.session_state.conversation_history = []  
    st.success("Conversation history cleared!")  
  
# Display recent conversation messages  
if st.session_state.conversation_history:  
    st.info("Generating complete Cosmos DB query using conversation history…")  
    with st.spinner("Calling OpenAI to generate the query…"):  
        query_json = generate_search_query(st.session_state.conversation_history)  
        if query_json is None:  
            st.error("Unable to generate a valid search query. Please try again.")  
        else:  
            st.markdown("Generated Cosmos DB Search Query:")  
            st.json(query_json)  
            with st.spinner("Running query against Cosmos DB…"):  
                results_list = run_search_query(query_json)  
                time.sleep(0.5)  
  
                st.subheader("Search Results")  
                if not results_list:  
                    st.write("No results found.")  
                else:  
                    for idx, res in enumerate(results_list):  
                        st.markdown(f"**Result {idx+1}:**")  
                        st.write(f"**From:** {res.get('from', 'N/A')}")  
                        st.write(f"**Subject:** {res.get('subject', 'N/A')}")  
                        st.write(f"**Sent Time:** {res.get('sent_time', 'N/A')}")  
                        body_text = res.get("body", "")  
                        body_preview = body_text[:200] + ("..." if len(body_text) > 200 else "")  
                        st.write(f"**Body Preview:** {body_preview}")  
                        st.markdown("---")  