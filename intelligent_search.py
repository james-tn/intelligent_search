#!/usr/bin/env python3  
  
# ───────────────────────────── Begin search_app.py ─────────────────────────────  
import os  
import json  
import time  
import streamlit as st  
from azure.search.documents import SearchClient  
from azure.core.credentials import AzureKeyCredential  
from openai import AzureOpenAI  
from azure.search.documents.models import VectorizableTextQuery, VectorFilterMode  
from dotenv import load_dotenv  
from pydantic import BaseModel  
from azure.search.documents.models import QueryType, QueryCaptionType, QueryAnswerType  
import time
# Load environment variables  
load_dotenv()  
  
# 1. Azure OpenAI configuration (new syntax)  
azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")  
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")  
chat_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")  
  
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
#print out the current time in 2025-06-13T00:00:00Z format
def get_current_time():
    """Get the current time in ISO 8601 format."""  
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
# 2. Generate a complete Azure Cognitive Search query from the conversation history  
def generate_search_query(conversation_history: list) -> dict:  
    """  
    Converts the conversation of natural language email search queries into a complete Azure Cognitive  
    Search query in JSON format. The JSON object must contain exactly the following keys:  
    "search_text" and "filter".  
    """  
    system_prompt = (  
        f"Today is {get_current_time()}. You are an expert query translator for Azure Cognitive Search. Convert the conversation of natural language "  
        "email search queries into a complete Azure Cognitive Search query in JSON format. The JSON object must contain "  
        "exactly the following keys: 'search_text' and 'filter'.\n\n"  
        "Additional guidelines:\n"  
        "1. The 'search_text' key should include useful free-text search terms (from subject, body, and attachments).\n"  
        "2. The 'filter' key should include any structured OData filter expressions. When filtering on dates, use ISO 8601 format.\n"  
        "   For example, if the user says 'before June 13 2025', output a filter like: sent_time lt 2025-06-13T00:00:00Z.\n"  
        "   Likewise, use 'gt' for 'after'. For sender emails or other string properties, use eq with proper quoting.\n"  
        "3. Output only the JSON object – do not include any additional text or commentary.\n\n"  
        "Index schema:\n"  
        " - id (string, key)\n"  
        " - from (string): Sender Email\n"  
        " - to_list (string): Recipient list\n"  
        " - cc_list (string): CC list\n"  
        " - subject (string)\n"  
        " - important (int)\n"  
        " - body (string)\n"  
        " - category (string)\n"  
        " - attachment_names (collection of string)\n"  
        " - received_time (DateTimeOffset)   // e.g., \"2025-06-13T00:00:00Z\"\n"  
        " - sent_time (DateTimeOffset)         // e.g., \"2025-06-13T00:00:00Z\"\n"  
        " - size (int)\n\n"  
        "Each user input may add additional constraints to your query and should be combined with previous inputs. \n"  
        "Now convert the following conversation of natural language queries into a complete Azure Cognitive Search query JSON:"  
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
        messages.append({"role": "user", "content": f"Error: {error_msg} Please provide a valid JSON object with only the keys 'search_text', 'filter'"})  
        time.sleep(1)  
  
    return None  
  
# 3. Azure Cognitive Search configuration and query execution  
service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")  # e.g., "https://.search.windows.net"  
index_name = os.getenv("AZURE_SEARCH_INDEX")  # e.g., "emails-index"  
admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")  
credential = AzureKeyCredential(admin_key)  
  
search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)  
  
def run_search_query(query_json: dict):  
    """Execute the search query on Azure Cognitive Search using the generated JSON."""  
    # Use the search_text and filter returned from our generated query JSON  
    search_text = query_json.get("search_text", "")  
    filter_str = query_json.get("filter", None)  
    vector_query = VectorizableTextQuery(text=search_text, k_nearest_neighbors=50, fields="bodyVector")  
  
    results = search_client.search(  
        vector_queries=[vector_query],  
        search_text=search_text,  
        vector_filter_mode=VectorFilterMode.PRE_FILTER,  
        filter=filter_str,  
        query_type=QueryType.SEMANTIC,  
        semantic_configuration_name='my-semantic-config',  
        query_caption=QueryCaptionType.EXTRACTIVE,  
        query_answer=QueryAnswerType.EXTRACTIVE,  
        top=3  
    )  
    # Convert the generator into a list  
    return [doc for doc in results]  
  
# 4. Streamlit UI  
st.set_page_config(page_title="Intelligent Email Search", layout="wide")  
st.title("Intelligent Email Search")  
st.markdown(  
    "Enter a natural language email search query. For example:\n\n"  
    "show me the emails created with content about project management\n\n"  
    "Then after seeing the results, you can add follow-up criteria (for example, "  
    "only emails created before Apr 02 2025) and a new query that combines all requests will be generated."  
)  
  
# Initialize conversation history in session state if not already present.  
if "conversation_history" not in st.session_state:  
    st.session_state.conversation_history = []  
  
# Use a Streamlit form so that the input text clears after submission.  
with st.form("query_form", clear_on_submit=True):  
    # Create two columns for layout to place buttons side by side  
    col1, col2 = st.columns([4, 1])  # Adjust the column widths as needed  
  
    new_query = col1.text_input("Search Query:")  # Input field in the first column  
  
    # Buttons at the bottom of the form  
    col1, col2 = st.columns([1, 1])  # Create two equal-width columns for buttons  
    with col1:  
        submitted = st.form_submit_button("Submit")  
    with col2:  
        clear_history = st.form_submit_button("Clear Conversation History")  
  
# Handle the submission and clearing functionality  
if submitted and new_query:  
    # Append the new query to our conversation history.  
    st.session_state.conversation_history.append({"role": "user", "content": new_query})  
  
if clear_history:  
    # Clear the conversation history  
    st.session_state.conversation_history = []  
    st.success("Conversation history cleared!")  
# Display the last two queries (if available) beneath the search box.  
if st.session_state.conversation_history:  
    last_queries = st.session_state.conversation_history[-2:]  
    for idx, msg in enumerate(last_queries, start=1):  
        st.write(f"{msg['content']}")  
  
# If there is any conversation history, generate a new combined query and run it against Azure Cognitive Search.  
if st.session_state.conversation_history:  
    st.info("Generating complete Azure Cognitive Search query using conversation history…")  
    with st.spinner("Calling OpenAI to generate the query…"):  
        query_json = generate_search_query(st.session_state.conversation_history)  
        if query_json is None:  
            st.error("Unable to generate a valid search query. Please try again.")  
        else:  
            st.markdown("Generated Azure Cognitive Search Query")  
            st.json(query_json)  
  
            with st.spinner("Running query against Azure Cognitive Search…"):  
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
                    body_text = res.get('body', '')  
                    body_preview = body_text[:200] + ("..." if len(body_text) > 200 else "")  
                    st.write(f"**Body Preview:** {body_preview}")  
                    st.markdown("---")  
  
# ───────────────────────────── End search_app.py ─────────────────────────────  