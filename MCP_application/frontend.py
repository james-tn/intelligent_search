# streamlit_ui.py

import os
import streamlit as st
import asyncio
from agent import SearchAgent  # Make sure agent.py is in same folder
from base_agent import MemoryStateStore  # Assumes base_agent.py exists and defines this

# ────────────────────────────── App Setup ──────────────────────────────
st.set_page_config(page_title="Intelligent Email Search", layout="wide")
st.title("📧 Intelligent Email Search (Cosmos DB + Azure OpenAI)")

st.markdown(
    """
    Enter a natural language search query. Example:
    - *Show me all emails from Alice about project deadlines.*
    - *Find emails sent before May 1st about product launches.*

    Your input will be processed by a smart agent that uses Azure OpenAI + Cosmos DB.
    """
)

# ────────────────────────────── Session Setup ──────────────────────────────
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "search_agent" not in st.session_state:
    state_store = MemoryStateStore()
    session_id = "streamlit-session"  # could be user ID, etc.
    st.session_state.search_agent = SearchAgent(state_store, session_id)

# ────────────────────────────── Input Form ──────────────────────────────
with st.form("query_form", clear_on_submit=True):
    query = st.text_input("Search Query", placeholder="e.g. Show me emails from Bob about sales")
    submitted = st.form_submit_button("Submit")
    clear_history = st.form_submit_button("Clear Conversation")

# ────────────────────────────── Logic ──────────────────────────────
if clear_history:
    st.session_state.conversation = []
    st.success("Conversation cleared.")

if submitted and query:
    st.session_state.conversation.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        agent_response = loop.run_until_complete(
            st.session_state.search_agent.chat_async(query)
        )

    st.session_state.conversation.append({"role": "assistant", "content": agent_response})

# ────────────────────────────── Display Chat ──────────────────────────────
if st.session_state.conversation:
    st.subheader("🔎 Conversation")
    for msg in st.session_state.conversation:
        if msg["role"] == "user":
            st.markdown(f"**🧑 User:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Agent:** {msg['content']}")
        st.markdown("---")
