# agents/base_agent.py

import os
import uuid
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class BaseAgent:
    def __init__(self, state_store: Optional[Dict[str, Any]], session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.state_store = state_store or {}
        self._chat_history = []

        # Load common settings
        self.azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.mcp_server_uri = os.getenv("MCP_SERVER_URI", "http://localhost:8000/sse")

    def append_to_chat_history(self, messages: list[dict]) -> None:
        self._chat_history.extend(messages)

    def _setstate(self, state: dict) -> None:
        self.state_store.update(self.session_id, state)

    @property
    def state(self):
        return self.state_store

# base_agent.py

class MemoryStateStore:
    def __init__(self):
        self._store = {}

    def get(self, session_id):
        return self._store.get(session_id, {})

    def set(self, session_id, state):
        self._store[session_id] = state

    def update(self, session_id, state):
        # Just use set for update
        self.set(session_id, state)
