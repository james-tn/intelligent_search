import os
import logging
from typing import Optional

from dotenv import load_dotenv
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.mcp import MCPSsePlugin

# Load environment variables
load_dotenv()

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SearchAgent:
    def __init__(self, state_store: Optional[dict] = None, session_id: Optional[str] = None) -> None:
        self._agent = None
        self._thread: Optional[ChatHistoryAgentThread] = None
        self._initialized = False

        self.session_id = session_id
        self.state = state_store or {}

    def _setstate(self, new_state: dict):
        self.state.update(new_state)

    def append_to_chat_history(self, messages: list[dict]):
        if "chat_history" not in self.state:
            self.state["chat_history"] = []
        self.state["chat_history"].extend(messages)

    async def _setup_agent(self) -> None:
        if self._initialized:
            return

        cosmos_plugin = MCPSsePlugin(
            name="EmailMCP",
            description="Cosmos Email Search Plugin",
            url="http://localhost:8000/sse",
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        await cosmos_plugin.connect()

        self._agent = ChatCompletionAgent(
            service=AzureChatCompletion(
                api_key=AZURE_OPENAI_KEY,
                endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_API_VERSION,
                deployment_name=AZURE_DEPLOYMENT,
            ),
            name="EmailSearchBot",
            instructions=(
                "You are a helpful assistant. You can search emails stored in Cosmos DB as per user queries. "
                "You take user input which is natural language and pass it as it is to process using MCP tools. "
                "Return most relevant resuls only"
                "For others you can say 'less possible ones can be'"
            ),
            plugins=[cosmos_plugin],
        )

        if "thread" in self.state:
            try:
                self._thread = self.state["thread"]
                logger.info("Restored thread from state store")
            except Exception as e:
                logger.warning(f"Could not restore thread: {e}")

        self._initialized = True

    async def chat_async(self, prompt: str) -> str:
        await self._setup_agent()
        params = {
            "search_text": prompt,
            "filter": ""
        }
        response = await self._agent.get_response(messages=prompt, thread=self._thread)
        response_content = str(response.content)

        self._thread = response.thread
        if self._thread:
            self._setstate({"thread": self._thread})

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response_content},
        ]
        self.append_to_chat_history(messages)

        return response_content






# ✅ TEST BLOCK TO VERIFY MCP INTEGRATION
if __name__ == "__main__":
    async def main():
        print("⚙️  Starting SearchAgent Test (MCP & Cosmos Integration)...\n")
        state = {}
        agent = SearchAgent(state_store=state, session_id="test-session")

        user_input = input("Enter your email search query: ").strip()
        try:
            response = await agent.chat_async(user_input)
            print("\n🧠 Agent Response:")
            print(response)
        except Exception as e:
            logger.exception("🚨 Error during agent test run:")
            print(f"\n❌ Exception: {e}")
    import asyncio
    asyncio.run(main())