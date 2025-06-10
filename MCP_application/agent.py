# agent.py (matches pattern from chat_agent.py)

import logging
from base_agent import BaseAgent
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.mcp import MCPSsePlugin


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)



class SearchAgent(BaseAgent):
    def __init__(self, state_store, session_id) -> None:
        super().__init__(state_store, session_id)
        self._agent = None
        self._initialized = False

    async def _setup_agent(self) -> None:
        if self._initialized:
            return

        cosmos_plugin = MCPSsePlugin(
            name="EmailMCP",
            description="Cosmos Email Search Plugin",
            url="http://localhost:8000/sse",  # from BaseAgent config
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        await cosmos_plugin.connect()

        self._agent = ChatCompletionAgent(
            service=AzureChatCompletion(
                api_key=self.azure_openai_key,
                endpoint=self.azure_openai_endpoint,
                api_version=self.api_version,
                deployment_name=self.azure_deployment,
            ),
            name="EmailSearchBot",
            instructions="You are a helpful assistant. You can search emails stored in Cosmos DB as per user queries"
            "you take user input which is natural language and create embedding to find best match from cosmos db."
            "Make sure to use appropriate tools from the MCPTools plugin in the right order to provide user with right answers.",
            plugins=[cosmos_plugin],
        )

        self._thread: ChatHistoryAgentThread | None = None
        if self.state and isinstance(self.state, dict) and "thread" in self.state:
            try:
                self._thread = self.state["thread"]
                logger.info("Restored thread from SESSION_STORE")
            except Exception as e:
                logger.warning(f"Could not restore thread: {e}")

        self._initialized = True

    async def chat_async(self, prompt: str) -> str:
        await self._setup_agent()
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
