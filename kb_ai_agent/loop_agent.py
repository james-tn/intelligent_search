"""  
Agent implementation that uses  
 • Autogen agents/teams  
 • A Google Custom Search tool that is HARD‑WIRED to “site:veeam.com …”  
The agent state (chat history) is still kept in the in‑memory SESSION_STORE  
dict that the FastAPI backend supplies.  
"""  
from __future__ import annotations  
  
import asyncio  
import logging  
import os  
from typing import Any, Dict, List, Optional  
from autogen_core import CancellationToken  
from requests_html import HTMLSession  
import re
  
import httpx  
from dotenv import load_dotenv  
  
# ─────────────────── Autogen imports ────────────────────  
from autogen_agentchat.agents import AssistantAgent  
from autogen_agentchat.conditions import TextMessageTermination  
from autogen_agentchat.teams import RoundRobinGroupChat  
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient  
  
# ─────────────────── env / logging ──────────────────────  
load_dotenv()  
logger = logging.getLogger(__name__)  
logger.setLevel(logging.INFO)  
  
  
# ─────────────────── Google search tool ─────────────────  
def veeam_google_search(query: str, num_results: int = 6) -> List[Dict[str, str]]:  
    print(f"veeam_google_search: {query}")
    """  
    Search ONLY the veeam.com domain by means of Google Custom Search.  
  
    Parameters  
    ----------  
    query : str  
        The user question WITHOUT the “site:” restriction.  
    num_results : int, optional  
        Max. number of results to return (<= 10), by default 6.  
  
    Returns  
    -------  
    List[dict]  
        Each list item has keys: title, link, snippet  
    """  
    api_key = os.getenv("GOOGLE_API_KEY")  
    cse_id = os.getenv("GOOGLE_CSE_ID")  
    if not api_key or not cse_id:  
        raise RuntimeError(  
            "GOOGLE_API_KEY and GOOGLE_CSE_ID env variables must be configured!"  
        )  
  
    #   Use the Custom Search REST endpoint  
    url = "https://www.googleapis.com/customsearch/v1"  
    params = {  
        "key": api_key,  
        "cx": cse_id,  
        "q": f"site:veeam.com {query}",  
        "num": max(1, min(num_results, 10)),  
    }  
  
    with httpx.Client(timeout=15) as client:  
        r = client.get(url, params=params)  
        r.raise_for_status()  
        data = r.json()  
  
    items: list[dict] = data.get("items", [])  
    results: List[Dict[str, str]] = [  
        {"title": it["title"], "link": it["link"], "snippet": it.get("snippet", "")}  
        for it in items  
    ]  
    return results  
def fetch_url_content(url: str, max_chars: int = 28000) -> str:  
    print(f"fetch_url: {url}")
    """  
    Returns the text content of the given veeam.com URL.  
  
    Parameters  
    ----------  
    url : str  
        Target URL (must be from the veeam.com domain)  
    max_chars : int  
        Maximum number of characters to return to prevent LLM token overflow  
    """  
    if not re.match(r"^https?://([a-z0-9.-]*\.)?veeam\.com", url, re.I):  
        raise ValueError("fetch_url_content: Only veeam.com domain is allowed")  
  
    session = HTMLSession()  
    r = session.get(url, timeout=20)  
  
    # ⚠️ Some pages may require JS rendering.  
    r.html.render(headless=True)  # Uncomment if needed  
  
    text = r.html.text  # Visible text  
    # if len(text) > max_chars:  
    #     text = text[:max_chars] + "\n...[truncated]"  
    print("fetch_url_content:\n ", text)
    return text  
  
# ─────────────────── Stateful Base class ────────────────  
class BaseAgent:  
    """  
    Keeps the chat history and arbitrary state in the supplied dict (SESSION_STORE).  
    NOT responsible for the actual language‑model call; that is done by the concrete class.  
    """  
  
    def __init__(self, state_store: Dict[str, Any], session_id: str):  
        self.session_id = session_id  
        self.state_store = state_store  
  
        # reusable history  
        self.chat_history: List[Dict[str, str]] = self.state_store.get(  
            f"{session_id}_chat_history", []  
        )  
        self.state: Optional[Any] = self.state_store.get(session_id)  
  
    # ------------- convenience helpers ------------------  
    def append_to_chat_history(self, messages: List[Dict[str, str]]) -> None:  
        self.chat_history.extend(messages)  
        self.state_store[f"{self.session_id}_chat_history"] = self.chat_history  
  
    async def chat_async(self, prompt: str) -> str:  # noqa: D401  
        """Must be implemented by subclass."""  
        raise NotImplementedError  
  
  
# ─────────────────── Concrete Autogen agent ─────────────  
class Agent(BaseAgent):  
    """  
    One‑agent “team” that is able to call the `veeam_google_search` tool  
    whenever it needs external knowledge.  
    """  
  
    def __init__(self, state_store: Dict[str, Any], session_id: str):  
        super().__init__(state_store, session_id)  
          # Set up the OpenAI/Azure model client  
        self.model_client = AzureOpenAIChatCompletionClient(  
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),  
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),  
            model=os.getenv("OPENAI_MODEL_NAME"),  
        )  

        # --- the autogen assistant with the Google tool attached ---  
        prompt = """  
You are a **research‑focused AI assistant** specializing in **Veeam products**. Your job is to provide the most accurate, comprehensive answer possible by investigating all official information available from veeam.com.  
  
### TOOLS AVAILABLE  
  
- `veeam_google_search(query)`    
    *Google Custom Search restricted to veeam.com.*  
  
- `fetch_url_content(url)`    
    *Fetch the plain textual content of a given veeam.com URL.*  
  
---  
  
### WORKFLOW & INSTRUCTIONS  
  
1. **Decompose** the user’s question into key sub‑topics and list the facts or specifics you must verify.  
2. **Be creative and persistent in searching:**  
    - When your initial search does not yield good results, creatively invent new search queries—rephrase, broaden, narrow, or switch out keywords to surface relevant information.  
    - Think like a researcher. Use synonyms, related concepts, or even ask the question in a different way.  
    - Be patient—retry until you are confident you have found the best possible answer based on available information.  
3. **veeam_google_search** each sub-topic, **carefully review snippets** for relevance.  
4. If a result looks promising, **fetch_url_content** from that URL to read its full text (you may “double‑click” and check several links for thoroughness).  
5. **Iterate:** Cross‑check information from multiple sources, compare findings, and synthesize a clear, complete answer.  
6. **If information is unclear or insufficient:**    
    - Ask relevant follow‑up questions to clarify the user’s intent.  
    - If official info is missing, clearly state that veeam.com does not have the answer—never make up information.  
7. In your **final response**:  
    - Present a **clear, well‑structured answer.**  
    - Add an explicit **“Sources”** section listing every veeam.com URL you used.  
  
---  
  
*Output everything in markdown format.*  """
        self._assistant = AssistantAgent(  
            "veeam_search_assistant",  
            model_client=self.model_client,  
            tools=[veeam_google_search,fetch_url_content],  
            system_message=(  prompt
            ),  
        )  
  
        # Terminate as soon as the assistant responds with a text message  
        self._termination = TextMessageTermination("veeam_search_assistant")  
  
        # A Round‑Robin team although there is only one member → simpler API usage  
        self._team = RoundRobinGroupChat(  
            [self._assistant], termination_condition=self._termination  
        )  
  

    async def chat_async(self, prompt: str) -> str:  
        """Ensure agent/tools are ready and process the prompt."""  
        if self.state:  
            await self._team.load_state(self.state)  

        response = await self._team.run(task=prompt, cancellation_token=CancellationToken())  
        assistant_response = response.messages[-1].content  
  
        messages = [  
            {"role": "user", "content": prompt},  
            {"role": "assistant", "content": assistant_response}  
        ]  
        self.append_to_chat_history(messages)  
  
        # Update/store latest agent state  
        new_state = await self._team.save_state()  
        self.state_store[self.session_id] = new_state  
  
        return assistant_response  
    # Optional – close the client when the process terminates  
    async def __aexit__(self, *exc):  
        await self.model_client.close()  