# run_mcp.py — Entry point to start your MCP tools server (mcp.py)

import asyncio
from mcp_server import mcp

if __name__ == "__main__":
    # Run the FastMCP server on localhost:8000
    asyncio.run(mcp.run_sse_async(host="0.0.0.0", port=8000))
