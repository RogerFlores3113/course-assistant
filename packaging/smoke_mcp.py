"""CI smoke test: launch a built executable as an MCP server over stdio, the way Claude Desktop does.

Usage: python packaging/smoke_mcp.py <path-to-executable>
"""
import os
import sys
import tempfile

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED_TOOLS = {"list_courses", "search_materials", "read_passage", "list_upcoming", "add_course"}


async def main(exe: str) -> None:
    env = {**os.environ, "COURSE_ASSISTANT_DATA_DIR": tempfile.mkdtemp()}
    params = StdioServerParameters(command=exe, args=["mcp"], env=env)
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = {t.name for t in (await s.list_tools()).tools}
            missing = EXPECTED_TOOLS - tools
            assert not missing, f"MCP server is missing tools: {sorted(missing)} (has {sorted(tools)})"
            res = await s.call_tool("list_courses", {})
            assert not res.isError, res
    print(f"MCP smoke test passed ({len(tools)} tools): {exe}")


if __name__ == "__main__":
    anyio.run(main, sys.argv[1])
