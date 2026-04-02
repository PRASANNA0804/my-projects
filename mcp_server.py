"""
mcp_server.py — MCP (Model Context Protocol) Server
=====================================================
Exposes the RAG system as MCP tools so that any MCP-compatible
LLM host (Claude Desktop, LangChain, etc.) can call them.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  MCP Host (e.g. Claude Desktop / LangChain)             │
  │    │  calls tool via MCP protocol                       │
  │    ▼                                                     │
  │  mcp_server.py  (this file — thin MCP adapter layer)    │
  │    │  imports and delegates to                           │
  │    ▼                                                     │
  │  agent.py / retriever.py / ingest.py  (business logic)  │
  │    │                                                     │
  │    ▼                                                     │
  │  ChromaDB  +  OpenAI LLM                                │
  └─────────────────────────────────────────────────────────┘

The MCP server is stateless: each tool call is independent.

Usage:
  pip install mcp
  python mcp_server.py

Then add to claude_desktop_config.json:
  {
    "mcpServers": {
      "rag": {
        "command": "python",
        "args": ["/path/to/backend/mcp_server.py"]
      }
    }
  }
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from retriever import retrieve, format_context
from agent     import get_agent
from ingest    import ingest_document

# ── Create the MCP server instance ────────────────────────────────────────────
server = Server("rag-agent")


# ── Tool definitions ───────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Declare which tools this MCP server exposes."""
    return [
        types.Tool(
            name        ="search_documents",
            description =(
                "Search the RAG knowledge base for text chunks relevant to "
                "a user question. Returns raw text passages and their sources. "
                "Use this when you need to look up specific information."
            ),
            inputSchema ={
                "type"      : "object",
                "properties": {
                    "question": {
                        "type"       : "string",
                        "description": "The search query or user question.",
                    },
                    "top_k": {
                        "type"       : "integer",
                        "description": "Number of chunks to return (default 5).",
                        "default"    : 5,
                    },
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name        ="answer_question",
            description =(
                "Answer a user question end-to-end using RAG. "
                "Retrieves relevant document chunks, builds a prompt, "
                "calls the LLM, and returns a natural-language answer "
                "with cited sources."
            ),
            inputSchema ={
                "type"      : "object",
                "properties": {
                    "question": {
                        "type"       : "string",
                        "description": "The user's question to answer.",
                    },
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name        ="ingest_document",
            description =(
                "Ingest a text file into the RAG knowledge base. "
                "Pass an absolute path to a .txt file on the server. "
                "Returns the number of chunks stored."
            ),
            inputSchema ={
                "type"      : "object",
                "properties": {
                    "file_path": {
                        "type"       : "string",
                        "description": "Absolute path to the .txt file to ingest.",
                    },
                },
                "required": ["file_path"],
            },
        ),
    ]


# ── Tool handlers ──────────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Dispatch incoming tool calls to the appropriate RAG function."""

    # ── search_documents ──────────────────────────────────────────────────────
    if name == "search_documents":
        question = arguments.get("question", "")
        top_k    = int(arguments.get("top_k", 5))

        chunks = retrieve(question, top_k=top_k)
        if not chunks:
            return [types.TextContent(
                type="text",
                text="No relevant documents found. Run ingest_document first.",
            )]

        lines = [f"Found {len(chunks)} relevant chunk(s):\n"]
        for i, c in enumerate(chunks, 1):
            lines.append(
                f"[{i}] Source: {c.source}  |  Relevance: {c.relevance:.2%}\n"
                f"{c.text[:400]}\n"
            )

        return [types.TextContent(type="text", text="\n".join(lines))]

    # ── answer_question ───────────────────────────────────────────────────────
    elif name == "answer_question":
        question = arguments.get("question", "")
        agent    = get_agent()
        resp     = agent.answer(question)

        text = (
            f"Answer:\n{resp.answer}\n\n"
            f"Sources: {', '.join(resp.unique_sources) or 'none'}"
        )
        return [types.TextContent(type="text", text=text)]

    # ── ingest_document ───────────────────────────────────────────────────────
    elif name == "ingest_document":
        file_path = arguments.get("file_path", "")
        n_chunks  = ingest_document(file_path)
        return [types.TextContent(
            type="text",
            text=f"Successfully ingested {n_chunks} chunks from '{file_path}'.",
        )]

    else:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


# ── Main ───────────────────────────────────────────────────────────────────────
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
