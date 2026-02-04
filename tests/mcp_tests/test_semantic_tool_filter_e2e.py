"""End-to-end test for MCP Semantic Tool Filtering.

This test should not depend on external embedding providers or API keys.
"""

import importlib.util
import os
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from mcp.types import Tool as MCPTool

SEMANTIC_ROUTER_AVAILABLE = (
    importlib.util.find_spec("semantic_router") is not None
)


@pytest.mark.asyncio
@pytest.mark.skipif(
    not SEMANTIC_ROUTER_AVAILABLE,
    reason=(
        "semantic-router not installed. Install with: pip install 'litellm[semantic-router]'"
    ),
)
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set in environment"
)
async def test_e2e_semantic_filter():
    """E2E: Load router/filter and verify hook filters tools."""
    pytest.importorskip("semantic_router")

    from litellm import Router
    from litellm.proxy.hooks.mcp_semantic_filter import SemanticToolFilterHook
    from litellm.proxy._experimental.mcp_server.semantic_tool_filter import (
        SemanticMCPToolFilter,
    )
    from litellm.types.utils import Embedding, EmbeddingResponse

    # Create router and filter
    router = Router(
        model_list=[
            {
                "model_name": "text-embedding-3-small",
                "litellm_params": {"model": "openai/text-embedding-3-small"},
            }
        ]
    )

    # Avoid external network calls: mock embeddings at the Router boundary.
    def _mock_embedding_response(input_docs):
        if isinstance(input_docs, str):
            docs = [input_docs]
        else:
            docs = list(input_docs)

        # Return deterministic embeddings; semantic-router only needs consistent
        # vector shapes for ranking.
        data = [
            Embedding(embedding=[0.1] * 1536, index=i, object="embedding")
            for i in range(len(docs))
        ]
        return EmbeddingResponse(
            data=data,
            model="text-embedding-3-small",
            object="list",
            usage={"prompt_tokens": 10, "total_tokens": 10},
        )

    def mock_embedding_sync(*args, **kwargs):
        return _mock_embedding_response(kwargs.get("input"))

    async def mock_embedding_async(*args, **kwargs):
        return _mock_embedding_response(kwargs.get("input"))

    router.embedding = mock_embedding_sync  # type: ignore[assignment]
    router.aembedding = mock_embedding_async  # type: ignore[assignment]

    filter_instance = SemanticMCPToolFilter(
        embedding_model="text-embedding-3-small",
        litellm_router_instance=router,
        top_k=3,
        enabled=True,
    )

    # Create 10 tools
    tools = [
        MCPTool(
            name="gmail_send",
            description="Send an email via Gmail",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="calendar_create",
            description="Create a calendar event",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="file_upload",
            description="Upload a file",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="web_search",
            description="Search the web",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="slack_send",
            description="Send Slack message",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="doc_read",
            description="Read document",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="db_query",
            description="Query database",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="api_call",
            description="Make API call",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="task_create",
            description="Create task",
            inputSchema={"type": "object"},
        ),
        MCPTool(
            name="note_add",
            description="Add note",
            inputSchema={"type": "object"},
        ),
    ]

    # Build router with test tools
    filter_instance._build_router(tools)

    hook = SemanticToolFilterHook(filter_instance)

    data = {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": "Send an email and create a calendar event",
            }
        ],
        "tools": tools,
        "metadata": {},  # Initialize metadata dict for hook to store filter stats
    }

    # Call hook
    result = await hook.async_pre_call_hook(
        user_api_key_dict=Mock(),
        cache=Mock(),
        data=data,
        call_type="completion",
    )

    # Single assertion: hook filtered tools
    assert result and len(result["tools"]) < len(
        tools
    ), f"Expected filtered tools, got {len(result['tools'])} tools (original: {len(tools)})"

    # Intentionally avoid print() in tests to keep linting strict.
