"""LangGraph chat workflow: understand → retrieve → generate → respond.

``build_chat_graph`` is the only public entry point.  It accepts the I/O
dependencies (database session, LLM clients), creates closed-over node
callables via the factories in ``nodes.py``, and compiles the graph.

The conditional edge from ``retrieve`` lets the graph skip ``generate``
entirely when the retrieve node has already set ``outcome = "abstained"``
or ``outcome = "error"``.  In that case control jumps straight to
``respond`` which will emit an empty citations list.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from ai.agent.nodes import (
    make_generate_node,
    make_retrieve_node,
    respond_node,
    understand_node,
)
from ai.agent.state import ChatState
from ai.llm.interfaces import ChatClient, EmbeddingClient


def _route_after_retrieve(
    state: ChatState,
) -> Literal["generate", "respond"]:
    """Skip generation when retrieval already reached a terminal outcome."""
    if state.outcome in ("abstained", "error", "refused", "blocked"):
        return "respond"
    return "generate"


def build_chat_graph(
    db_session: AsyncSession,
    chat_client: ChatClient,
    embedding_client: EmbeddingClient,
) -> RunnableConfig:
    """Compile and return the chat workflow graph.

    Dependencies are closed over by the node factories so the compiled graph
    carries no references to FastAPI's DI machinery and can be invoked with
    a plain ``graph.ainvoke(state_dict)``.
    """
    retrieve = make_retrieve_node(db_session, embedding_client)
    generate = make_generate_node(chat_client)

    workflow: StateGraph = StateGraph(ChatState)

    workflow.add_node("understand", understand_node)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_node("respond", respond_node)

    workflow.set_entry_point("understand")
    workflow.add_edge("understand", "retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"generate": "generate", "respond": "respond"},
    )
    workflow.add_edge("generate", "respond")
    workflow.set_finish_point("respond")

    return workflow.compile()
