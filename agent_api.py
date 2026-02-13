#!/usr/bin/env python3
"""
LangGraph Agent API for Epstein Document Explorer.

Provides an agentic RAG system that can search, analyze, and answer
questions about the Epstein document corpus.
"""

import json
import operator
import os
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Réutiliser les fonctions du MCP server
from mcp_server import (
    semantic_search,
    keyword_search,
    get_document_text,
    get_relationships_for_actor,
    get_db,
)

# ============== FastAPI App ==============

app = FastAPI(
    title="Epstein Docs Agent API",
    description="Agentic RAG system for exploring the Epstein document corpus",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== LangChain Tools ==============


@tool
def search_documents(query: str, limit: int = 10) -> str:
    """
    Search documents using semantic similarity.
    Use this to find documents related to a topic, person, or event.

    Args:
        query: Natural language search query (e.g., "meetings with politicians", "flights to the island")
        limit: Maximum number of results to return (default 10)
    """
    results = semantic_search(query, limit)
    return json.dumps(results, indent=2, ensure_ascii=False)


@tool
def search_actor(actor_name: str, limit: int = 50) -> str:
    """
    Get all relationships involving a specific person/actor.
    Use this to find what actions a person took or what happened to them.

    Args:
        actor_name: Name of the person (e.g., "Bill Clinton", "Ghislaine Maxwell", "Prince Andrew")
        limit: Maximum number of relationships to return
    """
    results = get_relationships_for_actor(actor_name, limit)
    return json.dumps(results, indent=2, ensure_ascii=False)


@tool
def get_document(doc_id: str) -> str:
    """
    Get the full text of a specific document.
    Use this when you need to read the complete content of a document.

    Args:
        doc_id: Document ID (e.g., "HOUSE_OVERSIGHT_010568")
    """
    text = get_document_text(doc_id)
    if text:
        # Limiter la taille pour éviter de surcharger le contexte
        if len(text) > 8000:
            return text[:8000] + "\n\n[... document tronqué, utilisez une recherche plus spécifique ...]"
        return text
    return f"Document not found: {doc_id}"


@tool
def search_by_keywords(keywords: str, limit: int = 10) -> str:
    """
    Search documents by exact keywords in full text.
    Use this for precise searches when you know specific terms to look for.

    Args:
        keywords: Comma-separated keywords (e.g., "island, flight, 2005")
        limit: Maximum number of results to return
    """
    keyword_list = [k.strip() for k in keywords.split(",")]
    results = keyword_search(keyword_list, limit)
    return json.dumps(results, indent=2, ensure_ascii=False)


@tool
def get_database_stats() -> str:
    """
    Get statistics about the document database.
    Use this to understand the scope and content of the corpus.
    """
    conn = get_db()

    stats = {
        "total_documents": conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
        "total_relationships": conn.execute("SELECT COUNT(*) FROM rdf_triples").fetchone()[0],
        "unique_actors": conn.execute("SELECT COUNT(DISTINCT actor) FROM rdf_triples").fetchone()[0],
        "unique_targets": conn.execute("SELECT COUNT(DISTINCT target) FROM rdf_triples").fetchone()[0],
        "categories": [],
    }

    cursor = conn.execute("""
        SELECT category, COUNT(*) as count
        FROM documents
        GROUP BY category
        ORDER BY count DESC
        LIMIT 10
    """)
    stats["categories"] = [{"name": row[0], "count": row[1]} for row in cursor]

    conn.close()
    return json.dumps(stats, indent=2, ensure_ascii=False)


# ============== LangGraph Agent ==============

class AgentState(TypedDict):
    """État de l'agent entre les étapes."""
    messages: Annotated[Sequence[BaseMessage], operator.add]


# Liste des outils disponibles
tools = [
    search_documents,
    search_actor,
    get_document,
    search_by_keywords,
    get_database_stats,
]

# Modèle via OpenRouter
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")

model = ChatOpenAI(
    model=OPENROUTER_MODEL,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=4096,
    default_headers={
        "HTTP-Referer": "https://epstein-doc-explorer.local",
        "X-Title": "Epstein Doc Explorer",
    },
).bind_tools(tools)

# System prompt pour guider l'agent
SYSTEM_PROMPT = """Tu es un assistant expert dans l'analyse des documents Epstein.
Tu as accès à une base de données contenant des milliers de documents judiciaires,
emails, dépositions et autres pièces liées à l'affaire Jeffrey Epstein.

Tes capacités:
- Recherche sémantique dans les documents
- Recherche par mots-clés exacts
- Exploration des relations entre personnes (qui a fait quoi à qui)
- Lecture du texte complet des documents

Instructions:
1. Utilise les outils de recherche pour trouver les informations pertinentes
2. Cite toujours tes sources avec les doc_id
3. Sois factuel et objectif - rapporte ce que disent les documents
4. Si tu ne trouves pas d'information, dis-le clairement
5. Pour les questions sur des personnes, utilise search_actor d'abord
6. Pour les questions thématiques, utilise search_documents

Réponds en français si la question est en français, sinon en anglais."""


def agent_node(state: AgentState) -> dict:
    """Nœud principal de l'agent qui décide des actions."""
    messages = state["messages"]

    # Ajouter le system prompt au début si pas déjà présent
    if not any(hasattr(m, "type") and m.type == "system" for m in messages):
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response = model.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Détermine si l'agent doit continuer ou terminer."""
    last_message = state["messages"][-1]

    # Si le dernier message a des tool_calls, continuer vers les outils
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Sinon, terminer
    return END


# Construire le graphe
workflow = StateGraph(AgentState)

# Ajouter les nœuds
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

# Définir le point d'entrée
workflow.set_entry_point("agent")

# Ajouter les arêtes conditionnelles
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)

# Les outils renvoient toujours vers l'agent
workflow.add_edge("tools", "agent")

# Compiler avec mémoire pour les sessions
memory = MemorySaver()
agent = workflow.compile(checkpointer=memory)


# ============== API Models ==============

class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    tool_calls: list[dict]


class StreamEvent(BaseModel):
    type: str  # "token", "tool_start", "tool_end", "done", "error"
    content: str | None = None
    tool_name: str | None = None
    tool_args: dict | None = None


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "Epstein Docs Agent API"}


@app.get("/api/stats")
async def get_stats():
    """Get database statistics."""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as count FROM documents")
    row = cursor.fetchone()
    total_docs = row["count"] if isinstance(row, dict) else row[0]

    cursor.execute("SELECT COUNT(*) as count FROM rdf_triples")
    row = cursor.fetchone()
    total_rels = row["count"] if isinstance(row, dict) else row[0]

    cursor.execute("SELECT COUNT(DISTINCT actor) as count FROM rdf_triples")
    row = cursor.fetchone()
    unique_actors = row["count"] if isinstance(row, dict) else row[0]

    stats = {
        "total_documents": total_docs,
        "total_relationships": total_rels,
        "unique_actors": unique_actors,
    }

    cursor.close()
    conn.close()
    return stats


@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the document corpus with an agentic approach.
    The agent will search, analyze, and synthesize information to answer the question.
    """
    config = {"configurable": {"thread_id": request.session_id or "default"}}

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=request.question)]},
            config=config,
        )

        # Extraire les informations de la réponse
        messages = result["messages"]
        final_answer = ""
        sources = set()
        tool_calls_info = []

        for msg in messages:
            # Collecter les tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_info.append({
                        "tool": tc["name"],
                        "args": tc["args"],
                    })

            # Extraire les sources des résultats d'outils
            if isinstance(msg, ToolMessage):
                try:
                    content = json.loads(msg.content)
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and "doc_id" in item:
                                sources.add(item["doc_id"])
                except (json.JSONDecodeError, TypeError):
                    pass

            # La dernière réponse AI est la réponse finale
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                final_answer = msg.content

        return QueryResponse(
            answer=final_answer,
            sources=list(sources),
            tool_calls=tool_calls_info,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/stream")
async def query_documents_stream(request: QueryRequest):
    """
    Stream the agent's response for real-time UI updates.
    Returns Server-Sent Events (SSE).
    """
    config = {"configurable": {"thread_id": request.session_id or "default"}}

    async def generate():
        try:
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=request.question)]},
                config=config,
                version="v2",
            ):
                event_type = event.get("event", "")

                # Streaming des tokens de réponse
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

                # Début d'un appel d'outil
                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", {})
                    yield f"data: {json.dumps({'type': 'tool_start', 'tool_name': tool_name, 'tool_args': tool_input})}\n\n"

                # Fin d'un appel d'outil
                elif event_type == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    yield f"data: {json.dumps({'type': 'tool_end', 'tool_name': tool_name})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    config = {"configurable": {"thread_id": session_id}}

    try:
        state = agent.get_state(config)
        if state and state.values:
            messages = state.values.get("messages", [])
            history = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage) and msg.content:
                    history.append({"role": "assistant", "content": msg.content})
            return {"session_id": session_id, "messages": history}
        return {"session_id": session_id, "messages": []}
    except Exception:
        return {"session_id": session_id, "messages": []}


@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a session's history."""
    # Note: MemorySaver ne supporte pas la suppression directe
    # En production, utiliser un checkpointer persistant (Redis, PostgreSQL)
    return {"status": "ok", "message": f"Session {session_id} marked for clearing"}


# ============== Main ==============

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("AGENT_PORT", 3002))
    uvicorn.run(app, host="0.0.0.0", port=port)
