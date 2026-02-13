#!/usr/bin/env python3
"""
Test script for the LangGraph agent.
Run this to verify the agent works before starting the API server.
"""

import asyncio
import os

from dotenv import load_dotenv
load_dotenv()

# Vérifier que la clé API est définie
if not os.environ.get("OPENROUTER_API_KEY"):
    print("⚠️  OPENROUTER_API_KEY non définie")
    print("   export OPENROUTER_API_KEY='sk-or-...'")
    exit(1)

from langchain_core.messages import HumanMessage

# Import de l'agent
from agent_api import agent, tools


def test_tools():
    """Test que les outils fonctionnent."""
    print("\n📦 Test des outils...")

    # Test search_documents
    print("  - search_documents...", end=" ")
    result = tools[0].invoke({"query": "flight logs", "limit": 3})
    assert result and len(result) > 0
    print("✅")

    # Test search_actor
    print("  - search_actor...", end=" ")
    result = tools[1].invoke({"actor_name": "Ghislaine Maxwell", "limit": 5})
    assert result and len(result) > 0
    print("✅")

    # Test get_database_stats
    print("  - get_database_stats...", end=" ")
    result = tools[4].invoke({})
    assert "total_documents" in result
    print("✅")

    print("✅ Tous les outils fonctionnent!")


def test_agent_simple():
    """Test basique de l'agent."""
    print("\n🤖 Test de l'agent (question simple)...")

    result = agent.invoke(
        {"messages": [HumanMessage(content="Combien de documents y a-t-il dans la base?")]},
        config={"configurable": {"thread_id": "test-1"}},
    )

    # Vérifier qu'on a une réponse
    messages = result["messages"]
    assert len(messages) > 0

    # La dernière message devrait être une réponse AI
    last_msg = messages[-1]
    assert hasattr(last_msg, "content") and last_msg.content

    print(f"  Réponse: {last_msg.content[:200]}...")
    print("✅ Agent fonctionne!")


def test_agent_search():
    """Test de l'agent avec recherche."""
    print("\n🔍 Test de l'agent (avec recherche)...")

    result = agent.invoke(
        {"messages": [HumanMessage(content="Quelles sont les relations documentées entre Epstein et Bill Clinton?")]},
        config={"configurable": {"thread_id": "test-2"}},
    )

    messages = result["messages"]

    # Compter les appels d'outils
    tool_calls = sum(1 for m in messages if hasattr(m, "tool_calls") and m.tool_calls)
    print(f"  Nombre d'appels d'outils: {tool_calls}")

    # Afficher la réponse finale
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
            print(f"  Réponse: {msg.content[:300]}...")
            break

    print("✅ Recherche fonctionne!")


async def test_streaming():
    """Test du streaming."""
    print("\n📡 Test du streaming...")

    events_count = 0
    tokens = []

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content="Qui est Ghislaine Maxwell?")]},
        config={"configurable": {"thread_id": "test-3"}},
        version="v2",
    ):
        events_count += 1
        if event.get("event") == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                tokens.append(chunk.content)

    print(f"  Événements reçus: {events_count}")
    print(f"  Tokens streamés: {len(tokens)}")
    print(f"  Aperçu: {''.join(tokens[:20])}...")
    print("✅ Streaming fonctionne!")


def main():
    print("=" * 50)
    print("🧪 Tests de l'Agent LangGraph")
    print("=" * 50)

    try:
        test_tools()
        test_agent_simple()
        test_agent_search()
        asyncio.run(test_streaming())

        print("\n" + "=" * 50)
        print("✅ TOUS LES TESTS PASSENT!")
        print("=" * 50)
        print("\nPour lancer le serveur API:")
        print("  python agent_api.py")
        print("\nOu avec uvicorn (hot reload):")
        print("  uvicorn agent_api:app --reload --port 3002")

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
