# Epstein Docs Agent API

API LangGraph pour explorer les documents Epstein avec un agent RAG.

## Stack

- **FastAPI** - API REST
- **LangGraph** - Agent conversationnel
- **sentence-transformers** - Embeddings pour recherche sémantique
- **SQLite** - Base de données avec embeddings

## Installation locale

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Variables d'environnement

```
OPENROUTER_API_KEY=your_key
OPENROUTER_MODEL=anthropic/claude-sonnet-4  # optionnel
```

## Lancer le serveur

```bash
uvicorn agent_api:app --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /` - Health check
- `GET /api/stats` - Statistiques de la base
- `POST /api/query` - Interroger l'agent
- `POST /api/query/stream` - Streaming SSE
- `GET /api/sessions/{id}/history` - Historique conversation

## Déploiement

Configuré pour Railway/Render avec `nixpacks.toml` et `Procfile`.
