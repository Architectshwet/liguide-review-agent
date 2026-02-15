# Liquide Review RAG Agent

A multi-turn conversational AI agent for app store review insights, built with LangGraph, LangChain, and Retrieval-Augmented Generation (RAG).

## Features

- **LangGraph + FastAPI Streaming**: Multi-turn conversations with real-time SSE token-by-token streaming.
- **Session-based Memory**: Persistent conversation history via PostgreSQL checkpointer.
- **Embedding-based RAG**: Semantic search over app store reviews with Qdrant + OpenAI embeddings.
- **Single Agent Architecture**: `LiquideAgent` with a specialized review RAG tool; the tool is invoked only when relevant.
- **Review Ingestion**: Sample dataset or live fetch from Google Play and App Store; vector store is cleared on each ingest.
- **Background Live Ingest Jobs**: Large live ingests can run asynchronously with `job_id` tracking and status polling.
- **Web Chat UI**: Built-in interface at `/web` with token-by-token streaming.
- **Docker + PostgreSQL**: Containerized dev setup with persistent database.

## Quick Start (Docker)

### 1. Setup Environment

```bash
cd liquide_chatbot
cp .env.example .env
# Edit .env with OPENAI_API_KEY (and SERPAPI_API_KEY for live ingest)
```

### 2. Build and Run

```bash
docker-compose -f docker-compose.dev.yml up --build
```

Rebuilds the image when `pyproject.toml` or the Dockerfile changes, then starts the stack.

### 3. Ingest Reviews

Sample ingest:

```bash
curl "http://localhost:8000/reviews/ingest/sample"
```

Live ingest (background job, recommended for larger `max_pages`):

```bash
curl -X POST "http://localhost:8000/reviews/ingest/live" \
  -H "Content-Type: application/json" \
  -d '{"max_pages": 20, "fallback_to_sample": true, "run_in_background": true}'
```

This returns a `job_id`. Poll job status:

```bash
curl "http://localhost:8000/reviews/ingest/live/jobs/<job_id>"
```

Live ingest (blocking mode, optional):

```bash
curl -X POST "http://localhost:8000/reviews/ingest/live" \
  -H "Content-Type: application/json" \
  -d '{"max_pages": 5, "fallback_to_sample": true, "run_in_background": false}'
```

### 4. Chat

- **Web UI**: http://localhost:8000/web

### 5. Stop

```bash
docker-compose -f docker-compose.dev.yml down
```

To start again after `down`, run `docker-compose -f docker-compose.dev.yml up`.  
Use `--build` only when you changed dependencies or the Dockerfile.

Do not use `down -v` if you want to keep persisted Docker volumes.  
Qdrant data is also removed if you delete the local `qdrant_vectordb` folder.

## Architecture

- **Single Agent** (`LiquideAgent`) with `QueryReviewRAG` for semantic search over indexed reviews.
- **Streaming Chat** (`/chat/stream`) emits SSE events including `session`, `tool_start`, `tool_end`, `token`, `say`, and `end_of_response`.
- **Live Ingest Background Service** (`liquide/services/live_ingest_background_service.py`) manages long-running live ingest jobs (`queued`, `running`, `ok`, `fallback_to_sample`, `failed`).
- **Storage Layer**:
  - Qdrant for embeddings and filtered retrieval (local `qdrant_vectordb`).
  - PostgreSQL for checkpointer and conversation history.
- **Ingest Pipeline**:
  - Fetch reviews from store APIs.
  - Merge live data with bundled sample data.
  - Generate embeddings.
  - Upsert into Qdrant (collection is reset before ingest).

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /web` | Web chat interface |
| `POST /chat/stream` | Streaming chat |
| `GET /reviews/ingest/sample` | Ingest sample data (clears store) |
| `POST /reviews/ingest/live` | Ingest live data from stores (background or blocking mode) |
| `GET /reviews/ingest/live/jobs/{job_id}` | Get background live ingest job status/result |
| `POST /reviews/live-preview` | Preview live reviews |
| `GET /reviews/sample-preview` | Preview sample reviews |
| `GET /reviews/data` | Inspect indexed reviews |
| `GET /health` | Health check |

## Knowledge Base

- `liquide/sample_data/reviews_sample.json`: bundled sample reviews used for local bootstrap and fallback.
- Live reviews from Google Play and Apple App Store (via `/reviews/ingest/live`).

Indexed in Qdrant with metadata used for filtering and evidence generation:
- `rating`
- `device` (`android` / `ios`)
- `country`
- `version`
- `date` and normalized timestamp (`date_ts`)
