# MediaMind QA

MediaMind QA is a full-stack document and multimedia question-answering application built for the SDE-1 programming assignment. It ingests PDFs, audio, and video, extracts or transcribes their contents, stores indexed chunks and metadata, and answers grounded questions with summaries, timestamp matches, and clip playback links.

## What is implemented

- PDF, audio, and video upload from a React frontend
- Automatic PDF extraction and multimedia transcription
- Per-file summaries and grounded Q&A
- Timestamp/topic lookup for audio and video
- In-browser playback jump to the relevant timestamp
- JWT-based multi-user authentication
- Rotatable API keys for direct API access
- Vector-backed retrieval with FAISS support and an in-memory fallback
- Streaming chat responses over an NDJSON event stream
- Redis-ready caching and rate limiting with an in-memory fallback for local development
- Docker, Docker Compose, and GitHub Actions CI
- Backend test coverage above the assignment target

## Stack

- Backend: FastAPI, SQLAlchemy, Pydantic Settings
- Frontend: React, TypeScript, Vite
- Database: PostgreSQL in Docker, SQLite for local development
- AI/ASR: OpenAI-compatible chat, embeddings, and Whisper transcription
- Vector search: FAISS
- Cache and rate limiting: Redis

## Project layout

```text
backend/
  app/          FastAPI application
  tests/        Backend test suite
frontend/
  src/          React application
.github/
  workflows/    CI pipeline
docker-compose.yml
.env.example
README.md
```

## Quick start

### Docker Compose

This is the easiest way to run the full stack.

```bash
docker compose up --build
```

Endpoints after startup:

- App UI: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Direct API port: `http://localhost:8001`

The browser should be opened on `http://localhost:8000`, not `http://localhost:8001`.

### Local development

Backend:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e "backend[dev]"
uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

Frontend dev runs on `http://localhost:5173`.

## Authentication

- Create a new user directly in the UI using the built-in register form.
- There is no seeded default account; choose any username and password.
- The backend issues a JWT access token after registration or login.
- Users can rotate an API key from the UI and use it through the `X-API-Key` header.
- Uploaded assets are scoped per user.

Example API registration:

```bash
curl -X POST http://localhost:8001/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"newuser","password":"strongpassword"}'
```

Example API login:

```bash
curl -X POST http://localhost:8001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"newuser","password":"strongpassword"}'
```

Example authenticated asset list:

```bash
curl http://localhost:8001/api/assets \
  -H "Authorization: Bearer <token>"
```

## Environment variables

Copy `.env.example` to `.env` and adjust values as needed.

```bash
AI_PROVIDER=auto
OPENAI_API_KEY=
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_TRANSCRIPTION_MODEL=whisper-1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DATABASE_URL=sqlite:///./data/app.db
STORAGE_DIR=./data
REDIS_URL=redis://localhost:6379/0
AUTH_SECRET_KEY=change-me-for-shared-environments
CORS_ORIGINS=http://localhost:8000,http://127.0.0.1:8000,http://localhost:5173,http://127.0.0.1:5173
```

Notes:

- `AI_PROVIDER=auto` uses OpenAI when `OPENAI_API_KEY` is set and falls back to a deterministic mock client otherwise.
- **Summarization with BART**: Set `USE_BART_FOR_SUMMARIZATION=true` to use local BART models instead of OpenAI. Choose a model: `facebook/bart-large-cnn` (default, ~1.6GB), `facebook/bart-base-cnn` (smaller). First run downloads the model automatically.
- PDF extraction works without OpenAI, but audio/video transcription and real OpenAI-powered embeddings/chat require `OPENAI_API_KEY`.
- If OpenAI credentials are unavailable, media files cannot be transcribed/indexed and may fail at upload.
- If Redis is not reachable, the app falls back to in-memory caching and rate limiting.
- `CORS_ORIGINS` accepts a comma-separated string, which now works correctly in Docker Compose.

## API summary

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/auth/api-key`
- `GET /api/assets`
- `POST /api/assets/upload`
- `GET /api/assets/{asset_id}`
- `GET /api/assets/{asset_id}/content`
- `POST /api/assets/{asset_id}/topics`
- `POST /api/chat`
- `POST /api/chat/stream`

`/api/chat/stream` returns newline-delimited JSON events of the form:

- `{"type":"answer_delta","delta":"..."}`
- `{"type":"metadata","sources":[...],"timestamp_matches":[...]}`
- `{"type":"done"}`

## Testing

Backend tests with coverage:

```bash
./.venv/bin/python -m pytest backend/tests -q
```

Current verified result:

- `54` tests passing
- `98.25%` backend coverage

Frontend production build:

```bash
cd frontend
npm run build
```

## Assignment status

Completed in this repository:

- Core document and multimedia Q&A workflow
- Timestamp extraction and playback jump support
- Vector search
- Streaming chat responses
- Multi-user authentication
- Rate limiting and caching
- Docker, CI, tests, and documentation

Still manual outside the repo:

- Cloud deployment is not configured in this repository and is optional.
- Live demo URL
- Walkthrough recording upload
