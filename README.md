# Celine - AI Healthcare Multi-Agent Starter

Celine is a starter implementation for an **intelligent AI healthcare chat interface** with:

- **Coordinator orchestration**
- Specialized agents: **Triage, Diagnosis, Safety, Data**
- **Memory** for conversation context (persisted in SQLite)
- Basic tools (risk keyword heuristic + timestamps)
- **Human handoff admin queue** for high-acuity escalation
- Gemini via **LangChain** (`langchain-google-genai`) when `GOOGLE_API_KEY` is set

## Architecture

```text
User
  ↓
Coordinator Agent
  ↓
Triage Agent
Diagnosis Agent
Safety Agent
Data Agent
  ↓
Final Aggregated Response
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY=your_google_key_here   # optional for live Gemini calls
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then open:

- Chat UI: `http://localhost:8000`
- Admin handoff: `http://localhost:8000/admin`

## Railway deployment

This project is ready for Railway as a Python web service.

1. Push this repo to GitHub.
2. In Railway, create a new project from GitHub and select this repo.
3. Set environment variables:
   - `GOOGLE_API_KEY` (required for live Gemini calls)
   - `CELINE_DB_PATH=data/celine.db` (optional; default already set)
4. Railway will detect `Procfile` and run:
   - `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Important deployment note

- SQLite persistence is included for message memory and handoff tickets.
- On Railway, SQLite works for starter/single-instance usage.
- For production scale, migrate storage to managed Postgres/Redis for durability and multi-instance safety.

## Notes

- If `GOOGLE_API_KEY` is not available, agents return deterministic fallback summaries so local development still works.
- This project is a foundation. You can next add:
  - clinician identity, RBAC, audit logging
  - FHIR/EMR integrations
  - retrieval + clinical guideline tool-calling pipelines
