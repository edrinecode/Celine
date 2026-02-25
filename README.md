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
- Health check: `http://localhost:8000/health`

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

## Post-deploy smoke test checklist

Once logs show `Application startup complete` and `Uvicorn running`, run a quick validation:

```bash
# 1) Health endpoint should return HTTP 200 with {"ok": true, ...}
curl -sS https://<your-deployment-url>/health

# 2) Basic chat call should return JSON with `response`
curl -sS https://<your-deployment-url>/chat \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id":"smoke-1","message":"I have a mild cough for 2 days"}'

# 3) High-risk prompt should trigger handoff
curl -sS https://<your-deployment-url>/chat \
  -H 'Content-Type: application/json' \
  -d '{"conversation_id":"smoke-2","message":"I have chest pain and severe shortness of breath"}'
```

Then manually verify:

1. Chat UI loads at `/`.
2. Admin queue at `/admin` shows any new high-acuity handoff tickets.
3. You can resolve a ticket from `/admin`.

## About the `google.generativeai` deprecation warning

If you see:

- `FutureWarning: All support for the google.generativeai package has ended ...`

your app is still running, and this is not a startup failure. It comes from upstream dependencies used by `langchain-google-genai`.

Near-term options:

1. Keep running as-is while monitoring updates to `langchain-google-genai`.
2. Plan a migration path to the `google.genai` SDK once your LangChain stack fully supports it.

## Notes

- If `GOOGLE_API_KEY` is not available, agents return deterministic fallback summaries so local development still works.
- This project is a foundation. You can next add:
  - clinician identity, RBAC, audit logging
  - FHIR/EMR integrations
  - retrieval + clinical guideline tool-calling pipelines
