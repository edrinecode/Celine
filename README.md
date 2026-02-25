# Celine - Deterministic Hospital Triage Platform

Production-oriented multi-agent triage intake service designed for hospital front-door workflows.

## Safety posture

- Deterministic state machine controls all transitions.
- Red-flag detection runs before every response.
- No diagnosis, no prescribing, no autonomous LLM state control.
- Failsafe defaults to emergency recommendation on uncertain failures.
- Every decision/event is appended to an immutable audit trail.
- Session payloads and audit events are encrypted at rest (Fernet/AES-128 in CBC/HMAC construction).

## Core architecture

- **API Layer**: FastAPI (`app/main.py`)
- **Orchestrator**: deterministic controller (`app/orchestrator.py`)
- **Intent Classification Agent**: rule/pattern based (`app/agents.py`)
- **Front Desk Agent**: non-clinical routing (`app/agents.py`)
- **Triage Agent**: one-question-at-a-time structured intake (`app/agents.py`)
- **Red-Flag Detection Engine**: hard-coded override rules (`app/agents.py`)
- **Clinical Rules Engine**: editable config-based urgency logic (`app/config/clinical_rules.json`)
- **Risk Scoring Agent**: supplemental signal only (`app/agents.py`)
- **Escalation Agent**: emergency/urgent/routine handoff text (`app/agents.py`)
- **Logging/Audit**: encrypted persistent events (`app/storage.py`)
- **Admin dashboard**: queue + session + audit inspection (`templates/admin.html`)

## State machine

Allowed states:

- `IDLE`
- `GREETING`
- `INTAKE`
- `TRIAGE`
- `EMERGENCY`
- `ESCALATED`
- `CLOSED`

Emergency override (`EMERGENCY`) supersedes all agent outputs.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export CELINE_ENCRYPTION_KEY="replace-with-secret"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API highlights

- `POST /chat`: process one user turn through deterministic orchestration
- `GET /session/{conversation_id}`: encrypted session snapshot + audit trail
- `GET /admin`: handoff queue + traceability dashboard
- `GET /health`: service and mode status

## Compliance-ready implementation notes

This starter includes conservative safeguards and traceability controls intended to support HIPAA/GDPR-aligned implementations. For production deployment, add:

- managed PostgreSQL + Redis
- centralized IAM/JWT + RBAC
- key management system (KMS/HSM)
- immutable external log sink (SIEM/WORM)
- formal clinical governance + validation protocols
