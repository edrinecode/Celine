import os
from datetime import datetime, timezone

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .models import ChatRequest, HandoffTicket
from .orchestrator import DeterministicOrchestrator
from .storage import SQLiteStore

app = FastAPI(title="Celine Hospital Triage System")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

store = SQLiteStore(db_path=os.getenv("CELINE_DB_PATH", "data/celine.db"))
orchestrator = DeterministicOrchestrator(store=store)


@app.get("/health")
def healthcheck():
    return {"ok": True, "service": app.title, "safe_mode": "deterministic"}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
def admin(request: Request, conversation_id: str | None = None):
    selected_messages = store.get_messages(conversation_id) if conversation_id else []
    selected_session = store.get_session_snapshot(conversation_id) if conversation_id else None
    selected_audit = store.get_audit_events(conversation_id) if conversation_id else []
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "tickets": store.list_handoff_tickets(),
            "current_model": "Not used (deterministic controller)",
            "selected_conversation_id": conversation_id,
            "selected_messages": selected_messages,
            "selected_session": selected_session,
            "selected_audit": selected_audit,
            "prompt_entries": [],
        },
    )


@app.post("/chat")
def chat(chat_request: ChatRequest):
    result = orchestrator.process(chat_request.conversation_id, chat_request.patient_id, chat_request.message)
    if result.handoff_ticket:
        store.add_handoff_ticket(HandoffTicket(**result.handoff_ticket, created_at=datetime.now(timezone.utc)))
    return result.response


@app.post("/admin/resolve")
def resolve_ticket(ticket_id: str = Form(...)):
    remaining = store.resolve_handoff_ticket(ticket_id=ticket_id)
    return {"ok": True, "remaining": remaining}


@app.post("/admin/reply")
def admin_reply(conversation_id: str = Form(...), message: str = Form(...)):
    cleaned = message.strip()
    if cleaned:
        store.add_message(conversation_id, "human", cleaned, datetime.now(timezone.utc))
    return RedirectResponse(url=f"/admin?conversation_id={conversation_id}", status_code=303)


@app.get("/chat/history/{conversation_id}")
def chat_history(conversation_id: str):
    messages = store.get_messages(conversation_id)
    return {"conversation_id": conversation_id, "messages": messages}


@app.get("/session/{conversation_id}")
def session_snapshot(conversation_id: str):
    snapshot = store.get_session_snapshot(conversation_id)
    audit = store.get_audit_events(conversation_id)
    return {"session": snapshot, "audit_log": audit}
