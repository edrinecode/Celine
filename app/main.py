import os

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .memory import ConversationMemory
from .models import ChatRequest
from .orchestrator import Coordinator
from .storage import SQLiteStore

app = FastAPI(title="Celine Healthcare Multi-Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

store = SQLiteStore(db_path=os.getenv("CELINE_DB_PATH", "data/celine.db"))
memory = ConversationMemory(store=store)
coordinator = Coordinator(memory=memory)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
def admin(request: Request):
    return templates.TemplateResponse(
        "admin.html",
        {"request": request, "tickets": store.list_handoff_tickets()},
    )


@app.post("/chat")
def chat(chat_request: ChatRequest):
    result = coordinator.process(chat_request.conversation_id, chat_request.message)
    if result.handoff_ticket:
        store.add_handoff_ticket(result.handoff_ticket)
    return result.response


@app.post("/admin/resolve")
def resolve_ticket(ticket_id: str = Form(...)):
    remaining = store.resolve_handoff_ticket(ticket_id=ticket_id)
    return {"ok": True, "remaining": remaining}
