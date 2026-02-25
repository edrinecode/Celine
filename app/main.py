import os

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .memory import ConversationMemory
from .models import ChatRequest
from .orchestrator import Coordinator
from .prompts import DEFAULT_PROMPTS, PROMPT_KEYS, PROMPT_LABELS
from .storage import SQLiteStore

app = FastAPI(title="Celine Healthcare Multi-Agent")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

store = SQLiteStore(db_path=os.getenv("CELINE_DB_PATH", "data/celine.db"))
memory = ConversationMemory(store=store)
coordinator = Coordinator(memory=memory)


def _initialize_model_setting() -> str:
    model = os.getenv("GOOGLE_MODEL") or store.get_setting("google_model") or "gemini-3-flash-preview"
    os.environ["GOOGLE_MODEL"] = model
    store.set_setting("google_model", model)
    return model


DEFAULT_MODEL = _initialize_model_setting()


def _initialize_prompt_settings() -> dict[str, str]:
    prompts: dict[str, str] = {}
    for key, default_prompt in DEFAULT_PROMPTS.items():
        current = store.get_setting(key) or default_prompt
        store.set_setting(key, current)
        prompts[key] = current
    return prompts


def _apply_prompts_to_agents(prompts: dict[str, str]) -> None:
    coordinator.lead_agent.system_prompt = prompts[PROMPT_KEYS.lead]
    coordinator.triage_agent.system_prompt = prompts[PROMPT_KEYS.triage]
    coordinator.safety_agent.system_prompt = prompts[PROMPT_KEYS.safety]
    coordinator.data_agent.system_prompt = prompts[PROMPT_KEYS.data]
    coordinator.diagnosis_agent.system_prompt = prompts[PROMPT_KEYS.diagnosis]

PROMPT_SETTINGS = _initialize_prompt_settings()
_apply_prompts_to_agents(PROMPT_SETTINGS)


@app.get("/health")
def healthcheck():
    return {"ok": True, "service": app.title}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
def admin(request: Request, conversation_id: str | None = None):
    selected_messages = store.get_messages(conversation_id) if conversation_id else []
    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "tickets": store.list_handoff_tickets(),
            "current_model": store.get_setting("google_model") or DEFAULT_MODEL,
            "selected_conversation_id": conversation_id,
            "selected_messages": selected_messages,
            "prompt_entries": [
                {"key": key, "label": PROMPT_LABELS[key], "value": store.get_setting(key) or DEFAULT_PROMPTS[key]}
                for key in DEFAULT_PROMPTS
            ],
        },
    )




@app.post("/admin/prompt")
def update_prompt(key: str = Form(...), value: str = Form(...)):
    cleaned_key = key.strip()
    cleaned_value = value.strip()

    if cleaned_key not in DEFAULT_PROMPTS:
        return {"ok": False, "error": "Unknown prompt key."}
    if not cleaned_value:
        return {"ok": False, "error": "Prompt cannot be empty."}

    store.set_setting(cleaned_key, cleaned_value)
    _apply_prompts_to_agents({k: store.get_setting(k) or DEFAULT_PROMPTS[k] for k in DEFAULT_PROMPTS})
    return RedirectResponse(url="/admin", status_code=303)

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


@app.post("/admin/reply")
def admin_reply(conversation_id: str = Form(...), message: str = Form(...)):
    cleaned = message.strip()
    if cleaned:
        memory.add_message(conversation_id, "human", cleaned)
    return RedirectResponse(url=f"/admin?conversation_id={conversation_id}", status_code=303)


@app.get("/chat/history/{conversation_id}")
def chat_history(conversation_id: str):
    messages = memory.get_messages(conversation_id)
    return {"conversation_id": conversation_id, "messages": messages}


@app.post("/admin/model")
def update_model(model: str = Form(...)):
    cleaned = model.strip()
    if not cleaned:
        return {"ok": False, "error": "Model cannot be empty."}

    store.set_setting("google_model", cleaned)
    os.environ["GOOGLE_MODEL"] = cleaned
    return {"ok": True, "model": cleaned}
