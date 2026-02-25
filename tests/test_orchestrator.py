import importlib
from pathlib import Path

from fastapi.testclient import TestClient

from app.memory import ConversationMemory
from app.orchestrator import Coordinator
from app.storage import SQLiteStore


def _build_coordinator(tmp_path: Path) -> Coordinator:
    store = SQLiteStore(db_path=str(tmp_path / "test.db"))
    return Coordinator(memory=ConversationMemory(store=store))


def test_handoff_for_high_risk_signal(tmp_path):
    coordinator = _build_coordinator(tmp_path)
    result = coordinator.process("abc", "I have chest pain and shortness of breath")
    assert result.response.requires_handoff is True
    assert result.handoff_ticket is not None


def test_no_handoff_for_routine_request(tmp_path):
    coordinator = _build_coordinator(tmp_path)
    result = coordinator.process("abc2", "I need advice for mild seasonal allergies")
    assert "Celine Multi-Agent Summary" in result.response.response


def test_no_handoff_for_simple_greeting(tmp_path):
    coordinator = _build_coordinator(tmp_path)
    result = coordinator.process("abc3", "hey there")
    assert result.response.requires_handoff is False


def test_healthcheck_endpoint():
    from app.main import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_agent_falls_back_when_llm_call_fails(monkeypatch):
    from app import agents

    class BrokenLLM:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, *args, **kwargs):
            raise RuntimeError("provider unavailable")

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(agents, "ChatGoogleGenerativeAI", BrokenLLM)

    result = agents.TriageAgent().run("headache", [])

    assert "[Triage Agent fallback]" in result.summary


def test_admin_model_update(monkeypatch, tmp_path):
    monkeypatch.setenv("CELINE_DB_PATH", str(tmp_path / "admin.db"))
    monkeypatch.delenv("GOOGLE_MODEL", raising=False)

    import app.main as main_module

    main_module = importlib.reload(main_module)
    client = TestClient(main_module.app)

    admin_page = client.get("/admin")
    assert admin_page.status_code == 200
    assert "gemini-3-flash-preview" in admin_page.text

    update = client.post("/admin/model", data={"model": "gemini-2.5-pro"})
    assert update.status_code == 200
    assert update.json()["ok"] is True
    assert update.json()["model"] == "gemini-2.5-pro"

    refreshed_page = client.get("/admin")
    assert "gemini-2.5-pro" in refreshed_page.text


def test_human_handoff_chat_flow(monkeypatch, tmp_path):
    monkeypatch.setenv("CELINE_DB_PATH", str(tmp_path / "handoff-chat.db"))

    import app.main as main_module

    main_module = importlib.reload(main_module)
    client = TestClient(main_module.app)

    conversation_id = "conv-1"
    chat_response = client.post(
        "/chat",
        json={"conversation_id": conversation_id, "message": "I have chest pain now"},
    )
    assert chat_response.status_code == 200
    assert chat_response.json()["requires_handoff"] is True

    reply = client.post(
        "/admin/reply",
        data={"conversation_id": conversation_id, "message": "A clinician has joined this chat."},
        follow_redirects=True,
    )
    assert reply.status_code == 200
    assert "A clinician has joined this chat." in reply.text

    history = client.get(f"/chat/history/{conversation_id}")
    assert history.status_code == 200
    roles = [item["role"] for item in history.json()["messages"]]
    assert "human" in roles
