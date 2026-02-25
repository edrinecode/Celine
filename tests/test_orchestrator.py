from fastapi.testclient import TestClient
from pathlib import Path

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
