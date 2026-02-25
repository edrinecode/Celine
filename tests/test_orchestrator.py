from pathlib import Path

from fastapi.testclient import TestClient

from app.orchestrator import DeterministicOrchestrator
from app.storage import SQLiteStore


def build_orchestrator(tmp_path: Path) -> DeterministicOrchestrator:
    return DeterministicOrchestrator(SQLiteStore(db_path=str(tmp_path / "triage.db"), encryption_key="test-key"))


def run_turns(orchestrator: DeterministicOrchestrator, conversation_id: str, turns: list[str]):
    result = None
    for turn in turns:
        result = orchestrator.process(conversation_id, "p1", turn)
    return result


def test_chest_pain_red_flag_emergency(tmp_path):
    orchestrator = build_orchestrator(tmp_path)
    result = orchestrator.process("c1", "p1", "I have chest pain and I can't breathe")
    assert result.response.urgency_level == "EMERGENCY"
    assert result.session.state.value == "CLOSED"
    assert "immediate medical care" in result.response.response.lower()


def test_stroke_symptoms_red_flag_override(tmp_path):
    orchestrator = build_orchestrator(tmp_path)
    result = orchestrator.process("c2", "p2", "My mom has slurred speech and one-sided weakness")
    assert result.response.requires_handoff is True
    assert "Emergency" in (result.response.handoff_reason or "")


def test_pediatric_fever_emergency(tmp_path):
    orchestrator = build_orchestrator(tmp_path)
    run_turns(orchestrator, "c3", ["I need symptom help", "8 months old"])
    result = orchestrator.process("c3", "p3", "high fever now")
    assert result.response.urgency_level == "EMERGENCY"


def test_allergic_reaction_emergency(tmp_path):
    orchestrator = build_orchestrator(tmp_path)
    result = orchestrator.process("c4", "p4", "throat swelling after peanut exposure")
    assert result.response.urgency_level == "EMERGENCY"


def test_minor_cold_routine_path(tmp_path):
    orchestrator = build_orchestrator(tmp_path)
    result = run_turns(
        orchestrator,
        "c5",
        [
            "I have mild cough",
            "32",
            "male",
            "mild cough and runny nose",
            "2 days",
            "3",
            "sore throat",
            "none",
            "none",
            "none",
        ],
    )
    assert result.response.urgency_level == "ROUTINE"
    assert "book a standard appointment" in result.response.response.lower()


def test_abdominal_pain_pregnancy_urgent(tmp_path):
    orchestrator = build_orchestrator(tmp_path)
    result = run_turns(
        orchestrator,
        "c6",
        [
            "abdominal pain",
            "29",
            "female",
            "yes pregnant",
            "abdominal pain",
            "1 day",
            "7",
            "nausea",
            "none",
            "prenatal vitamins",
            "none",
        ],
    )
    assert result.response.urgency_level == "URGENT"
    assert result.response.requires_handoff is True


def test_greeting_only_conversation(tmp_path):
    orchestrator = build_orchestrator(tmp_path)
    result = orchestrator.process("c7", "p7", "hello")
    assert result.response.state.value == "GREETING"
    assert "triage assistant" in result.response.response.lower()


def test_identity_question_in_idle_routes_without_intent_dependency(tmp_path):
    orchestrator = build_orchestrator(tmp_path)
    result = orchestrator.process("c7a", "p7", "what is your name")
    assert result.response.state.value == "GREETING"
    assert "i am celine" in result.response.response.lower()

def test_identity_question_after_greeting_gets_specific_identity_response(tmp_path):
    orchestrator = build_orchestrator(tmp_path)
    run_turns(orchestrator, "c7b", ["hello"])
    result = orchestrator.process("c7b", "p7", "what ur name")
    assert "i am celine" in result.response.response.lower()


def test_unclear_follow_up_after_greeting_gets_clarifying_prompt(tmp_path):
    orchestrator = build_orchestrator(tmp_path)
    run_turns(orchestrator, "c7c", ["hello"])
    result = orchestrator.process("c7c", "p7", "ok")
    assert "what do you need help with" in result.response.response.lower()


def test_time_question_gets_direct_answer(tmp_path):
    orchestrator = build_orchestrator(tmp_path)
    run_turns(orchestrator, "c7d", ["hello"])
    result = orchestrator.process("c7d", "p7", "whats the time")
    assert "it is" in result.response.response.lower()


def test_services_question_gets_specific_capabilities(tmp_path):
    orchestrator = build_orchestrator(tmp_path)
    run_turns(orchestrator, "c7e", ["hello"])
    result = orchestrator.process("c7e", "p7", "what services do you offer")
    assert "symptom triage" in result.response.response.lower()
    assert "appointment" in result.response.response.lower()


def test_robotic_feedback_gets_explanation(tmp_path):
    orchestrator = build_orchestrator(tmp_path)
    run_turns(orchestrator, "c7f", ["hello"])
    result = orchestrator.process("c7f", "p7", "why are you robotic")
    assert "deterministic" in result.response.response.lower()


def test_one_question_at_a_time_enforced(tmp_path):
    orchestrator = build_orchestrator(tmp_path)
    result = orchestrator.process("c8", "p8", "I have a headache")
    assert result.response.response.endswith("?")
    assert "and" not in result.response.response.lower().split("?")[0]


def test_logging_integrity_and_export(tmp_path):
    store = SQLiteStore(db_path=str(tmp_path / "api.db"), encryption_key="test-key")
    orchestrator = DeterministicOrchestrator(store)
    orchestrator.process("c9", "p9", "hello")
    events = store.get_audit_events("c9")
    assert len(events) > 0
    assert all("agent" in event and "action" in event for event in events)


def test_api_health_and_session_endpoint():
    from app.main import app
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["safe_mode"] == "deterministic"
