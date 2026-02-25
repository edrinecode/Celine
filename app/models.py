from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TriageState(str, Enum):
    IDLE = "IDLE"
    GREETING = "GREETING"
    INTAKE = "INTAKE"
    TRIAGE = "TRIAGE"
    EMERGENCY = "EMERGENCY"
    ESCALATED = "ESCALATED"
    CLOSED = "CLOSED"


class UrgencyLevel(str, Enum):
    EMERGENCY = "EMERGENCY"
    URGENT = "URGENT"
    ROUTINE = "ROUTINE"


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatRequest(BaseModel):
    conversation_id: str
    message: str
    patient_id: str | None = None


class AuditEvent(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent: str
    action: str
    details: dict[str, Any] = Field(default_factory=dict)


class Demographics(BaseModel):
    age: int | None = None
    sex: str | None = None
    pregnancy_status: str | None = None


class TriageSession(BaseModel):
    session_id: str
    patient_id: str
    state: TriageState = TriageState.IDLE
    demographics: Demographics = Field(default_factory=Demographics)
    chief_complaint: str = ""
    symptoms: list[str] = Field(default_factory=list)
    onset_time: str | None = None
    severity: int | None = None
    associated_symptoms: list[str] = Field(default_factory=list)
    chronic_conditions: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    intake_progress: dict[str, bool] = Field(
        default_factory=lambda: {
            "age": False,
            "sex": False,
            "pregnancy_status": False,
            "chief_complaint": False,
            "onset_time": False,
            "severity": False,
            "associated_symptoms": False,
            "chronic_conditions": False,
            "medications": False,
            "allergies": False,
        }
    )
    red_flags_detected: list[str] = Field(default_factory=list)
    urgency_level: str = ""
    risk_score: float = 0.0
    recommended_action: str = ""
    triggered_rules: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    audit_log: list[AuditEvent] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IntentResult(BaseModel):
    intent: str
    confidence: float


class RuleEngineResult(BaseModel):
    urgency_level: str
    triggered_rules: list[str] = Field(default_factory=list)
    confidence: float


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    state: TriageState
    requires_handoff: bool = False
    handoff_reason: str | None = None
    urgency_level: str = ""
    disclaimer: str = "This is a triage support tool and not a medical diagnosis."


class HandoffTicket(BaseModel):
    ticket_id: str
    conversation_id: str
    reason: str
    user_message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
