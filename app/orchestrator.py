from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .agents import (
    ClinicalRulesEngine,
    EscalationAgent,
    FrontDeskAgent,
    IntentClassificationAgent,
    RedFlagEngine,
    RiskScoringAgent,
    TriageAgent,
)
from .models import AuditEvent, ChatResponse, TriageSession, TriageState, UrgencyLevel
from .storage import SQLiteStore


@dataclass
class OrchestrationResult:
    response: ChatResponse
    handoff_ticket: dict | None
    session: TriageSession


class DeterministicOrchestrator:
    MIN_INTENT_CONFIDENCE = 0.55
    IDENTITY_PATTERNS = re.compile(
        r"\b(who\s+are\s+you|what('s|\s+is)\s+your\s+name|ur\s+name|your\s+name)\b", re.IGNORECASE
    )
    IDENTITY_REPLY = (
        "I am Celine, the hospital triage assistant. I can help with symptom triage or route front-desk requests."
    )

    def __init__(self, store: SQLiteStore, rules_path: str = "app/config/clinical_rules.json") -> None:
        self.store = store
        self.intent_agent = IntentClassificationAgent()
        self.front_desk = FrontDeskAgent()
        self.triage_agent = TriageAgent()
        self.red_flag_engine = RedFlagEngine()
        rules = json.loads(Path(rules_path).read_text())
        self.rules_engine = ClinicalRulesEngine(rules)
        self.risk_agent = RiskScoringAgent()
        self.escalation_agent = EscalationAgent()

    def process(self, conversation_id: str, patient_id: str | None, user_message: str) -> OrchestrationResult:
        session = self.store.get_or_create_session(conversation_id=conversation_id, patient_id=patient_id or conversation_id)
        self.store.add_message(conversation_id, "user", user_message, datetime.now(timezone.utc))
        self._log(session, "orchestrator", "message_received", {"message": user_message, "state": session.state.value})

        red_flags = self.red_flag_engine.detect(user_message, session)
        if red_flags:
            session.red_flags_detected.extend([flag for flag in red_flags if flag not in session.red_flags_detected])
            self._transition(session, TriageState.EMERGENCY, "red_flag_override", {"red_flags": red_flags})
            session.urgency_level = UrgencyLevel.EMERGENCY.value
            session.triggered_rules.extend(red_flags)
            message, requires_handoff, reason = self.escalation_agent.handoff_message(UrgencyLevel.EMERGENCY.value)
            self._transition(session, TriageState.ESCALATED, "emergency_handoff", {"reason": reason})
            self._transition(session, TriageState.CLOSED, "session_closed_after_emergency", {})
            return self._finalize(conversation_id, session, message, requires_handoff, reason)

        if session.state == TriageState.IDLE:
            if self._is_identity_question(user_message):
                self._transition(session, TriageState.GREETING, "identity_route", {})
                return self._finalize(conversation_id, session, self.IDENTITY_REPLY, False, None)

            intent = self.intent_agent.classify(user_message)
            self._log(session, "intent_classifier", "classified", intent.model_dump())
            if intent.confidence < self.MIN_INTENT_CONFIDENCE:
                self._transition(session, TriageState.ESCALATED, "low_classifier_confidence", {"confidence": intent.confidence})
                msg = "I need a human clinician to review this safely before proceeding."
                return self._finalize(conversation_id, session, msg, True, "Low intent confidence")

            if intent.intent == "greeting":
                self._transition(session, TriageState.GREETING, "greeting_route", {})
                msg = self.front_desk.respond("greeting")
                return self._finalize(conversation_id, session, msg, False, None)
            if intent.intent in {"appointment_request", "admin_question"}:
                self._transition(session, TriageState.GREETING, "front_desk_route", {"intent": intent.intent})
                msg = self.front_desk.respond(intent.intent)
                return self._finalize(conversation_id, session, msg, False, None)
            if intent.intent == "medical_symptom":
                self._transition(session, TriageState.INTAKE, "medical_route", {})
            else:
                msg = "Could you clarify if you need symptom triage, appointment help, or another front-desk service?"
                return self._finalize(conversation_id, session, msg, False, None)

        if session.state in {TriageState.GREETING, TriageState.IDLE}:
            if self._is_identity_question(user_message):
                return self._finalize(conversation_id, session, self.IDENTITY_REPLY, False, None)

            intent = self.intent_agent.classify(user_message)
            if intent.intent == "medical_symptom":
                self._transition(session, TriageState.INTAKE, "symptom_detected_post_greeting", {})
            else:
                msg = self.front_desk.respond(intent.intent)
                return self._finalize(conversation_id, session, msg, False, None)

        if session.state in {TriageState.INTAKE, TriageState.TRIAGE}:
            self.triage_agent.update_from_user(session, user_message)
            self._transition(session, TriageState.TRIAGE, "triage_progress", {"progress": session.intake_progress})
            next_question = self.triage_agent.next_pending_question(session)
            if next_question:
                return self._finalize(conversation_id, session, next_question.question, False, None)

            rules = self.rules_engine.evaluate(session)
            session.urgency_level = rules.urgency_level
            session.triggered_rules = rules.triggered_rules
            session.confidence = rules.confidence
            risk_score, risk_confidence = self.risk_agent.score(session)
            session.risk_score = risk_score
            self._log(session, "rules_engine", "urgency_classified", rules.model_dump())
            self._log(session, "risk_scoring_agent", "supplemental_risk", {"risk_score": risk_score, "confidence": risk_confidence})

            if risk_score >= 0.85 and session.urgency_level != UrgencyLevel.EMERGENCY.value:
                self._transition(session, TriageState.ESCALATED, "risk_uncertain_escalation", {"risk_score": risk_score})
                return self._finalize(
                    conversation_id,
                    session,
                    "Your case needs human clinician review now for safety.",
                    True,
                    "High risk score uncertainty",
                )

            msg, requires_handoff, reason = self.escalation_agent.handoff_message(session.urgency_level)
            if requires_handoff:
                self._transition(session, TriageState.ESCALATED, "urgency_handoff", {"urgency": session.urgency_level})
            self._transition(session, TriageState.CLOSED, "triage_completed", {"urgency": session.urgency_level})
            return self._finalize(conversation_id, session, msg, requires_handoff, reason)

        fallback = "Seek immediate medical care. Call emergency services or go to nearest emergency department."
        self._transition(session, TriageState.EMERGENCY, "failsafe_default", {})
        self._transition(session, TriageState.CLOSED, "failsafe_closed", {})
        return self._finalize(conversation_id, session, fallback, True, "Failsafe default")

    def _is_identity_question(self, message: str) -> bool:
        return bool(self.IDENTITY_PATTERNS.search(message.strip()))

    def _transition(self, session: TriageSession, new_state: TriageState, reason: str, details: dict) -> None:
        old_state = session.state
        session.state = new_state
        self._log(
            session,
            "orchestrator",
            "state_transition",
            {"from": old_state.value, "to": new_state.value, "reason": reason, **details},
        )

    @staticmethod
    def _log(session: TriageSession, agent: str, action: str, details: dict) -> None:
        session.audit_log.append(AuditEvent(agent=agent, action=action, details=details))
        session.timestamp = datetime.now(timezone.utc)

    def _finalize(
        self,
        conversation_id: str,
        session: TriageSession,
        assistant_message: str,
        requires_handoff: bool,
        handoff_reason: str | None,
    ) -> OrchestrationResult:
        self.store.save_session(session)
        self.store.add_message(conversation_id, "assistant", assistant_message, datetime.now(timezone.utc))

        response = ChatResponse(
            conversation_id=conversation_id,
            response=assistant_message,
            state=session.state,
            requires_handoff=requires_handoff,
            handoff_reason=handoff_reason,
            urgency_level=session.urgency_level,
        )

        ticket = None
        if requires_handoff:
            ticket = {
                "ticket_id": str(uuid4()),
                "conversation_id": conversation_id,
                "reason": handoff_reason or "Clinical escalation",
                "user_message": session.chief_complaint or "See session log",
            }

        return OrchestrationResult(response=response, handoff_ticket=ticket, session=session)
