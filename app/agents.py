from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from .models import IntentResult, RuleEngineResult, TriageSession, UrgencyLevel


class RedFlagEngine:
    RED_FLAG_RULES: dict[str, tuple[str, ...]] = {
        "difficulty_breathing": ("difficulty breathing", "shortness of breath", "can't breathe", "cannot breathe"),
        "chest_pain": ("chest pain",),
        "severe_bleeding": ("severe bleeding", "bleeding heavily", "won't stop bleeding"),
        "loss_of_consciousness": ("loss of consciousness", "passed out", "fainted"),
        "stroke_symptoms": ("face droop", "slurred speech", "one-sided weakness", "stroke"),
        "seizure": ("seizure", "convulsion"),
        "severe_allergic_reaction": ("anaphylaxis", "throat swelling", "severe allergic reaction"),
        "altered_mental_state": ("confused and disoriented", "altered mental state", "not making sense"),
        "high_fever_in_infant": ("infant with fever", "baby fever", "newborn fever"),
        "signs_of_shock": ("cold clammy skin", "weak rapid pulse", "signs of shock", "in shock"),
        "dying_statement": ("i am dying", "i'm dying"),
        "collapsed_statement": ("she collapsed", "he collapsed", "collapsed"),
    }

    def detect(self, message: str, session: TriageSession) -> list[str]:
        msg = message.lower()
        hits: list[str] = []
        for key, phrases in self.RED_FLAG_RULES.items():
            if any(phrase in msg for phrase in phrases):
                hits.append(key)
        # infant fever safety check with demographics/symptom combination
        if session.demographics.age is not None and session.demographics.age < 1 and "fever" in msg:
            hits.append("high_fever_in_infant")
        return sorted(set(hits))


class IntentClassificationAgent:
    GREETING_PATTERNS = re.compile(r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening))\b", re.IGNORECASE)
    APPOINTMENT_PATTERNS = re.compile(r"\b(book|schedule|appointment|follow[- ]?up)\b", re.IGNORECASE)
    ADMIN_PATTERNS = re.compile(r"\b(billing|insurance|hours|location|records)\b", re.IGNORECASE)
    MEDICAL_PATTERNS = re.compile(
        r"\b(pain|fever|cough|rash|vomit|nausea|headache|dizzy|bleeding|pregnan|symptom|breath)\b",
        re.IGNORECASE,
    )
    TIME_PATTERNS = re.compile(r"\b(time|date|today)\b", re.IGNORECASE)
    SERVICES_PATTERNS = re.compile(r"\b(service|services|offer|help with|what can you do)\b", re.IGNORECASE)
    ROBOTIC_PATTERNS = re.compile(r"\b(robotic|bot|human|too scripted)\b", re.IGNORECASE)

    def classify(self, message: str) -> IntentResult:
        text = message.strip()
        if self.GREETING_PATTERNS.search(text):
            return IntentResult(intent="greeting", confidence=0.98)
        if self.TIME_PATTERNS.search(text):
            return IntentResult(intent="time_question", confidence=0.9)
        if self.SERVICES_PATTERNS.search(text):
            return IntentResult(intent="services_question", confidence=0.9)
        if self.ROBOTIC_PATTERNS.search(text):
            return IntentResult(intent="style_feedback", confidence=0.89)
        if self.MEDICAL_PATTERNS.search(text):
            return IntentResult(intent="medical_symptom", confidence=0.88)
        if self.APPOINTMENT_PATTERNS.search(text):
            return IntentResult(intent="appointment_request", confidence=0.85)
        if self.ADMIN_PATTERNS.search(text):
            return IntentResult(intent="admin_question", confidence=0.84)
        return IntentResult(intent="unclear", confidence=0.4)


class FrontDeskAgent:
    def respond(self, intent: str) -> str:
        if intent == "time_question":
            now = datetime.now()
            return f"It is {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d')}."
        if intent == "services_question":
            return (
                "I can help with three things: (1) symptom triage, (2) routing appointment requests, and "
                "(3) front-desk questions like billing, records, hours, or location."
            )
        if intent == "style_feedback":
            return (
                "Good feedback — I am deterministic for clinical safety, so my phrasing can sound structured. "
                "I can still keep replies shorter and more conversational while staying within triage scope."
            )
        if intent == "appointment_request":
            return (
                "I can help route appointment requests. If you also have symptoms, tell me your main symptom so I can start triage safely."
            )
        if intent == "admin_question":
            return (
                "I can help with front-desk support and triage routing. For billing or records, a staff member can assist you directly."
            )
        if intent == "unclear":
            return "I can help with symptom triage, appointments, or admin questions. What do you need help with?"
        return "Hello, I am the hospital triage assistant. Tell me how I can help today."


@dataclass
class IntakeQuestion:
    field: str
    question: str
    relevant: callable | None = None


class TriageAgent:
    QUESTIONS = [
        IntakeQuestion("age", "How old is the patient?"),
        IntakeQuestion("sex", "What sex was assigned at birth?"),
        IntakeQuestion(
            "pregnancy_status",
            "Is the patient currently pregnant or possibly pregnant?",
            relevant=lambda s: (s.demographics.sex or "").lower() in {"female", "f", "woman"},
        ),
        IntakeQuestion("chief_complaint", "What is the main symptom or concern right now?"),
        IntakeQuestion("onset_time", "When did this problem start?"),
        IntakeQuestion("severity", "On a scale of 1 to 10, how severe is it now?"),
        IntakeQuestion("associated_symptoms", "Any other symptoms with it?"),
        IntakeQuestion("chronic_conditions", "Any chronic health conditions?"),
        IntakeQuestion("medications", "Any regular medications? (optional)"),
        IntakeQuestion("allergies", "Any known allergies? (optional)"),
    ]

    def update_from_user(self, session: TriageSession, message: str) -> None:
        pending = self.next_pending_question(session)
        if not pending:
            return
        value = message.strip()
        field = pending.field
        if field == "age":
            digits = re.findall(r"\d+", value)
            if digits:
                parsed_age = int(digits[0])
                if "month" in value.lower() and parsed_age < 12:
                    parsed_age = 0
                session.demographics.age = parsed_age
                session.intake_progress[field] = True
        elif field == "sex":
            session.demographics.sex = value
            session.intake_progress[field] = True
        elif field == "pregnancy_status":
            session.demographics.pregnancy_status = value
            session.intake_progress[field] = True
        elif field == "chief_complaint":
            session.chief_complaint = value
            session.symptoms.append(value)
            session.intake_progress[field] = True
        elif field == "onset_time":
            session.onset_time = value
            session.intake_progress[field] = True
        elif field == "severity":
            digits = re.findall(r"\d+", value)
            if digits:
                session.severity = min(10, max(1, int(digits[0])))
                session.intake_progress[field] = True
        else:
            cleaned = [x.strip() for x in re.split(r",|;| and ", value) if x.strip()]
            setattr(session, field, cleaned)
            session.intake_progress[field] = True

    def next_pending_question(self, session: TriageSession) -> IntakeQuestion | None:
        for question in self.QUESTIONS:
            if session.intake_progress.get(question.field):
                continue
            if question.relevant and not question.relevant(session):
                session.intake_progress[question.field] = True
                continue
            return question
        return None


class ClinicalRulesEngine:
    def __init__(self, config: dict) -> None:
        self.config = config

    def evaluate(self, session: TriageSession) -> RuleEngineResult:
        text_blob = " ".join([session.chief_complaint, *session.symptoms, *session.associated_symptoms]).lower()
        triggered: list[str] = []
        final_urgency = UrgencyLevel.ROUTINE
        confidence = 0.65

        for rule in self.config.get("rules", []):
            if self._rule_matches(rule, session, text_blob):
                triggered.append(rule["id"])
                urgency = UrgencyLevel(rule["urgency"])
                if urgency == UrgencyLevel.EMERGENCY:
                    final_urgency = urgency
                    confidence = max(confidence, 0.98)
                    break
                if urgency == UrgencyLevel.URGENT and final_urgency != UrgencyLevel.EMERGENCY:
                    final_urgency = urgency
                    confidence = max(confidence, 0.86)

        return RuleEngineResult(urgency_level=final_urgency.value, triggered_rules=triggered, confidence=confidence)

    @staticmethod
    def _rule_matches(rule: dict, session: TriageSession, text_blob: str) -> bool:
        phrases = rule.get("phrases_any", [])
        if phrases and not any(p.lower() in text_blob for p in phrases):
            return False
        phrases_all = rule.get("phrases_all", [])
        if phrases_all and not all(p.lower() in text_blob for p in phrases_all):
            return False
        min_age = rule.get("min_age")
        if min_age is not None and (session.demographics.age is None or session.demographics.age < min_age):
            return False
        max_duration_days = rule.get("max_duration_days")
        if max_duration_days is not None and session.onset_time:
            if "day" not in session.onset_time.lower():
                return False
            digits = re.findall(r"\d+", session.onset_time)
            if digits and int(digits[0]) > max_duration_days:
                return False
        severity_min = rule.get("severity_min")
        if severity_min is not None and (session.severity is None or session.severity < severity_min):
            return False
        return True


class RiskScoringAgent:
    def score(self, session: TriageSession) -> tuple[float, float]:
        # Supplemental conservative deterministic estimate (placeholder for ML integration).
        base = 0.1
        if session.severity:
            base += session.severity / 15
        if session.demographics.age and session.demographics.age >= 65:
            base += 0.15
        if any("pregnan" in s.lower() for s in [session.chief_complaint, *session.associated_symptoms]):
            base += 0.1
        risk = min(0.99, round(base, 3))
        confidence = 0.72
        return risk, confidence


class EscalationAgent:
    EMERGENCY_TEXT = (
        "Seek immediate medical care. Call emergency services or go to nearest emergency department. "
        "I am stopping this triage now for safety."
    )

    def handoff_message(self, urgency: str) -> tuple[str, bool, str | None]:
        if urgency == UrgencyLevel.EMERGENCY.value:
            return self.EMERGENCY_TEXT, True, "Emergency red-flag/rules trigger"
        if urgency == UrgencyLevel.URGENT.value:
            return (
                "Your symptoms should be evaluated within 24 hours. Please arrange urgent clinical review now.",
                True,
                "Urgent triage recommendation",
            )
        return (
            "This appears routine from triage data. Please book a standard appointment. If symptoms worsen, seek urgent care.",
            False,
            None,
        )


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat()
