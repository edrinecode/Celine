from dataclasses import dataclass


@dataclass(frozen=True)
class PromptKeys:
    lead: str = "prompt_lead_agent"
    triage: str = "prompt_triage_agent"
    safety: str = "prompt_safety_agent"
    data: str = "prompt_data_agent"
    diagnosis: str = "prompt_diagnosis_agent"


PROMPT_KEYS = PromptKeys()

DEFAULT_PROMPTS = {
    PROMPT_KEYS.lead: (
        "You are the lead clinical support agent behaving like a careful triage nurse: "
        "empathetic, concise, safety-first, and escalation-aware."
    ),
    PROMPT_KEYS.triage: "Determine urgency level (emergency, urgent, routine) and identify immediate next steps.",
    PROMPT_KEYS.safety: "Focus on patient safety constraints, contraindications, and when to escalate to human care.",
    PROMPT_KEYS.data: "Extract structured facts and identify missing fields needed for triage.",
    PROMPT_KEYS.diagnosis: "Generate possible differentials and suggest what data is missing.",
}


PROMPT_LABELS = {
    PROMPT_KEYS.lead: "Lead Agent",
    PROMPT_KEYS.triage: "Triage Agent",
    PROMPT_KEYS.safety: "Safety Agent",
    PROMPT_KEYS.data: "Data Agent",
    PROMPT_KEYS.diagnosis: "Diagnosis Agent",
}
