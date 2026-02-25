import logging
import os
import re
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from .models import AgentResult, ChatMessage
from .tools import ClinicalTools

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover
    ChatGoogleGenerativeAI = None


class BaseAgent:
    def __init__(self, name: str, system_prompt: str) -> None:
        self.name = name
        self.system_prompt = system_prompt

    def run(self, user_message: str, history: List[ChatMessage]) -> AgentResult:
        recent_history = history[-12:]
        context = "\n".join([f"{m.timestamp.isoformat()} | {m.role}: {m.content}" for m in recent_history])
        prompt = (
            f"{self.system_prompt}\n"
            f"Conversation context:\n{context}\n"
            f"User message: {user_message}\n"
            f"Current UTC time: {ClinicalTools.timestamp_tool()}\n"
            "Respond with concise clinical support summary, not final diagnosis."
        )
        summary = self._invoke_llm(
            prompt,
            user_message=user_message,
            system_prompt=self.system_prompt,
        )
        return AgentResult(agent=self.name, summary=summary)

    def _invoke_llm(self, prompt: str, user_message: str = "", system_prompt: str = "") -> str:
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key and ChatGoogleGenerativeAI:
            model = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
            llm = ChatGoogleGenerativeAI(model=model, temperature=0.2)
            try:
                result = llm.invoke(
                    [
                        SystemMessage(
                            content=(
                                f"You are {self.name}, a healthcare support AI agent. "
                                f"System directive: {system_prompt}"
                            )
                        ),
                        HumanMessage(content=prompt),
                    ]
                )
                return result.content
            except Exception:  # pragma: no cover - exercised through mocking in unit tests
                logging.exception(
                    "LLM invocation failed for %s with model %s. Using fallback response.",
                    self.name,
                    model,
                )
        excerpt = " ".join(user_message.split())[:160]
        symptom_note = excerpt or 'No symptoms provided.'
        return f"[{self.name} fallback] Unable to access model. Symptom note: {symptom_note}"


class TriageAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Triage Agent",
            system_prompt=(
                "Determine urgency level (emergency, urgent, routine) and identify immediate next steps."
            ),
        )


class DiagnosisAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Diagnosis Agent",
            system_prompt="Generate possible differentials and suggest what data is missing.",
        )


class SafetyAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Safety Agent",
            system_prompt=(
                "Focus on patient safety constraints, contraindications, and when to escalate to human care."
            ),
        )

    def run(self, user_message: str, history: List[ChatMessage]) -> AgentResult:
        heuristic = ClinicalTools.vitals_risk_heuristic(user_message)
        base = super().run(user_message, history)
        base.summary = f"{heuristic}\n{base.summary}"
        return base


class DataAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Data Agent",
            system_prompt="Extract structured facts and identify missing fields needed for triage.",
        )


class LeadAgent:
    SYSTEM_PROMPT = (
        "You are the lead clinical support agent behaving like a careful triage nurse: "
        "empathetic, concise, safety-first, and escalation-aware."
    )

    GREETING_PATTERNS = (
        re.compile(r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening))\b", re.IGNORECASE),
        re.compile(r"^\s*(thanks|thank you)\b", re.IGNORECASE),
    )
    CLINICAL_PATTERNS = (
        re.compile(r"\bpain\b", re.IGNORECASE),
        re.compile(r"\bfever\b", re.IGNORECASE),
        re.compile(r"\bcough\b", re.IGNORECASE),
        re.compile(r"\bnausea\b", re.IGNORECASE),
        re.compile(r"\bvomit\w*\b", re.IGNORECASE),
        re.compile(r"\bdizziness?\b", re.IGNORECASE),
        re.compile(r"\bdizzy\b", re.IGNORECASE),
        re.compile(r"\bheadache\b", re.IGNORECASE),
        re.compile(r"\bbreath\w*\b", re.IGNORECASE),
        re.compile(r"\ballerg\w*\b", re.IGNORECASE),
    )

    def __init__(
        self,
        triage_agent: TriageAgent,
        safety_agent: SafetyAgent,
        data_agent: DataAgent,
        diagnosis_agent: DiagnosisAgent,
    ) -> None:
        self.name = "Lead Agent"
        self.system_prompt = self.SYSTEM_PROMPT
        self.triage_agent = triage_agent
        self.safety_agent = safety_agent
        self.data_agent = data_agent
        self.diagnosis_agent = diagnosis_agent

    def receive_user_message(self, user_message: str):
        message = user_message.strip()
        is_greeting = any(pattern.search(message) for pattern in self.GREETING_PATTERNS)
        has_clinical_signal = any(pattern.search(message) for pattern in self.CLINICAL_PATTERNS)

        selected = [self.triage_agent]
        if is_greeting and not has_clinical_signal:
            return selected

        selected.append(self.safety_agent)
        if has_clinical_signal or len(message.split()) >= 6:
            selected.append(self.data_agent)
        if has_clinical_signal and len(message.split()) >= 8:
            selected.append(self.diagnosis_agent)
        return selected

    def format_for_user(self, agent_results: List[AgentResult], requires_handoff: bool, history: List[ChatMessage]) -> str:
        if self._is_social_only(agent_results):
            return (
                "Hi — I’m doing well, and I’m here for you. 😊 "
                "If you want, you can share what symptoms or health concern you’re dealing with, "
                "and I’ll guide you step by step."
            )

        if requires_handoff:
            return (
                "Thanks for sharing this — your symptoms may need urgent medical attention. "
                "I’m escalating this to a human clinician now. "
                "If you have severe chest pain, trouble breathing, fainting, or worsening symptoms, "
                "please call emergency services right away."
            )

        triage_summary = self._summary_for(agent_results, "Triage Agent")
        opening = "Thanks for sharing that."
        continuity = self._build_continuity_note(history)
        if "urgent" in triage_summary.lower():
            opening = "Thanks for sharing that — this sounds important to assess promptly."

        return (
            f"{opening} {continuity}"
            "I can help you narrow this down with a few quick questions: "
            "when did this start, how severe is it (0–10), and do you have any red-flag symptoms "
            "like trouble breathing, chest pain, confusion, or fainting?\n\n"
            "I can provide informational guidance, but this is not a definitive diagnosis."
        )


    @staticmethod
    def _build_continuity_note(history: List[ChatMessage]) -> str:
        previous_user_messages = [item.content for item in history if item.role == "user"]
        if len(previous_user_messages) < 2:
            return ""
        prior_context = LeadAgent._compact(previous_user_messages[-2], limit=120)
        return f"From earlier, I noted: '{prior_context}'. "

    @staticmethod
    def _is_social_only(agent_results: List[AgentResult]) -> bool:
        return len(agent_results) == 1 and agent_results[0].agent == "Triage Agent"

    @staticmethod
    def _summary_for(agent_results: List[AgentResult], agent_name: str) -> str:
        for result in agent_results:
            if result.agent == agent_name:
                return result.summary
        return ""

    @staticmethod
    def _compact(text: str, limit: int = 220) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."
