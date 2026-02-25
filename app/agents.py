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
        context = "\n".join([f"{m.role}: {m.content}" for m in history[-6:]])
        prompt = (
            f"{self.system_prompt}\n"
            f"Conversation context:\n{context}\n"
            f"User message: {user_message}\n"
            f"Current UTC time: {ClinicalTools.timestamp_tool()}\n"
            "Respond with concise clinical support summary, not final diagnosis."
        )
        summary = self._invoke_llm(prompt)
        return AgentResult(agent=self.name, summary=summary)

    def _invoke_llm(self, prompt: str) -> str:
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key and ChatGoogleGenerativeAI:
            model = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
            llm = ChatGoogleGenerativeAI(model=model, temperature=0.2)
            try:
                result = llm.invoke(
                    [
                        SystemMessage(content="You are a healthcare support AI agent."),
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
        return f"[{self.name} fallback] {prompt[:260]}..."


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

    def format_for_user(self, agent_results: List[AgentResult], requires_handoff: bool) -> str:
        consulted_agents = ", ".join(item.agent for item in agent_results)
        next_step = (
            "Escalating to a human clinician now."
            if requires_handoff
            else "Continuing with focused follow-up questions and routine guidance."
        )

        detail_lines = "\n".join(
            f"- **{item.agent}:** {self._compact(item.summary)}"
            for item in agent_results
        )

        return (
            "## Celine Lead Agent Summary\n"
            f"- **Consulted Agents:** {consulted_agents}\n"
            f"- **Urgency:** {'Urgent review needed' if requires_handoff else 'Routine at this stage'}\n"
            f"- **Next Step:** {next_step}\n\n"
            "### Agent Notes\n"
            f"{detail_lines}\n\n"
            "_Informational support only; not a definitive medical diagnosis._"
        )

    @staticmethod
    def _compact(text: str, limit: int = 220) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."
