import logging
import os
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
