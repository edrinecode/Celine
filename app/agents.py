import logging
import os
import re
from dataclasses import dataclass
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from .models import AgentResult, ChatMessage
from .prompts import DEFAULT_PROMPTS, PROMPT_KEYS
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
        excerpt = " ".join(user_message.split())[:180]
        symptom_note = excerpt or "No symptoms provided."
        return f"[{self.name} fallback] Unable to access model. Symptom note: {symptom_note}"


class TriageAgent(BaseAgent):
    def __init__(self, system_prompt: str | None = None) -> None:
        super().__init__(
            name="Triage Agent",
            system_prompt=system_prompt or DEFAULT_PROMPTS[PROMPT_KEYS.triage],
        )


class DiagnosisAgent(BaseAgent):
    def __init__(self, system_prompt: str | None = None) -> None:
        super().__init__(
            name="Diagnosis Agent",
            system_prompt=system_prompt or DEFAULT_PROMPTS[PROMPT_KEYS.diagnosis],
        )


class SafetyAgent(BaseAgent):
    def __init__(self, system_prompt: str | None = None) -> None:
        super().__init__(
            name="Safety Agent",
            system_prompt=system_prompt or DEFAULT_PROMPTS[PROMPT_KEYS.safety],
        )

    def run(self, user_message: str, history: List[ChatMessage]) -> AgentResult:
        heuristic = ClinicalTools.vitals_risk_heuristic(user_message)
        base = super().run(user_message, history)
        base.summary = f"{heuristic}\n{base.summary}"
        return base


class DataAgent(BaseAgent):
    def __init__(self, system_prompt: str | None = None) -> None:
        super().__init__(
            name="Data Agent",
            system_prompt=system_prompt or DEFAULT_PROMPTS[PROMPT_KEYS.data],
        )


class ConversationAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            name="Conversation Agent",
            system_prompt=(
                "Handle social/user-experience requests naturally. "
                "Introduce yourself as Celine when asked for your name."
            ),
        )

    def run(self, user_message: str, history: List[ChatMessage]) -> AgentResult:
        message = user_message.strip().lower()
        if re.search(r"\b(what('?s|\s+is)\s+(your|ur)\s+name|who\s+are\s+you|your\s+name)\b", message):
            summary = (
                "I'm Celine. I can help with initial clinical triage and care guidance. "
                "If you'd like, tell me what symptom is bothering you most right now."
            )
        elif re.search(r"^\s*(hi|hello|hey|yo|good\s+(morning|afternoon|evening))\b", message):
            summary = (
                "Hi — I’m Celine. I’m here with you. 😊 "
                "Whenever you’re ready, share any symptom or health concern and I’ll guide you step by step."
            )
        else:
            summary = (
                "I can absolutely help with that. I’m Celine, and I can offer initial clinical triage guidance. "
                "Share your symptoms and I’ll walk through them with you carefully."
            )
        return AgentResult(agent=self.name, summary=summary)


@dataclass
class RoutingDecision:
    mode: str
    agents: List[BaseAgent]


class LeadAgent:
    ACKNOWLEDGMENT_PATTERNS = (
        re.compile(r"^\s*(ok|okay|k|kk|got it|understood|sure|yep|yes|no|nah|nope)\s*[.!?]*\s*$", re.IGNORECASE),
        re.compile(r"^\s*(thanks|thank\s+you|thx)\b", re.IGNORECASE),
        re.compile(r"^\s*(i am just saying hi|not a symptom)\b", re.IGNORECASE),
    )

    PROFILE_PATTERNS = (
        re.compile(r"\b(what('?s|\s+is)\s+(your|ur)\s+name|who\s+are\s+you|your\s+name)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+can\s+you\s+do\b", re.IGNORECASE),
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
        re.compile(r"\bsymptom\w*\b", re.IGNORECASE),
        re.compile(r"\bsick\b", re.IGNORECASE),
    )

    GREETING_PATTERNS = (
        re.compile(r"^\s*(hi|hello|hey|yo|good\s+(morning|afternoon|evening))\b", re.IGNORECASE),
    )

    def __init__(
        self,
        triage_agent: TriageAgent,
        safety_agent: SafetyAgent,
        data_agent: DataAgent,
        diagnosis_agent: DiagnosisAgent,
        conversation_agent: ConversationAgent,
        system_prompt: str | None = None,
    ) -> None:
        self.name = "Lead Agent"
        self.system_prompt = system_prompt or DEFAULT_PROMPTS[PROMPT_KEYS.lead]
        self.triage_agent = triage_agent
        self.safety_agent = safety_agent
        self.data_agent = data_agent
        self.diagnosis_agent = diagnosis_agent
        self.conversation_agent = conversation_agent

    def receive_user_message(self, user_message: str) -> RoutingDecision:
        message = user_message.strip()
        token_count = len(message.split())

        has_clinical_signal = any(pattern.search(message) for pattern in self.CLINICAL_PATTERNS)
        is_profile_query = any(pattern.search(message) for pattern in self.PROFILE_PATTERNS)
        is_ack = any(pattern.search(message) for pattern in self.ACKNOWLEDGMENT_PATTERNS)
        is_greeting = any(pattern.search(message) for pattern in self.GREETING_PATTERNS)

        if is_profile_query or (is_greeting and not has_clinical_signal) or (is_ack and not has_clinical_signal):
            return RoutingDecision(mode="social", agents=[self.conversation_agent])

        selected: List[BaseAgent] = [self.triage_agent]
        if has_clinical_signal or token_count >= 5:
            selected.append(self.safety_agent)
        if has_clinical_signal or token_count >= 8:
            selected.append(self.data_agent)
        if has_clinical_signal and token_count >= 10:
            selected.append(self.diagnosis_agent)

        return RoutingDecision(mode="clinical", agents=selected)

    def format_for_user(
        self,
        routing_decision: RoutingDecision,
        agent_results: List[AgentResult],
        requires_handoff: bool,
        history: List[ChatMessage],
    ) -> str:
        if routing_decision.mode == "social":
            return agent_results[0].summary if agent_results else "Hi — I’m Celine. How can I help today?"

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
            "Here’s what I suggest next: tell me when this started, your severity (0–10), and whether you have "
            "red-flag symptoms like trouble breathing, chest pain, confusion, fainting, severe dehydration, or very high fever.\n\n"
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
