from dataclasses import dataclass
from typing import List
from uuid import uuid4

from .agents import DataAgent, DiagnosisAgent, SafetyAgent, TriageAgent
from .memory import ConversationMemory
from .models import AgentResult, ChatResponse, HandoffTicket


@dataclass
class OrchestrationResult:
    response: ChatResponse
    handoff_ticket: HandoffTicket | None


class Coordinator:
    def __init__(self, memory: ConversationMemory) -> None:
        self.memory = memory
        self.triage_agent = TriageAgent()
        self.diagnosis_agent = DiagnosisAgent()
        self.safety_agent = SafetyAgent()
        self.data_agent = DataAgent()

    def process(self, conversation_id: str, user_message: str) -> OrchestrationResult:
        self.memory.add_message(conversation_id, "user", user_message)
        history = self.memory.get_messages(conversation_id)

        trace: List[AgentResult] = [
            self.triage_agent.run(user_message, history),
            self.diagnosis_agent.run(user_message, history),
            self.safety_agent.run(user_message, history),
            self.data_agent.run(user_message, history),
        ]

        final_response, requires_handoff, reason = self._aggregate(trace)
        self.memory.add_message(conversation_id, "assistant", final_response)

        response = ChatResponse(
            conversation_id=conversation_id,
            response=final_response,
            agent_trace=trace,
            requires_handoff=requires_handoff,
            handoff_reason=reason,
        )

        ticket = None
        if requires_handoff:
            ticket = HandoffTicket(
                ticket_id=str(uuid4()),
                conversation_id=conversation_id,
                reason=reason,
                user_message=user_message,
            )

        return OrchestrationResult(response=response, handoff_ticket=ticket)

    def _aggregate(self, trace: List[AgentResult]) -> tuple[str, bool, str | None]:
        urgent_signals = [
            "high-risk",
            "emergency",
            "urgent",
            "chest pain",
            "shortness of breath",
            "suicidal",
        ]
        combined = "\n\n".join([f"{item.agent}: {item.summary}" for item in trace])
        lowered = combined.lower()
        requires_handoff = any(signal in lowered for signal in urgent_signals)
        reason = "Potential high-acuity concern; escalation to human clinician recommended." if requires_handoff else None

        final = (
            "Celine Multi-Agent Summary\n"
            "--------------------------------\n"
            f"{combined}\n\n"
            "Suggested next step: "
        )
        final += (
            "I am escalating this conversation to a human clinician now."
            if requires_handoff
            else "Continue with guided follow-up questions and routine care advice."
        )
        final += "\n\nNote: This is informational and not a definitive medical diagnosis."
        return final, requires_handoff, reason
