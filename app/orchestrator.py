import re
from dataclasses import dataclass
from typing import List
from uuid import uuid4

from .agents import DataAgent, DiagnosisAgent, LeadAgent, SafetyAgent, TriageAgent
from .memory import ConversationMemory
from .models import AgentResult, ChatResponse, HandoffTicket


@dataclass
class OrchestrationResult:
    response: ChatResponse
    handoff_ticket: HandoffTicket | None


class Coordinator:
    URGENT_PATTERNS = (
        re.compile(r"\bhigh[- ]risk\b", re.IGNORECASE),
        re.compile(r"\bemergency\b", re.IGNORECASE),
        re.compile(r"\burgent\b", re.IGNORECASE),
        re.compile(r"\bchest pain\b", re.IGNORECASE),
        re.compile(r"\bshortness of breath\b", re.IGNORECASE),
        re.compile(r"\bsuicidal\b", re.IGNORECASE),
    )

    def __init__(self, memory: ConversationMemory) -> None:
        self.memory = memory
        self.triage_agent = TriageAgent()
        self.diagnosis_agent = DiagnosisAgent()
        self.safety_agent = SafetyAgent()
        self.data_agent = DataAgent()
        self.lead_agent = LeadAgent(
            triage_agent=self.triage_agent,
            safety_agent=self.safety_agent,
            data_agent=self.data_agent,
            diagnosis_agent=self.diagnosis_agent,
        )

    def process(self, conversation_id: str, user_message: str) -> OrchestrationResult:
        self.memory.add_message(conversation_id, "user", user_message)
        history = self.memory.get_messages(conversation_id)

        trace: List[AgentResult] = []
        for agent in self.lead_agent.receive_user_message(user_message):
            trace.append(agent.run(user_message, history))

        requires_handoff, reason = self._handoff_decision(trace)
        final_response = self.lead_agent.format_for_user(trace, requires_handoff)
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

    def _handoff_decision(self, trace: List[AgentResult]) -> tuple[bool, str | None]:
        combined = "\n\n".join([f"{item.agent}: {item.summary}" for item in trace])
        requires_handoff = any(pattern.search(combined) for pattern in self.URGENT_PATTERNS)
        reason = "Potential high-acuity concern; escalation to human clinician recommended." if requires_handoff else None
        return requires_handoff, reason
