from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    conversation_id: str
    message: str


class AgentResult(BaseModel):
    agent: str
    summary: str


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    agent_trace: List[AgentResult]
    requires_handoff: bool = False
    handoff_reason: Optional[str] = None


class HandoffTicket(BaseModel):
    ticket_id: str
    conversation_id: str
    reason: str
    user_message: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
