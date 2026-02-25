from datetime import datetime
from typing import List

from .models import ChatMessage
from .storage import SQLiteStore


class ConversationMemory:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        self.store.add_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
        )

    def get_messages(self, conversation_id: str) -> List[ChatMessage]:
        return self.store.get_messages(conversation_id=conversation_id)
