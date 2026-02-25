from datetime import datetime

from .storage import SQLiteStore


class ConversationMemory:
    def __init__(self, store: SQLiteStore) -> None:
        self.store = store

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        # Message writes are managed in orchestrator for strict ordering.
        self.store.add_message(conversation_id, role, content, datetime.utcnow())

    def get_messages(self, conversation_id: str):
        return self.store.get_messages(conversation_id)
