import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List

from .models import ChatMessage, HandoffTicket


class SQLiteStore:
    def __init__(self, db_path: str = "celine.db") -> None:
        self.db_path = db_path
        self._lock = Lock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS handoff_tickets (
                    ticket_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

    def add_message(self, conversation_id: str, role: str, content: str, timestamp: datetime) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                    (conversation_id, role, content, timestamp.isoformat()),
                )

    def get_messages(self, conversation_id: str, limit: int = 50) -> List[ChatMessage]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT role, content, timestamp
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (conversation_id, limit),
                ).fetchall()
        return [
            ChatMessage(
                role=row["role"],
                content=row["content"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
            for row in reversed(rows)
        ]

    def add_handoff_ticket(self, ticket: HandoffTicket) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO handoff_tickets
                    (ticket_id, conversation_id, reason, user_message, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        ticket.ticket_id,
                        ticket.conversation_id,
                        ticket.reason,
                        ticket.user_message,
                        ticket.created_at.isoformat(),
                    ),
                )

    def list_handoff_tickets(self, limit: int = 200) -> List[HandoffTicket]:
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT ticket_id, conversation_id, reason, user_message, created_at
                    FROM handoff_tickets
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()

        return [
            HandoffTicket(
                ticket_id=row["ticket_id"],
                conversation_id=row["conversation_id"],
                reason=row["reason"],
                user_message=row["user_message"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def resolve_handoff_ticket(self, ticket_id: str) -> int:
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM handoff_tickets WHERE ticket_id = ?", (ticket_id,))
                count_row = conn.execute("SELECT COUNT(*) as total FROM handoff_tickets").fetchone()
                return int(count_row["total"])

    def get_setting(self, key: str) -> str | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        return str(row["value"])

    def set_setting(self, key: str, value: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO settings (key, value)
                    VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value
                    """,
                    (key, value),
                )
