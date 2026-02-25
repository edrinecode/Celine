from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from cryptography.fernet import Fernet

from .models import ChatMessage, HandoffTicket, TriageSession


class SQLiteStore:
    def __init__(self, db_path: str = "celine.db", encryption_key: str | None = None) -> None:
        self.db_path = db_path
        self._lock = Lock()
        self._fernet = Fernet(self._derive_key(encryption_key or os.getenv("CELINE_ENCRYPTION_KEY", "dev-key")))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @staticmethod
    def _derive_key(seed: str) -> bytes:
        digest = hashlib.sha256(seed.encode()).digest()
        return base64.urlsafe_b64encode(digest)

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
                CREATE TABLE IF NOT EXISTS sessions (
                    conversation_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    event_time TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )

    def encrypt(self, payload: dict[str, Any]) -> str:
        return self._fernet.encrypt(json.dumps(payload, default=str).encode()).decode()

    def decrypt(self, token: str) -> dict[str, Any]:
        return json.loads(self._fernet.decrypt(token.encode()).decode())

    def add_message(self, conversation_id: str, role: str, content: str, timestamp: datetime) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, timestamp.isoformat()),
            )

    def get_messages(self, conversation_id: str, limit: int = 50) -> list[ChatMessage]:
        with self._lock, self._connect() as conn:
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
            ChatMessage(role=row["role"], content=row["content"], timestamp=datetime.fromisoformat(row["timestamp"]))
            for row in reversed(rows)
        ]

    def get_or_create_session(self, conversation_id: str, patient_id: str) -> TriageSession:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT payload FROM sessions WHERE conversation_id = ?", (conversation_id,)).fetchone()
            if row:
                payload = self.decrypt(row["payload"])
                return TriageSession.model_validate(payload)

        session = TriageSession(session_id=conversation_id, patient_id=patient_id)
        self.save_session(session)
        return session

    def save_session(self, session: TriageSession) -> None:
        payload = self.encrypt(session.model_dump(mode="json"))
        now = datetime.utcnow().isoformat()

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (conversation_id, payload, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at
                """,
                (session.session_id, payload, now, now),
            )
            for event in session.audit_log:
                conn.execute(
                    "INSERT INTO audit_events (conversation_id, event_time, payload) VALUES (?, ?, ?)",
                    (
                        session.session_id,
                        datetime.utcnow().isoformat(),
                        self.encrypt(event.model_dump(mode="json")),
                    ),
                )

    def get_session_snapshot(self, conversation_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT payload FROM sessions WHERE conversation_id = ?", (conversation_id,)).fetchone()
            if not row:
                return None
            return self.decrypt(row["payload"])

    def get_audit_events(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM audit_events WHERE conversation_id = ? ORDER BY id ASC", (conversation_id,)
            ).fetchall()
        return [self.decrypt(row["payload"]) for row in rows]

    def add_handoff_ticket(self, ticket: HandoffTicket) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO handoff_tickets
                (ticket_id, conversation_id, reason, user_message, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ticket.ticket_id, ticket.conversation_id, ticket.reason, ticket.user_message, ticket.created_at.isoformat()),
            )

    def list_handoff_tickets(self, limit: int = 200) -> list[HandoffTicket]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT ticket_id, conversation_id, reason, user_message, created_at FROM handoff_tickets ORDER BY created_at DESC LIMIT ?",
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
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM handoff_tickets WHERE ticket_id = ?", (ticket_id,))
            row = conn.execute("SELECT COUNT(*) AS total FROM handoff_tickets").fetchone()
            return int(row["total"])
