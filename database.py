from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "documind.db"


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE IF NOT EXISTS documents (id INTEGER PRIMARY KEY, name TEXT NOT NULL, file_type TEXT, pages INTEGER, characters INTEGER, created_at TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS questions (id INTEGER PRIMARY KEY, document_id INTEGER, question TEXT NOT NULL, answer TEXT NOT NULL, created_at TEXT NOT NULL, FOREIGN KEY(document_id) REFERENCES documents(id))")
    conn.commit()
    return conn


def add_document(name: str, file_type: str, pages: int | None, characters: int) -> int:
    with connect() as conn:
        cur = conn.execute("INSERT INTO documents(name,file_type,pages,characters,created_at) VALUES(?,?,?,?,?)", (name, file_type, pages, characters, datetime.now().isoformat(timespec="seconds")))
        return int(cur.lastrowid)


def add_question(document_id: int, question: str, answer: str) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO questions(document_id,question,answer,created_at) VALUES(?,?,?,?)", (document_id, question, answer, datetime.now().isoformat(timespec="seconds")))


def recent_documents(limit: int = 20):
    with connect() as conn:
        return conn.execute("SELECT * FROM documents ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def recent_questions(limit: int = 20):
    with connect() as conn:
        return conn.execute("SELECT q.*, d.name FROM questions q LEFT JOIN documents d ON d.id=q.document_id ORDER BY q.id DESC LIMIT ?", (limit,)).fetchall()
