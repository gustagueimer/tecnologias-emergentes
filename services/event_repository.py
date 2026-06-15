import os
import sqlite3
from dotenv import load_dotenv
from datetime import datetime

# =========================
# BANCO DE DADOS
# =========================
load_dotenv()

def init_db():
    conn = sqlite3.connect(os.getenv("DB_PATH"))
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            event_time TEXT,
            label TEXT,
            confidence REAL,
            image_path TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_event(event_id: str, label: str, confidence: float, image_path: str):
    conn = sqlite3.connect(os.getenv("DB_PATH"))
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO events (id, event_time, label, confidence, image_path)
        VALUES (?, ?, ?, ?, ?)
    """, (
        event_id,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        label,
        confidence,
        image_path
    ))
    conn.commit()
    conn.close()


def list_events(limit: int = 50):
    conn = sqlite3.connect(os.getenv("DB_PATH"))
    cur = conn.cursor()
    cur.execute("""
        SELECT id, event_time, label, confidence, image_path
        FROM events
        ORDER BY event_time DESC
        LIMIT ?
    """, (limit,))
    rows = cur.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "event_time": r[1],
            "label": r[2],
            "confidence": r[3],
            "image_path": r[4]
        }
        for r in rows
    ]
