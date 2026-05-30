"""
SQLite Attendance Database.
Stores only valid (access granted) entries.
Enforces unique constraint: one entry per person per day.
"""

import sqlite3
import os
from datetime import datetime
import config


class AttendanceDB:
    """Thread-safe SQLite attendance logger."""

    def __init__(self):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        self.db_path = config.DATABASE_FILE
        self._init_db()

    def _init_db(self):
        """Create the attendance table if it doesn't exist."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                confidence REAL,
                UNIQUE(name, date)
            )
        """)
        conn.commit()
        conn.close()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def log_attendance(self, name: str, confidence: float = 0.0) -> bool:
        """
        Log attendance for a user.
        Returns True if entry was inserted, False if duplicate (already logged today).
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        conn = self._connect()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "INSERT OR IGNORE INTO attendance (name, date, time, confidence) VALUES (?, ?, ?, ?)",
                (name, date_str, time_str, round(confidence, 4))
            )
            conn.commit()
            inserted = cursor.rowcount > 0

            if inserted:
                print(f"[DB] Attendance logged: {name} at {time_str} on {date_str}")
            else:
                print(f"[DB] Duplicate: {name} already logged for {date_str}")

            return inserted
        except Exception as e:
            print(f"[DB] Error logging attendance: {e}")
            return False
        finally:
            conn.close()

    def get_attendance(self, date: str = None) -> list:
        """
        Get attendance records.
        If date is None, returns today's records.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, date, time, confidence FROM attendance WHERE date = ? ORDER BY time",
            (date,)
        )
        records = cursor.fetchall()
        conn.close()

        return [
            {"name": r[0], "date": r[1], "time": r[2], "confidence": r[3]}
            for r in records
        ]

    def get_all_attendance(self) -> list:
        """Get all attendance records."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT name, date, time, confidence FROM attendance ORDER BY date DESC, time DESC")
        records = cursor.fetchall()
        conn.close()

        return [
            {"name": r[0], "date": r[1], "time": r[2], "confidence": r[3]}
            for r in records
        ]

    def get_count(self, date: str = None) -> int:
        """Get attendance count for a date."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM attendance WHERE date = ?", (date,))
        count = cursor.fetchone()[0]
        conn.close()
        return count
