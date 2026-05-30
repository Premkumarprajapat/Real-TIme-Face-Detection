"""
Tamper-Proof Logging System.
Append-only log with SHA-256 hash chaining.
Any modification to past entries breaks the chain → detectable.
"""

import os
import json
import hashlib
from datetime import datetime
import config


class TamperProofLog:
    """Hash-chained append-only log for access events."""

    def __init__(self):
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        self.log_file = config.ACCESS_LOG_FILE
        self._last_hash = self._get_last_hash()

    def _get_last_hash(self) -> str:
        """Get the hash of the last log entry, or genesis hash."""
        if not os.path.exists(self.log_file):
            return "0" * 64  # Genesis hash

        try:
            with open(self.log_file, "r") as f:
                lines = f.readlines()
                if not lines:
                    return "0" * 64

                last_line = lines[-1].strip()
                if last_line:
                    entry = json.loads(last_line)
                    return entry.get("hash", "0" * 64)
        except (json.JSONDecodeError, KeyError):
            return "0" * 64

        return "0" * 64

    def _compute_hash(self, content: str, prev_hash: str) -> str:
        """Compute SHA-256 hash of content + previous hash."""
        data = f"{prev_hash}{content}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def append_log(self, event: str, name: str = "", confidence: float = 0.0,
                   details: str = "") -> dict:
        """
        Append a new log entry with hash chaining.

        Args:
            event: Event type (e.g., "ACCESS_GRANTED", "ACCESS_DENIED", "SYSTEM_LOCKED")
            name: Person name (if applicable)
            confidence: Recognition confidence (if applicable)
            details: Additional details

        Returns:
            The log entry dict
        """
        now = datetime.now()
        timestamp = now.isoformat()

        # Build content string for hashing (excludes hash fields)
        content = f"{timestamp}|{event}|{name}|{confidence:.4f}|{details}"

        # Compute chained hash
        entry_hash = self._compute_hash(content, self._last_hash)

        entry = {
            "timestamp": timestamp,
            "event": event,
            "name": name,
            "confidence": round(confidence, 4),
            "details": details,
            "prev_hash": self._last_hash,
            "hash": entry_hash
        }

        # Append to log file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        self._last_hash = entry_hash
        print(f"[LOG] {event}: {name} ({confidence:.1%}) — {details}")

        return entry

    def verify_integrity(self) -> dict:
        """
        Verify the integrity of the entire log chain.

        Returns dict:
            {
                "valid": bool,
                "total_entries": int,
                "first_broken_at": int or None,
                "details": str
            }
        """
        if not os.path.exists(self.log_file):
            return {
                "valid": True,
                "total_entries": 0,
                "first_broken_at": None,
                "details": "No log file exists yet."
            }

        with open(self.log_file, "r") as f:
            lines = f.readlines()

        if not lines:
            return {
                "valid": True,
                "total_entries": 0,
                "first_broken_at": None,
                "details": "Log file is empty."
            }

        prev_hash = "0" * 64  # Genesis hash
        total = 0

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                return {
                    "valid": False,
                    "total_entries": total,
                    "first_broken_at": i + 1,
                    "details": f"Line {i + 1}: Invalid JSON"
                }

            # Check prev_hash chain
            if entry.get("prev_hash") != prev_hash:
                return {
                    "valid": False,
                    "total_entries": total,
                    "first_broken_at": i + 1,
                    "details": f"Line {i + 1}: Previous hash mismatch (chain broken)"
                }

            # Recompute hash and verify
            content = (
                f"{entry['timestamp']}|{entry['event']}|{entry['name']}"
                f"|{entry['confidence']:.4f}|{entry.get('details', '')}"
            )
            expected_hash = self._compute_hash(content, prev_hash)

            if entry.get("hash") != expected_hash:
                return {
                    "valid": False,
                    "total_entries": total,
                    "first_broken_at": i + 1,
                    "details": f"Line {i + 1}: Content hash mismatch (data tampered)"
                }

            prev_hash = entry["hash"]

        return {
            "valid": True,
            "total_entries": total,
            "first_broken_at": None,
            "details": f"All {total} entries verified. Log integrity intact."
        }

    def get_recent_entries(self, count: int = 10) -> list:
        """Get the most recent log entries."""
        if not os.path.exists(self.log_file):
            return []

        with open(self.log_file, "r") as f:
            lines = f.readlines()

        entries = []
        for line in lines[-count:]:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        return entries
