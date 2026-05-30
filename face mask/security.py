"""
Anti-Spoofing & Security Lock Module.
Tracks failed attempts and enters LOCK MODE after threshold.
Admin password stored as salted SHA-256 hash.
"""

import os
import json
import hashlib
import secrets
import time
import config


class SecurityManager:
    """Manages failed attempts, lock mode, and admin authentication."""

    def __init__(self):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        self.failed_attempts = 0
        self.locked = False
        self.lock_time = None
        self._load_lock_state()
        self._ensure_admin_password()

    # ─── Lock State Persistence ─────────────────────────────────────────────

    def _load_lock_state(self):
        """Load lock state from disk."""
        if os.path.exists(config.LOCK_STATE_FILE):
            try:
                with open(config.LOCK_STATE_FILE, "r") as f:
                    state = json.load(f)
                self.locked = state.get("locked", False)
                self.failed_attempts = state.get("failed_attempts", 0)
                self.lock_time = state.get("lock_time", None)
            except (json.JSONDecodeError, KeyError):
                self.locked = False
                self.failed_attempts = 0

    def _save_lock_state(self):
        """Persist lock state to disk."""
        state = {
            "locked": self.locked,
            "failed_attempts": self.failed_attempts,
            "lock_time": self.lock_time
        }
        with open(config.LOCK_STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    # ─── Admin Password ────────────────────────────────────────────────────

    def _ensure_admin_password(self):
        """Create default admin password if none exists."""
        if not os.path.exists(config.ADMIN_PASSWORD_FILE):
            self.set_admin_password(config.DEFAULT_ADMIN_PASSWORD)
            print(f"[SECURITY] Default admin password set. Change it with admin.py --set-password")

    def set_admin_password(self, password: str):
        """Set admin password (stored as salted SHA-256 hash)."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((salt + password).encode()).hexdigest()

        with open(config.ADMIN_PASSWORD_FILE, "w") as f:
            json.dump({"salt": salt, "hash": password_hash}, f)

        print("[SECURITY] Admin password updated.")

    def verify_admin_password(self, password: str) -> bool:
        """Verify admin password against stored hash."""
        if not os.path.exists(config.ADMIN_PASSWORD_FILE):
            return False

        try:
            with open(config.ADMIN_PASSWORD_FILE, "r") as f:
                data = json.load(f)

            salt = data["salt"]
            stored_hash = data["hash"]
            input_hash = hashlib.sha256((salt + password).encode()).hexdigest()

            return input_hash == stored_hash
        except (json.JSONDecodeError, KeyError):
            return False

    # ─── Attempt Tracking ──────────────────────────────────────────────────

    def record_attempt(self, success: bool):
        """
        Record an authentication attempt.
        Resets counter on success, increments on failure.
        Enters LOCK MODE when threshold exceeded.
        """
        if success:
            self.failed_attempts = 0
            self.locked = False
            self.lock_time = None
            self._save_lock_state()
            return

        self.failed_attempts += 1
        print(f"[SECURITY] Failed attempt {self.failed_attempts}/{config.MAX_FAILED_ATTEMPTS}")

        if self.failed_attempts >= config.MAX_FAILED_ATTEMPTS:
            self.locked = True
            self.lock_time = time.time()
            print(f"[SECURITY] ⚠ SYSTEM LOCKED — Too many failed attempts!")

        self._save_lock_state()

    def is_locked(self) -> bool:
        """Check if system is in LOCK MODE."""
        return self.locked

    def admin_unlock(self, password: str) -> bool:
        """
        Attempt to unlock system with admin password.
        Returns True if successfully unlocked.
        """
        if self.verify_admin_password(password):
            self.locked = False
            self.failed_attempts = 0
            self.lock_time = None
            self._save_lock_state()
            print("[SECURITY] ✓ System unlocked by admin.")
            return True
        else:
            print("[SECURITY] ✗ Invalid admin password.")
            return False

    def get_status(self) -> dict:
        """Get current security status."""
        return {
            "locked": self.locked,
            "failed_attempts": self.failed_attempts,
            "max_attempts": config.MAX_FAILED_ATTEMPTS,
            "lock_time": self.lock_time
        }

    def draw_lock_screen(self, frame):
        """Draw lock mode overlay on frame."""
        import cv2
        h, w = frame.shape[:2]

        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 50), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Lock icon (text-based)
        cv2.putText(frame, "SYSTEM LOCKED", (w // 2 - 180, h // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.putText(frame, f"Too many failed attempts ({self.failed_attempts})",
                    (w // 2 - 200, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, "Run: python admin.py --unlock", (w // 2 - 190, h // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.putText(frame, "Press 'q' to quit", (w // 2 - 100, h // 2 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        return frame
