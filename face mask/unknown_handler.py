"""
Unknown Face Handler.
Tracks unknown faces, captures images periodically, and times out.
"""

import os
import time
import cv2
from datetime import datetime
import config


class UnknownFaceHandler:
    """Manages unknown face tracking and image capture."""

    def __init__(self):
        os.makedirs(config.UNKNOWN_FACES_DIR, exist_ok=True)
        # Track unknown faces by approximate location hash
        self._trackers = {}  # {face_key: {"first_seen": time, "last_capture": time}}

    def _face_key(self, face_location) -> str:
        """Generate a rough key for a face based on its location."""
        if face_location is None:
            return "none"
        top, right, bottom, left = face_location
        # Quantize to reduce sensitivity to small movements
        return f"{top // 50}_{left // 50}"

    def process_unknown(self, frame, face_location) -> dict:
        """
        Process an unknown face detection.

        Returns dict:
            {
                "status": "tracking" | "captured" | "timed_out",
                "time_remaining": float,
                "captures": int
            }
        """
        key = self._face_key(face_location)
        now = time.time()

        if key not in self._trackers:
            self._trackers[key] = {
                "first_seen": now,
                "last_capture": 0,
                "captures": 0
            }

        tracker = self._trackers[key]
        elapsed = now - tracker["first_seen"]
        time_remaining = max(0, config.UNKNOWN_TIMEOUT - elapsed)

        # Timeout — stop tracking
        if elapsed >= config.UNKNOWN_TIMEOUT:
            self._trackers.pop(key, None)
            return {
                "status": "timed_out",
                "time_remaining": 0,
                "captures": tracker["captures"]
            }

        # Capture every UNKNOWN_CAPTURE_INTERVAL seconds
        since_last_capture = now - tracker["last_capture"]
        if since_last_capture >= config.UNKNOWN_CAPTURE_INTERVAL:
            self._capture_image(frame, face_location)
            tracker["last_capture"] = now
            tracker["captures"] += 1
            return {
                "status": "captured",
                "time_remaining": time_remaining,
                "captures": tracker["captures"]
            }

        return {
            "status": "tracking",
            "time_remaining": time_remaining,
            "captures": tracker["captures"]
        }

    def _capture_image(self, frame, face_location):
        """Save an unknown face image to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unknown_{timestamp}.jpg"
        filepath = os.path.join(config.UNKNOWN_FACES_DIR, filename)

        # Crop face with some padding
        if face_location is not None:
            top, right, bottom, left = face_location
            h, w = frame.shape[:2]
            pad = 30
            top = max(0, top - pad)
            left = max(0, left - pad)
            bottom = min(h, bottom + pad)
            right = min(w, right + pad)
            face_img = frame[top:bottom, left:right]
        else:
            face_img = frame

        cv2.imwrite(filepath, face_img)
        print(f"[UNKNOWN] Captured unknown face: {filename}")

    def reset(self):
        """Clear all trackers."""
        self._trackers.clear()

    def get_active_count(self) -> int:
        """Get number of currently tracked unknown faces."""
        # Clean up expired trackers
        now = time.time()
        expired = [k for k, v in self._trackers.items()
                   if now - v["first_seen"] >= config.UNKNOWN_TIMEOUT]
        for k in expired:
            del self._trackers[k]
        return len(self._trackers)
