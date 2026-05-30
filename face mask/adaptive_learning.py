"""
Adaptive Learning Module.
Stores low-confidence matches for admin review.
Supports incremental embedding updates without full retraining.
"""

import os
import pickle
import time
from datetime import datetime
import cv2
import config
from face_encoder import add_encoding


class AdaptiveLearner:
    """Manages low-confidence match review queue and incremental learning."""

    def __init__(self):
        os.makedirs(config.DATA_DIR, exist_ok=True)
        self.review_queue = self._load_queue()

    def _load_queue(self) -> list:
        """Load review queue from disk."""
        if os.path.exists(config.REVIEW_QUEUE_FILE):
            try:
                with open(config.REVIEW_QUEUE_FILE, "rb") as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                return []
        return []

    def _save_queue(self):
        """Save review queue to disk."""
        with open(config.REVIEW_QUEUE_FILE, "wb") as f:
            pickle.dump(self.review_queue, f)

    def should_queue(self, confidence: float) -> bool:
        """Check if a match should be queued for review (between low and high thresholds)."""
        return config.ADAPTIVE_LOW_THRESHOLD <= confidence < config.ADAPTIVE_HIGH_THRESHOLD

    def add_to_queue(self, name: str, confidence: float, encoding, frame=None):
        """
        Add a low-confidence match to the review queue.

        Args:
            name: Best-match name
            confidence: Confidence score
            encoding: 128-d face encoding
            frame: Optional frame for reference (saved as image)
        """
        timestamp = datetime.now().isoformat()
        entry = {
            "name": name,
            "confidence": round(confidence, 4),
            "encoding": encoding,
            "timestamp": timestamp,
            "reviewed": False,
            "image_path": None
        }

        # Save reference image if provided
        if frame is not None:
            img_dir = os.path.join(config.DATA_DIR, "review_images")
            os.makedirs(img_dir, exist_ok=True)
            img_name = f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            img_path = os.path.join(img_dir, img_name)
            cv2.imwrite(img_path, frame)
            entry["image_path"] = img_path

        self.review_queue.append(entry)
        self._save_queue()

        print(f"[ADAPTIVE] Queued for review: {name} ({confidence:.1%})")

    def get_pending_reviews(self) -> list:
        """Get all pending (unreviewed) entries."""
        return [e for e in self.review_queue if not e["reviewed"]]

    def approve_entry(self, index: int) -> bool:
        """
        Approve a review entry — adds its encoding to the known set.

        Args:
            index: Index in the pending review list

        Returns:
            True if approved successfully
        """
        pending = self.get_pending_reviews()
        if 0 <= index < len(pending):
            entry = pending[index]
            name = entry["name"]
            encoding = entry["encoding"]

            # Add encoding to known set (incremental, no full retrain)
            add_encoding(name, encoding)

            # Mark as reviewed
            entry["reviewed"] = True
            entry["approved"] = True
            self._save_queue()

            print(f"[ADAPTIVE] ✓ Approved: {name} — encoding added")
            return True

        print(f"[ADAPTIVE] Invalid index: {index}")
        return False

    def reject_entry(self, index: int) -> bool:
        """
        Reject a review entry — marks as reviewed but does not add encoding.

        Args:
            index: Index in the pending review list

        Returns:
            True if rejected successfully
        """
        pending = self.get_pending_reviews()
        if 0 <= index < len(pending):
            entry = pending[index]
            entry["reviewed"] = True
            entry["approved"] = False
            self._save_queue()

            print(f"[ADAPTIVE] ✗ Rejected: {entry['name']}")
            return True

        print(f"[ADAPTIVE] Invalid index: {index}")
        return False

    def clear_reviewed(self):
        """Remove all reviewed entries from the queue."""
        self.review_queue = [e for e in self.review_queue if not e["reviewed"]]
        self._save_queue()
        print("[ADAPTIVE] Cleared reviewed entries.")

    def get_stats(self) -> dict:
        """Get adaptive learning statistics."""
        pending = len(self.get_pending_reviews())
        total = len(self.review_queue)
        reviewed = total - pending
        return {
            "total": total,
            "pending": pending,
            "reviewed": reviewed
        }
