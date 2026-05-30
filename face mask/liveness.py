"""
Liveness Detection — Blink-Based Anti-Spoofing.
Simple blink counter using MediaPipe Face Mesh (fully offline).
No internal timer — the caller controls timing.
"""

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import config


# MediaPipe Face Mesh landmark indices for eyes
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]


class LivenessDetector:
    """Simple blink counter using MediaPipe Face Mesh."""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.blink_count = 0
        self.eye_was_closed = False

    def reset(self):
        """Reset blink count."""
        self.blink_count = 0
        self.eye_was_closed = False

    def _calculate_ear(self, eye_landmarks) -> float:
        """
        Calculate Eye Aspect Ratio (EAR).
        EAR ≈ 0.25-0.30 when open, drops to ~0.05 when closed.
        """
        v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        h = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        if h == 0:
            return 0.3
        return (v1 + v2) / (2.0 * h)

    def _get_eye_landmarks(self, landmarks, indices, frame_shape):
        """Extract eye landmark coordinates."""
        h, w = frame_shape[:2]
        return np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices],
                        dtype=np.float64)

    def detect_blink(self, frame) -> dict:
        """
        Process a frame and detect blinks.
        Returns dict with blink_count and current EAR.
        No internal timer — caller handles timing.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        ear = 0.0
        blinked_now = False

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0].landmark

            left_eye = self._get_eye_landmarks(face, LEFT_EYE_IDX, frame.shape)
            right_eye = self._get_eye_landmarks(face, RIGHT_EYE_IDX, frame.shape)

            left_ear = self._calculate_ear(left_eye)
            right_ear = self._calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Detect blink: eyes closed then opened
            if ear < config.EAR_THRESHOLD:
                self.eye_was_closed = True
            else:
                if self.eye_was_closed:
                    # Eye was closed and now opened = one blink
                    self.blink_count += 1
                    blinked_now = True
                    print(f"[BLINK] Detected! Count: {self.blink_count}/{config.REQUIRED_BLINKS}")
                self.eye_was_closed = False

        return {
            "blink_count": self.blink_count,
            "ear": round(ear, 3),
            "blinked_now": blinked_now,
            "face_detected": results.multi_face_landmarks is not None
        }

    def release(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()
