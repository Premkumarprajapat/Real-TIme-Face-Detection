"""
Depth & Motion Anti-Spoofing — Random Task Verification.
Asks the user to perform random physical movements and verifies
that face metrics change in ways consistent with a real 3D person,
not a flat screen replay.

Uses MediaPipe Face Mesh (fully offline).
Uses pyttsx3 for offline text-to-speech announcements.
"""

import cv2
import math
import time
import random
import threading
import numpy as np
import mediapipe as mp
import pyttsx3
import config


# ═══════════════════════════════════════════════════════════════════════════════
#  Text-to-Speech Engine (non-blocking)
# ═══════════════════════════════════════════════════════════════════════════════

_tts_lock = threading.Lock()


def _speak_async(text: str):
    """
    Speak text aloud in a background thread so the camera loop
    doesn't freeze while the voice plays.
    """
    def _run():
        with _tts_lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 160)   # slightly slower for clarity
                engine.setProperty('volume', 1.0)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"[TTS] Speech error (non-fatal): {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ═══════════════════════════════════════════════════════════════════════════════
#  Task Definitions
# ═══════════════════════════════════════════════════════════════════════════════

TASKS = [
    {
        "id": "move_closer",
        "instruction": "Move your face CLOSER to the camera",
        "icon": "🔍",
        "metric": "face_area",
        "expected_direction": "increase",
        "description": "Face should get bigger",
    },
    {
        "id": "move_away",
        "instruction": "Move your face AWAY from the camera",
        "icon": "🔭",
        "metric": "face_area",
        "expected_direction": "decrease",
        "description": "Face should get smaller",
    },
    {
        "id": "turn_left",
        "instruction": "Turn your head to the LEFT",
        "icon": "👈",
        "metric": "nose_x",
        "expected_direction": "increase",   # webcam mirrors — user's left = image right
        "description": "Nose shifts right in mirrored view",
    },
    {
        "id": "turn_right",
        "instruction": "Turn your head to the RIGHT",
        "icon": "👉",
        "metric": "nose_x",
        "expected_direction": "decrease",   # webcam mirrors — user's right = image left
        "description": "Nose shifts left in mirrored view",
    },
    {
        "id": "look_up",
        "instruction": "Look UP slowly",
        "icon": "👆",
        "metric": "nose_y",
        "expected_direction": "decrease",
        "description": "Nose should shift up",
    },
    {
        "id": "look_down",
        "instruction": "Look DOWN slowly",
        "icon": "👇",
        "metric": "nose_y",
        "expected_direction": "increase",
        "description": "Nose should shift down",
    },
    {
        "id": "tilt_left",
        "instruction": "Tilt your head to your LEFT SHOULDER",
        "icon": "↩️",
        "metric": "eye_angle",
        "expected_direction": "any",        # accept tilt in either direction
        "description": "Eye line should tilt",
    },
    {
        "id": "tilt_right",
        "instruction": "Tilt your head to your RIGHT SHOULDER",
        "icon": "↪️",
        "metric": "eye_angle",
        "expected_direction": "any",        # accept tilt in either direction
        "description": "Eye line should tilt",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Face Metrics Extraction
# ═══════════════════════════════════════════════════════════════════════════════

# MediaPipe Face Mesh key landmarks
NOSE_TIP = 1
LEFT_EYE_OUTER = 263
RIGHT_EYE_OUTER = 33
FOREHEAD = 10
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454


class FaceMetrics:
    """Extract relevant metrics from MediaPipe Face Mesh landmarks."""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame) -> dict:
        """
        Extract face metrics from a frame.

        Returns dict with:
            face_detected: bool
            face_area: float  (bounding box area as fraction of frame)
            nose_x: float     (nose tip X as fraction of frame width)
            nose_y: float     (nose tip Y as fraction of frame height)
            eye_angle: float  (angle of line between outer eye corners, degrees)
            face_width: float (cheek-to-cheek distance as fraction of frame width)
            face_height: float (forehead-to-chin distance as fraction of frame height)
            landmarks: list   (raw landmark data)
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return {"face_detected": False}

        face = results.multi_face_landmarks[0].landmark

        # Nose tip position (normalised 0-1)
        nose_x = face[NOSE_TIP].x
        nose_y = face[NOSE_TIP].y

        # Eye outer corners for angle calculation
        left_eye = (face[LEFT_EYE_OUTER].x * w, face[LEFT_EYE_OUTER].y * h)
        right_eye = (face[RIGHT_EYE_OUTER].x * w, face[RIGHT_EYE_OUTER].y * h)

        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]
        eye_angle = math.degrees(math.atan2(dy, dx))

        # Face bounding box from cheeks and forehead/chin
        forehead_y = face[FOREHEAD].y
        chin_y = face[CHIN].y
        left_cheek_x = face[LEFT_CHEEK].x
        right_cheek_x = face[RIGHT_CHEEK].x

        face_width = abs(left_cheek_x - right_cheek_x)
        face_height = abs(chin_y - forehead_y)
        face_area = face_width * face_height  # normalised (fraction of frame)

        return {
            "face_detected": True,
            "face_area": face_area,
            "nose_x": nose_x,
            "nose_y": nose_y,
            "eye_angle": eye_angle,
            "face_width": face_width,
            "face_height": face_height,
        }

    def release(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  Depth / Motion Checker — State Machine
# ═══════════════════════════════════════════════════════════════════════════════

class DepthMotionChecker:
    """
    Random-task-based depth & motion anti-spoofing.

    Usage:
        checker = DepthMotionChecker()
        checker.start_session()           # picks random tasks

        while not checker.is_complete():
            metrics = checker.process_frame(frame)  # feed frames
            task_info = checker.get_current_task_info()
            # draw task_info["instruction"] on screen

        result = checker.get_result()     # {"passed": bool, "details": [...]}
        checker.release()
    """

    # States
    IDLE = "IDLE"
    BASELINE = "BASELINE"           # Collecting baseline metrics (1s)
    TASK_ACTIVE = "TASK_ACTIVE"     # User performing task
    MEASURING = "MEASURING"         # Evaluating result
    TASK_DONE = "TASK_DONE"         # Single task evaluated
    ALL_DONE = "ALL_DONE"           # All tasks completed

    def __init__(self):
        self.face_metrics = FaceMetrics()
        self.state = self.IDLE
        self.selected_tasks = []
        self.current_task_idx = 0
        self.task_results = []

        # Timing
        self.baseline_start = 0
        self.task_start = 0

        # Metrics buffers
        self.baseline_samples = []
        self.task_samples = []

        # Config
        self.num_tasks = config.DEPTH_TASKS_COUNT
        self.task_timeout = config.DEPTH_TASK_TIMEOUT
        self.baseline_duration = 1.5  # seconds to collect baseline

    def start_session(self):
        """Pick random tasks and begin the verification session."""
        # Select N non-conflicting random tasks
        self.selected_tasks = self._pick_random_tasks(self.num_tasks)
        self.current_task_idx = 0
        self.task_results = []
        self.state = self.BASELINE
        self.baseline_start = time.time()
        self.baseline_samples = []
        self.task_samples = []
        print(f"[DEPTH] Session started — {self.num_tasks} tasks selected")
        for t in self.selected_tasks:
            print(f"  → {t['icon']} {t['instruction']}")

        # Announce start via TTS
        _speak_async("Depth verification starting. Hold still for baseline.")

    def _pick_random_tasks(self, count: int) -> list:
        """
        Pick `count` non-conflicting random tasks.
        Avoid picking both 'move_closer' and 'move_away' etc.
        """
        conflict_pairs = [
            ("move_closer", "move_away"),
            ("turn_left", "turn_right"),
            ("look_up", "look_down"),
            ("tilt_left", "tilt_right"),
        ]

        available = list(TASKS)
        selected = []

        for _ in range(count):
            if not available:
                break
            task = random.choice(available)
            selected.append(task)
            available.remove(task)

            # Remove conflicting task
            for pair in conflict_pairs:
                if task["id"] in pair:
                    conflict_id = pair[0] if task["id"] == pair[1] else pair[1]
                    available = [t for t in available if t["id"] != conflict_id]

        return selected

    def process_frame(self, frame) -> dict:
        """
        Process a single frame. Call this every frame during verification.

        Returns current metrics dict (or empty if no face).
        """
        if self.state in (self.IDLE, self.ALL_DONE):
            return {}

        metrics = self.face_metrics.extract(frame)
        if not metrics.get("face_detected", False):
            return metrics

        now = time.time()

        # ── BASELINE COLLECTION ──
        if self.state == self.BASELINE:
            self.baseline_samples.append(metrics)
            if now - self.baseline_start >= self.baseline_duration:
                # Transition to TASK_ACTIVE
                self.state = self.TASK_ACTIVE
                self.task_start = time.time()
                self.task_samples = []
                task = self.selected_tasks[self.current_task_idx]
                print(f"[DEPTH] Baseline collected ({len(self.baseline_samples)} samples)")
                print(f"[DEPTH] Task {self.current_task_idx + 1}: {task['instruction']}")

                # 🔊 SPEAK the instruction aloud
                _speak_async(task["instruction"])

        # ── TASK ACTIVE — collecting movement samples ──
        elif self.state == self.TASK_ACTIVE:
            self.task_samples.append(metrics)

            # Check timeout
            if now - self.task_start >= self.task_timeout:
                self.state = self.MEASURING

        # ── MEASURING — evaluate the completed task ──
        if self.state == self.MEASURING:
            result = self._evaluate_task(
                self.selected_tasks[self.current_task_idx],
                self.baseline_samples,
                self.task_samples,
            )
            self.task_results.append(result)
            print(f"[DEPTH] Task {self.current_task_idx + 1} result: "
                  f"{'✅ PASS' if result['passed'] else '❌ FAIL'} — {result['reason']}")

            # Advance to next task or finish
            self.current_task_idx += 1
            if self.current_task_idx >= len(self.selected_tasks):
                self.state = self.ALL_DONE
                print(f"[DEPTH] All tasks done. Overall: "
                      f"{'PASS' if self.is_passed() else 'FAIL'}")
            else:
                # Collect new baseline for next task
                self.state = self.BASELINE
                self.baseline_start = time.time()
                self.baseline_samples = []
                self.task_samples = []

        return metrics

    def _evaluate_task(self, task: dict, baseline: list, samples: list) -> dict:
        """
        Compare baseline metrics to task-period metrics.
        Determine if the change is consistent with a real person.
        """
        if not baseline or not samples:
            return {"task": task["id"], "passed": False,
                    "reason": "Insufficient data"}

        metric_key = task["metric"]
        direction = task["expected_direction"]

        # Extract metric values
        baseline_vals = [s[metric_key] for s in baseline if metric_key in s]
        task_vals = [s[metric_key] for s in samples if metric_key in s]

        if not baseline_vals or not task_vals:
            return {"task": task["id"], "passed": False,
                    "reason": "No face metrics captured"}

        baseline_mean = np.mean(baseline_vals)
        baseline_std = np.std(baseline_vals)

        # Use the peak (most extreme) value during the task to catch the movement
        if direction == "increase":
            task_peak = np.max(task_vals)
            change = task_peak - baseline_mean
        else:
            task_peak = np.min(task_vals)
            change = baseline_mean - task_peak

        # Determine threshold based on metric type
        if metric_key == "face_area":
            # Face area is normalised (fraction of frame), ~0.02 - 0.15
            threshold = config.DEPTH_SIZE_CHANGE_THRESHOLD * baseline_mean
            abs_change = abs(task_peak - baseline_mean)
            passed = change > 0 and abs_change >= threshold
            reason = (f"Area: baseline={baseline_mean:.4f}, "
                      f"peak={task_peak:.4f}, change={abs_change:.4f}, "
                      f"threshold={threshold:.4f}")

        elif metric_key in ("nose_x", "nose_y"):
            # Nose position is normalised 0-1
            threshold = config.DEPTH_POSITION_SHIFT_THRESHOLD / 640.0  # convert pixels to normalised
            abs_change = abs(task_peak - baseline_mean)
            passed = change > 0 and abs_change >= threshold
            reason = (f"Nose: baseline={baseline_mean:.4f}, "
                      f"peak={task_peak:.4f}, shift={abs_change:.4f}, "
                      f"threshold={threshold:.4f}")

        elif metric_key == "eye_angle":
            # Eye angle in degrees — use absolute change (tilt either way counts)
            threshold = config.DEPTH_ANGLE_CHANGE_THRESHOLD
            # For tilt, check max deviation in EITHER direction from baseline
            max_val = float(np.max(task_vals))
            min_val = float(np.min(task_vals))
            abs_change = max(abs(max_val - baseline_mean), abs(baseline_mean - min_val))
            passed = abs_change >= threshold  # any direction counts
            reason = (f"Angle: baseline={baseline_mean:.1f}°, "
                      f"max={max_val:.1f}°, min={min_val:.1f}°, change={abs_change:.1f}°, "
                      f"threshold={threshold}°")
        else:
            passed = False
            reason = f"Unknown metric: {metric_key}"

        return {
            "task": task["id"],
            "passed": passed,
            "reason": reason,
            "change": float(abs(task_peak - baseline_mean)),
        }

    # ─── Status Queries ───────────────────────────────────────────────────────

    def is_complete(self) -> bool:
        """All tasks have been evaluated."""
        return self.state == self.ALL_DONE

    def is_passed(self) -> bool:
        """Did the user pass ALL tasks?"""
        if not self.task_results:
            return False
        return all(r["passed"] for r in self.task_results)

    def get_result(self) -> dict:
        """Get final verification result."""
        return {
            "passed": self.is_passed(),
            "tasks_completed": len(self.task_results),
            "tasks_required": len(self.selected_tasks),
            "details": self.task_results,
        }

    def get_current_task_info(self) -> dict:
        """
        Get info about the current task (for UI display).

        Returns dict with:
            state: current state string
            task_index: 0-based index of current task
            total_tasks: total number of tasks
            instruction: text to show the user
            icon: emoji icon
            time_remaining: seconds left for current task
            progress: list of completed task results
        """
        if self.state == self.IDLE:
            return {"state": self.IDLE, "instruction": "Waiting to start..."}

        if self.state == self.ALL_DONE:
            return {
                "state": self.ALL_DONE,
                "instruction": "Verification complete!",
                "passed": self.is_passed(),
                "progress": self.task_results,
            }

        task = self.selected_tasks[self.current_task_idx]
        now = time.time()

        if self.state == self.BASELINE:
            remaining = max(0, self.baseline_duration - (now - self.baseline_start))
            return {
                "state": self.BASELINE,
                "task_index": self.current_task_idx,
                "total_tasks": len(self.selected_tasks),
                "instruction": "Hold still... capturing baseline",
                "icon": "📷",
                "time_remaining": remaining,
                "progress": self.task_results,
            }

        if self.state == self.TASK_ACTIVE:
            remaining = max(0, self.task_timeout - (now - self.task_start))
            return {
                "state": self.TASK_ACTIVE,
                "task_index": self.current_task_idx,
                "total_tasks": len(self.selected_tasks),
                "instruction": task["instruction"],
                "icon": task["icon"],
                "time_remaining": remaining,
                "progress": self.task_results,
            }

        return {
            "state": self.state,
            "task_index": self.current_task_idx,
            "total_tasks": len(self.selected_tasks),
            "instruction": "Processing...",
            "icon": "⏳",
            "time_remaining": 0,
            "progress": self.task_results,
        }

    def reset(self):
        """Reset to idle state."""
        self.state = self.IDLE
        self.selected_tasks = []
        self.current_task_idx = 0
        self.task_results = []
        self.baseline_samples = []
        self.task_samples = []

    def release(self):
        """Release MediaPipe resources."""
        self.face_metrics.release()
