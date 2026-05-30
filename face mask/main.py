"""
Main Application — Advanced Face Recognition System.
Orchestrates webcam capture, recognition, liveness detection,
depth/motion anti-spoofing, attendance logging, and security.
Fully offline operation.
"""

import sys
import os
import time
import cv2
import numpy as np
import face_recognition as fr
from datetime import datetime
import config
from face_encoder import load_encodings
from preprocessing import enhance_frame, get_brightness_status, is_too_dark_for_recognition
from liveness import LivenessDetector
from depth_motion_check import DepthMotionChecker
from database import AttendanceDB
from unknown_handler import UnknownFaceHandler
from security import SecurityManager
from tamper_proof_log import TamperProofLog
from adaptive_learning import AdaptiveLearner


class FaceRecognitionApp:
    """Main application class orchestrating all modules."""

    # ── Verification phases ──
    PHASE_RECOGNITION = "RECOGNITION"
    PHASE_BLINK = "BLINK"
    PHASE_DEPTH = "DEPTH_MOTION"
    PHASE_GRANTED = "GRANTED"
    PHASE_DENIED = "DENIED"

    def __init__(self):
        print("\n" + "=" * 60)
        print("  ADVANCED FACE RECOGNITION SYSTEM")
        print("  Offline · Secure · Adaptive · Anti-Spoof")
        print("=" * 60 + "\n")

        print("[INIT] Loading modules...")
        self.data = load_encodings()
        self.liveness = LivenessDetector()
        self.depth_checker = DepthMotionChecker() if config.DEPTH_CHECK_ENABLED else None
        self.db = AttendanceDB()
        self.unknown_handler = UnknownFaceHandler()
        self.security = SecurityManager()
        self.log = TamperProofLog()
        self.adaptive = AdaptiveLearner()

        if self.security.is_locked():
            print("[INIT] ⚠ System is LOCKED. Use: python admin.py --unlock")
        if not self.data["encodings"]:
            print("[INIT] ⚠ No face encodings! Run: python admin.py --encode")

        print("[INIT] System ready.\n")

    def run(self):
        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

        if not cap.isOpened():
            print("[ERROR] Cannot open camera!")
            sys.exit(1)

        print("[CAMERA] Opened. Press 'q' to quit.\n")

        # ─── State ───────────────────────────────────────────────────
        phase = self.PHASE_RECOGNITION
        attempt_start = time.time()
        blink_count = 0
        name_counter = {}
        last_name = "UNKNOWN"
        last_confidence = 0.0
        last_face_loc = None
        last_unknown_capture = time.time()
        attempt_recorded_this_window = False
        frame_count = 0
        granted_time = None
        confirmed_name = None
        confirmed_confidence = 0.0

        try:
            while True:
                # ─── LOCK CHECK ───────────────────────────────────
                if self.security.is_locked():
                    ret, frame = cap.read()
                    if ret:
                        frame = self.security.draw_lock_screen(frame)
                        cv2.imshow("Face Recognition System", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                now = time.time()
                elapsed = now - attempt_start
                remaining = max(0, int(config.UNKNOWN_TIMEOUT - elapsed))

                # ─── 30s TIMEOUT → full reset ────────────────────
                if elapsed > config.UNKNOWN_TIMEOUT and phase not in (self.PHASE_GRANTED, self.PHASE_DEPTH):
                    if last_name == "UNKNOWN":
                        self.unknown_handler._capture_image(frame, last_face_loc)
                        self.log.append_log("TIMEOUT", "Unknown", 0, "30s expired")
                        print("[SYSTEM] ⏱ 30s expired — unknown face.")
                    else:
                        print(f"[SYSTEM] ⏱ 30s expired — {last_name} didn't complete verification.")
                        self.log.append_log("TIMEOUT", last_name, last_confidence / 100.0,
                                            "Verification not completed")

                    # FULL RESET
                    attempt_start = time.time()
                    blink_count = 0
                    name_counter.clear()
                    last_name = "UNKNOWN"
                    last_confidence = 0.0
                    last_face_loc = None
                    attempt_recorded_this_window = False
                    self.liveness.reset()
                    if self.depth_checker:
                        self.depth_checker.reset()
                    phase = self.PHASE_RECOGNITION
                    continue

                # ─── PREPROCESSING ────────────────────────────────
                enhanced = enhance_frame(frame.copy(), last_face_loc)
                brightness = get_brightness_status(frame)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                display = enhanced.copy()
                h, w = display.shape[:2]

                # ═══════════════════════════════════════════════════
                #  PHASE: ACCESS GRANTED — show 3s then stop
                # ═══════════════════════════════════════════════════
                if phase == self.PHASE_GRANTED:
                    if last_face_loc:
                        top, right, bottom, left = last_face_loc
                        cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 3)
                    overlay = display.copy()
                    cv2.rectangle(overlay, (0, h - 90), (w, h), (0, 120, 0), -1)
                    cv2.addWeighted(overlay, 0.6, display, 0.4, 0, display)
                    cv2.putText(display, f"ACCESS GRANTED: {confirmed_name}",
                                (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(display, f"Confidence: {confirmed_confidence:.1f}% | Attendance Logged",
                                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
                    cv2.imshow("Face Recognition System", display)
                    if granted_time and now - granted_time > 3:
                        print("[SYSTEM] Stopping camera.")
                        break
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # ═══════════════════════════════════════════════════
                #  PHASE: DEPTH / MOTION CHECK
                # ═══════════════════════════════════════════════════
                if phase == self.PHASE_DEPTH and self.depth_checker:
                    self.depth_checker.process_frame(frame)
                    task_info = self.depth_checker.get_current_task_info()

                    # Draw depth/motion UI
                    self._draw_depth_ui(display, task_info, confirmed_name, confirmed_confidence)

                    if self.depth_checker.is_complete():
                        if self.depth_checker.is_passed():
                            # ✅ ALL CHECKS PASSED — grant access
                            phase = self.PHASE_GRANTED
                            granted_time = now
                            self.db.log_attendance(confirmed_name, confirmed_confidence / 100.0)
                            self.log.append_log("ACCESS_GRANTED", confirmed_name,
                                                confirmed_confidence / 100.0,
                                                f"Blinks: {blink_count} | Depth: PASS")
                            self.security.record_attempt(True)
                            print(f"\n✓ ACCESS GRANTED: {confirmed_name} ({confirmed_confidence}%)")
                        else:
                            # ❌ Depth check failed — spoofing suspected
                            phase = self.PHASE_DENIED
                            self.log.append_log("DEPTH_FAIL", confirmed_name,
                                                confirmed_confidence / 100.0,
                                                "Depth/motion check failed — possible spoofing")
                            self.security.record_attempt(False)
                            print(f"\n✗ DEPTH CHECK FAILED: {confirmed_name} — possible spoofing")

                            # Save spoofing image to digital_theft folder
                            try:
                                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                theft_path = os.path.join(
                                    config.DIGITAL_THEFT_DIR,
                                    f"spoof_{confirmed_name}_{ts}.jpg"
                                )
                                cv2.imwrite(theft_path, frame)
                                print(f"[THEFT] Saved spoofing image: {theft_path}")
                            except Exception as e:
                                print(f"[THEFT] Failed to save image: {e}")

                            # Show denial for 3s then reset
                            deny_start = now
                            while now - deny_start < 3:
                                ret2, frame2 = cap.read()
                                if not ret2:
                                    break
                                disp2 = enhance_frame(frame2.copy())
                                h2, w2 = disp2.shape[:2]
                                overlay2 = disp2.copy()
                                cv2.rectangle(overlay2, (0, h2 - 90), (w2, h2), (0, 0, 180), -1)
                                cv2.addWeighted(overlay2, 0.6, disp2, 0.4, 0, disp2)
                                cv2.putText(disp2, "SPOOFING DETECTED — ACCESS DENIED",
                                            (10, h2 - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                            (255, 255, 255), 2)
                                cv2.putText(disp2, "Depth/motion check failed. Try again.",
                                            (10, h2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            (200, 200, 255), 1)
                                cv2.imshow("Face Recognition System", disp2)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                                now = time.time()

                            # Reset everything
                            attempt_start = time.time()
                            blink_count = 0
                            name_counter.clear()
                            last_name = "UNKNOWN"
                            last_confidence = 0.0
                            last_face_loc = None
                            attempt_recorded_this_window = False
                            self.liveness.reset()
                            self.depth_checker.reset()
                            phase = self.PHASE_RECOGNITION

                    cv2.imshow("Face Recognition System", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # ═══════════════════════════════════════════════════
                #  PHASE: RECOGNITION + BLINK (combined)
                # ═══════════════════════════════════════════════════

                # ── BLINK DETECTION (every frame) ──
                try:
                    blink_result = self.liveness.detect_blink(frame)
                    blink_count = blink_result["blink_count"]
                except Exception:
                    pass

                # ── FACE RECOGNITION (every Nth frame) ──
                if frame_count % config.PROCESS_EVERY_N_FRAMES == 0:
                    try:
                        boxes = fr.face_locations(rgb, model=config.FACE_RECOGNITION_MODEL)
                        encodings = fr.face_encodings(rgb, boxes)
                    except Exception as e:
                        print(f"[WARN] Recognition error: {e}")
                        boxes, encodings = [], []

                    for (top, right, bottom, left), encoding in zip(boxes, encodings):
                        name = "UNKNOWN"
                        confidence = 0.0

                        if len(self.data["encodings"]) > 0:
                            distances = fr.face_distance(self.data["encodings"], encoding)
                            min_dist = float(np.min(distances))
                            index = int(np.argmin(distances))
                            confidence = round((1 - min_dist) * 100, 2)

                            if min_dist < 0.50 and confidence > 50:
                                name = self.data["names"][index]

                        if confidence < 45:
                            name = "UNKNOWN"

                        # Multi-frame confirmation
                        if name != "UNKNOWN":
                            name_counter[name] = name_counter.get(name, 0) + 1
                        else:
                            name_counter.clear()

                        last_name = name
                        last_confidence = confidence
                        last_face_loc = (top, right, bottom, left)

                        # ── FACE CONFIRMED + BLINKS DONE → start depth check ──
                        needed_frames = config.PROCESS_EVERY_N_FRAMES * 3
                        if (name != "UNKNOWN"
                                and name_counter.get(name, 0) >= needed_frames
                                and blink_count >= config.REQUIRED_BLINKS):

                            confirmed_name = name
                            confirmed_confidence = confidence

                            if config.DEPTH_CHECK_ENABLED and self.depth_checker:
                                # Start depth/motion verification
                                phase = self.PHASE_DEPTH
                                self.depth_checker.start_session()
                                self.liveness.reset()
                                print(f"\n[DEPTH] Starting depth/motion check for {name}...")
                            else:
                                # Depth check disabled — grant immediately
                                phase = self.PHASE_GRANTED
                                granted_time = now
                                self.db.log_attendance(name, confidence / 100.0)
                                self.log.append_log("ACCESS_GRANTED", name, confidence / 100.0,
                                                    f"Blinks: {blink_count}")
                                self.security.record_attempt(True)
                                self.liveness.reset()
                                print(f"\n✓ ACCESS GRANTED: {name} ({confidence}%)")
                            break

                        # ── Unknown face tracking ──
                        if name == "UNKNOWN" and confidence < 45:
                            if now - last_unknown_capture >= config.UNKNOWN_CAPTURE_INTERVAL:
                                self.unknown_handler.process_unknown(
                                    frame, (top, right, bottom, left))
                                last_unknown_capture = now

                            if not attempt_recorded_this_window:
                                attempt_recorded_this_window = True
                                self.security.record_attempt(False)
                                self.log.append_log("ACCESS_DENIED", "Unknown",
                                                    confidence / 100.0, "Unknown face")

                        # ── Near-miss → adaptive learning ──
                        elif name == "UNKNOWN" and 45 <= confidence < 50:
                            best_name = self.data["names"][
                                int(np.argmin(fr.face_distance(
                                    self.data["encodings"], encoding)))
                            ]
                            self.adaptive.add_to_queue(best_name, confidence / 100.0,
                                                       encoding, frame)

                        if self.security.is_locked():
                            break

                # ─── DRAW UI (recognition + blink phase) ─────────
                self._draw_recognition_ui(display, last_name, last_confidence,
                                          last_face_loc, blink_count, brightness,
                                          remaining, name_counter, phase)

                cv2.imshow("Face Recognition System", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n[SYSTEM] Interrupted.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.liveness.release()
            if self.depth_checker:
                self.depth_checker.release()
            print("[SYSTEM] Camera closed. Goodbye.")

    # ─── UI Drawing Helpers ──────────────────────────────────────────────────

    def _draw_recognition_ui(self, display, name, confidence, face_loc,
                              blink_count, brightness, remaining, name_counter, phase):
        """Draw the recognition + blink phase UI overlay."""
        h, w = display.shape[:2]

        # Top bar
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)
        cv2.putText(display, "Face Recognition System", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(display, f"Light: {brightness}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        sec = self.security.get_status()
        cv2.putText(display, f"Fails: {sec['failed_attempts']}/{sec['max_attempts']}",
                    (w - 130, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        cv2.putText(display, f"Time: {remaining}s",
                    (w - 130, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Phase indicator
        depth_status = " + Depth Check" if config.DEPTH_CHECK_ENABLED else ""
        cv2.putText(display, f"Blinks: {blink_count}/{config.REQUIRED_BLINKS}{depth_status}",
                    (w // 2 - 80, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Face box
        if face_loc is not None:
            top, right, bottom, left = face_loc
            color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)
            cv2.rectangle(display, (left, top), (right, bottom), color, 2)

            label = f"{name} ({confidence:.1f}%)"
            lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(display, (left, top - lsz[1] - 12),
                          (left + lsz[0] + 10, top), color, -1)
            cv2.putText(display, label, (left + 5, top - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if name != "UNKNOWN":
                needed = config.PROCESS_EVERY_N_FRAMES * 3
                confirmed = min(name_counter.get(name, 0), needed)
                status = f"Confirm: {confirmed}/{needed} | Blink to verify"
                cv2.putText(display, status, (left, bottom + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                cv2.putText(display, f"{remaining}s left",
                            (left, bottom + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Bottom bar
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
        cv2.putText(display, "Press 'q' to quit", (w - 160, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    def _draw_depth_ui(self, display, task_info, name, confidence):
        """Draw the depth/motion verification phase UI overlay."""
        h, w = display.shape[:2]

        state = task_info.get("state", "")
        instruction = task_info.get("instruction", "")
        icon = task_info.get("icon", "")
        task_idx = task_info.get("task_index", 0)
        total = task_info.get("total_tasks", 0)
        time_remaining = task_info.get("time_remaining", 0)
        progress = task_info.get("progress", [])

        # ── Top banner — depth check phase indicator ──
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (80, 40, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        cv2.putText(display, f"DEPTH/MOTION CHECK — {name} ({confidence:.1f}%)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        task_progress = f"Task {task_idx + 1}/{total}"
        if state == "ALL_DONE":
            task_progress = "All tasks complete"
        cv2.putText(display, task_progress,
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # ── Task results so far ──
        result_x = 200
        for i, res in enumerate(progress):
            icon_text = "PASS" if res["passed"] else "FAIL"
            color = (0, 255, 0) if res["passed"] else (0, 0, 255)
            cv2.putText(display, f"T{i+1}:{icon_text}",
                        (result_x + i * 100, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # ── Central instruction ──
        if state == "BASELINE":
            # Show "hold still" instruction
            text = "Hold STILL — capturing baseline..."
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2
            # Background box
            cv2.rectangle(display, (text_x - 15, text_y - 35),
                          (text_x + text_size[0] + 15, text_y + 15), (0, 0, 0), -1)
            cv2.putText(display, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        elif state == "TASK_ACTIVE":
            # Show task instruction prominently
            text = f"{icon}  {instruction}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_x = (w - text_size[0]) // 2
            text_y = h // 2

            # Pulsing background
            cv2.rectangle(display, (text_x - 20, text_y - 40),
                          (text_x + text_size[0] + 20, text_y + 20), (0, 80, 0), -1)
            cv2.putText(display, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Timer bar
            timer_fraction = time_remaining / config.DEPTH_TASK_TIMEOUT
            bar_width = int(w * 0.6)
            bar_x = (w - bar_width) // 2
            bar_y = text_y + 40
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15),
                          (50, 50, 50), -1)
            filled = int(bar_width * timer_fraction)
            bar_color = (0, 255, 0) if timer_fraction > 0.3 else (0, 0, 255)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + filled, bar_y + 15),
                          bar_color, -1)
            cv2.putText(display, f"{time_remaining:.1f}s",
                        (bar_x + bar_width + 10, bar_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # ── Bottom bar ──
        overlay = display.copy()
        cv2.rectangle(overlay, (0, h - 30), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
        cv2.putText(display, "Anti-spoofing verification in progress...",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(display, "Press 'q' to quit", (w - 160, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)


def main():
    app = FaceRecognitionApp()
    app.run()


if __name__ == "__main__":
    main()
