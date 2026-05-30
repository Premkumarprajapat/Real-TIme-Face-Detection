"""
Core Face Recognition Engine.
Compares live face encodings against stored embeddings.
Fully offline using face_recognition (dlib).
"""

import cv2
import numpy as np
import face_recognition
import config


def recognize_face(frame, known_data: dict) -> tuple:
    """
    Recognize a face in the given frame.

    Args:
        frame: BGR image (numpy array)
        known_data: dict with "names" and "encodings" lists

    Returns:
        (name, confidence, face_location) or ("Unknown", 0.0, None)
    """
    if not known_data["encodings"]:
        return ("Unknown", 0.0, None)

    # Convert BGR to RGB — MUST be contiguous for dlib 20+
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations
    face_locations = face_recognition.face_locations(
        rgb_frame, model=config.FACE_RECOGNITION_MODEL
    )

    if not face_locations:
        return ("No Face", 0.0, None)

    # Process only the largest face (by area)
    if len(face_locations) > 1:
        face_locations = [_largest_face(face_locations)]

    face_location = face_locations[0]

    # Generate encoding for detected face
    try:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    except Exception as e:
        print(f"[RECOGNIZER] Encoding error (skipping frame): {e}")
        return ("Unknown", 0.0, face_location)

    if not face_encodings:
        return ("Unknown", 0.0, face_location)

    face_encoding = face_encodings[0]

    # Compare against known encodings
    distances = face_recognition.face_distance(
        known_data["encodings"], face_encoding
    )

    if len(distances) == 0:
        return ("Unknown", 0.0, face_location)

    # Find best match
    best_match_idx = np.argmin(distances)
    best_distance = distances[best_match_idx]
    confidence = 1.0 - best_distance  # Convert distance to confidence

    if confidence >= config.CONFIDENCE_THRESHOLD:
        name = known_data["names"][best_match_idx]
        return (name, confidence, face_location)
    else:
        return ("Unknown", confidence, face_location)


def _largest_face(face_locations: list) -> tuple:
    """Return the face location with the largest area."""
    def area(loc):
        top, right, bottom, left = loc
        return (bottom - top) * (right - left)

    return max(face_locations, key=area)


def draw_recognition(frame, name: str, confidence: float, face_location, status: str = ""):
    """
    Draw recognition result overlay on frame.

    Args:
        frame: BGR image
        name: Recognized name
        confidence: Confidence score (0.0 - 1.0)
        face_location: (top, right, bottom, left)
        status: Additional status text
    """
    import cv2

    if face_location is None:
        # No face — show status at top
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame

    top, right, bottom, left = face_location

    # Color based on recognition status
    if name == "Unknown":
        color = (0, 0, 255)      # Red
    elif confidence >= config.CONFIDENCE_THRESHOLD:
        color = (0, 255, 0)      # Green
    else:
        color = (0, 255, 255)    # Yellow

    # Draw face rectangle
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Name and confidence label
    conf_pct = f"{confidence * 100:.1f}%"
    label = f"{name} ({conf_pct})"

    # Label background
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame, (left, top - label_size[1] - 10),
                  (left + label_size[0] + 10, top), color, -1)
    cv2.putText(frame, label, (left + 5, top - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Status text
    if status:
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame
