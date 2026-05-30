"""
Low-Light Preprocessing Pipeline — Advanced.
Multi-stage adaptive enhancement that aggressively boosts extremely dark frames
so that face_recognition can detect & recognise faces even in near-black conditions.
Fully offline — uses OpenCV only.
"""

import cv2
import numpy as np
import config


# ═══════════════════════════════════════════════════════════════════════════════
#  Brightness Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def _calculate_brightness(frame) -> float:
    """Calculate average brightness (0-255) of a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def _classify_brightness(brightness: float) -> str:
    """
    Classify brightness into 5 levels:
      PITCH_BLACK  <  15  → nearly invisible, maximum boost
      VERY_DARK    <  40  → can barely see shapes, aggressive boost
      DARK         <  70  → dim, standard enhancement
      DIM          < 100  → slightly below normal, light touch
      NORMAL       ≥ 100  → no enhancement needed
    """
    if brightness < config.LOW_LIGHT_PITCH_BLACK:
        return "PITCH_BLACK"
    elif brightness < config.LOW_LIGHT_VERY_DARK:
        return "VERY_DARK"
    elif brightness < config.LOW_LIGHT_DARK:
        return "DARK"
    elif brightness < config.LOW_LIGHT_DIM:
        return "DIM"
    else:
        return "NORMAL"


# ═══════════════════════════════════════════════════════════════════════════════
#  Enhancement Building Blocks
# ═══════════════════════════════════════════════════════════════════════════════

def _auto_exposure_boost(frame, target_brightness: float = 90.0):
    """
    Simulate auto-exposure by scaling pixel values so the average brightness
    approaches `target_brightness`.  Handles very dark frames where CLAHE
    alone cannot recover detail.
    """
    current = _calculate_brightness(frame)
    if current < 1.0:
        current = 1.0  # avoid division by zero

    scale = target_brightness / current
    # Clamp scale to prevent blow-out (max 8× boost)
    scale = min(scale, 8.0)

    boosted = cv2.convertScaleAbs(frame, alpha=scale, beta=10)
    return boosted


def _apply_clahe(frame, clip_limit: float = None, grid_size: tuple = None):
    """Apply Contrast Limited Adaptive Histogram Equalization on L channel."""
    if clip_limit is None:
        clip_limit = config.CLAHE_CLIP_LIMIT
    if grid_size is None:
        grid_size = config.CLAHE_GRID_SIZE

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_channel = clahe.apply(l_channel)

    enhanced = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _adaptive_gamma_correction(frame, brightness: float):
    """
    Compute gamma dynamically from actual brightness.
    Darker frames → lower gamma → bigger brightness boost.

    Mapping (approximate):
      brightness  5  → gamma 0.25  (extreme boost)
      brightness 30  → gamma 0.45
      brightness 60  → gamma 0.65
      brightness 90  → gamma 1.00  (no change)
      brightness 120 → gamma 1.20  (slight darken — not applied)
    """
    if brightness >= 100:
        return frame  # no correction needed

    # Map brightness [0..100] → gamma [0.20..1.0]
    gamma = max(0.20, brightness / 100.0)
    # Additional push for very dark
    if brightness < 30:
        gamma *= 0.7

    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(frame, table)


def _apply_noise_reduction(frame, strength: int = None):
    """Apply non-local means denoising (handles noise from brightness boost)."""
    if strength is None:
        strength = config.NOISE_REDUCTION_STRENGTH
    return cv2.fastNlMeansDenoisingColored(
        frame, None, strength, strength, 7, 21
    )


def _apply_light_denoise(frame):
    """Lightweight bilateral filter — faster than NLM, good for moderate noise."""
    return cv2.bilateralFilter(frame, 5, 50, 50)


def _apply_sharpening(frame):
    """Unsharp mask to recover edge detail after denoising/boosting."""
    gaussian = cv2.GaussianBlur(frame, (0, 0), 2.0)
    sharpened = cv2.addWeighted(frame, 1.5, gaussian, -0.5, 0)
    return sharpened


def _enhance_face_region(frame, face_box):
    """
    Apply extra local CLAHE specifically on the face region
    to maximise feature visibility for recognition.

    Args:
        frame: Full BGR image
        face_box: (top, right, bottom, left) or None
    Returns:
        Frame with face region further enhanced
    """
    if face_box is None:
        return frame

    top, right, bottom, left = face_box
    h, w = frame.shape[:2]
    # Add padding around face
    pad = 20
    top = max(0, top - pad)
    left = max(0, left - pad)
    bottom = min(h, bottom + pad)
    right = min(w, right + pad)

    face_roi = frame[top:bottom, left:right]
    if face_roi.size == 0:
        return frame

    # High clip-limit CLAHE on face region
    face_roi = _apply_clahe(face_roi, clip_limit=4.5, grid_size=(4, 4))
    frame[top:bottom, left:right] = face_roi
    return frame


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def enhance_frame(frame, face_box=None):
    """
    Main preprocessing pipeline.
    Applies multi-stage adaptive enhancement based on brightness level.

    Args:
        frame: BGR image (numpy array)
        face_box: Optional (top, right, bottom, left) for face-region boost

    Returns:
        Enhanced BGR image
    """
    if frame is None:
        return frame

    brightness = _calculate_brightness(frame)
    level = _classify_brightness(brightness)

    if level == "PITCH_BLACK":
        # ── STAGE 1: Auto-exposure boost to make *anything* visible ──
        frame = _auto_exposure_boost(frame, target_brightness=80.0)
        # ── STAGE 2: Double-pass CLAHE (progressive detail lift) ──
        frame = _apply_clahe(frame, clip_limit=4.0, grid_size=(4, 4))
        frame = _apply_clahe(frame, clip_limit=3.0, grid_size=(8, 8))
        # ── STAGE 3: Adaptive gamma on the now-brighter frame ──
        new_brightness = _calculate_brightness(frame)
        frame = _adaptive_gamma_correction(frame, new_brightness)
        # ── STAGE 4: Heavy denoise (boosted images are noisy) ──
        frame = _apply_noise_reduction(frame, strength=8)
        # ── STAGE 5: Sharpen to recover edges ──
        frame = _apply_sharpening(frame)
        # ── STAGE 6: Face-region local boost ──
        frame = _enhance_face_region(frame, face_box)

    elif level == "VERY_DARK":
        frame = _auto_exposure_boost(frame, target_brightness=70.0)
        frame = _apply_clahe(frame, clip_limit=3.5, grid_size=(8, 8))
        frame = _adaptive_gamma_correction(frame, _calculate_brightness(frame))
        frame = _apply_noise_reduction(frame, strength=6)
        frame = _apply_sharpening(frame)
        frame = _enhance_face_region(frame, face_box)

    elif level == "DARK":
        frame = _apply_clahe(frame, clip_limit=3.0, grid_size=(8, 8))
        frame = _adaptive_gamma_correction(frame, brightness)
        frame = _apply_noise_reduction(frame)
        frame = _enhance_face_region(frame, face_box)

    elif level == "DIM":
        frame = _apply_clahe(frame)
        frame = _adaptive_gamma_correction(frame, brightness)
        frame = _apply_light_denoise(frame)

    # NORMAL → no enhancement

    return frame


# ═══════════════════════════════════════════════════════════════════════════════
#  Status Reporting
# ═══════════════════════════════════════════════════════════════════════════════

def get_brightness_status(frame) -> str:
    """Return human-readable brightness status with level."""
    brightness = _calculate_brightness(frame)
    level = _classify_brightness(brightness)

    icons = {
        "PITCH_BLACK": "⚫",
        "VERY_DARK": "🌑",
        "DARK": "🌒",
        "DIM": "🌓",
        "NORMAL": "☀️",
    }
    icon = icons.get(level, "")
    return f"{icon} {level} ({brightness:.0f})"


def is_too_dark_for_recognition(frame) -> bool:
    """
    Quick check — is the frame so dark that even after enhancement
    face recognition is unlikely to work?
    Returns True if brightness < 5 (near-total darkness).
    """
    brightness = _calculate_brightness(frame)
    return brightness < 5
