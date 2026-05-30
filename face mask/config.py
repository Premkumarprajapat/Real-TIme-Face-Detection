"""
Configuration for the Advanced Face Recognition System.
All tuneable parameters are centralized here.
"""

import os

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
ENCODINGS_DIR = os.path.join(BASE_DIR, "encodings")
UNKNOWN_FACES_DIR = os.path.join(BASE_DIR, "unknown_faces")
DIGITAL_THEFT_DIR = os.path.join(BASE_DIR, "digital_theft")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ─── File Paths ───────────────────────────────────────────────────────────────
ENCODINGS_FILE = os.path.join(ENCODINGS_DIR, "face_encodings.pkl")
DATABASE_FILE = os.path.join(DATA_DIR, "attendance.db")
ACCESS_LOG_FILE = os.path.join(LOGS_DIR, "access_log.jsonl")
REVIEW_QUEUE_FILE = os.path.join(DATA_DIR, "review_queue.pkl")
ADMIN_PASSWORD_FILE = os.path.join(DATA_DIR, "admin.key")
LOCK_STATE_FILE = os.path.join(DATA_DIR, "lock_state.json")

# ─── Recognition ──────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60          # Minimum confidence to grant access (60%)
FACE_RECOGNITION_MODEL = "hog"       # "hog" (CPU) or "cnn" (GPU) — fully offline
FACE_ENCODING_JITTERS = 1            # Number of re-samples for encoding (higher = slower but better)

# ─── Liveness Detection ──────────────────────────────────────────────────────
REQUIRED_BLINKS = 2                  # Number of blinks required to confirm liveness
EAR_THRESHOLD = 0.21                 # Eye Aspect Ratio below this = eye closed
EAR_CONSECUTIVE_FRAMES = 2           # Frames the eye must be below threshold to count as blink
LIVENESS_TIMEOUT = 20                # Seconds to complete blink challenge

# ─── Unknown Face Handling ────────────────────────────────────────────────────
UNKNOWN_CAPTURE_INTERVAL = 10        # Seconds between unknown face captures
UNKNOWN_TIMEOUT = 50                 # Seconds before full session reset

# ─── Security & Anti-Spoofing ─────────────────────────────────────────────────
MAX_FAILED_ATTEMPTS = 5              # Failed attempts before LOCK MODE
DEFAULT_ADMIN_PASSWORD = "admin123"  # Default admin password (change on first run!)

# ─── Low-Light Preprocessing (Multi-Level) ───────────────────────────────────
GAMMA_VALUE = 1.8                    # Base gamma correction value
CLAHE_CLIP_LIMIT = 3.0               # Default CLAHE clip limit
CLAHE_GRID_SIZE = (8, 8)             # Default CLAHE grid size
NOISE_REDUCTION_STRENGTH = 5         # Strength of non-local means denoising
LOW_LIGHT_PITCH_BLACK = 15           # Avg brightness < 15 → near invisible
LOW_LIGHT_VERY_DARK = 40             # Avg brightness < 40 → barely see shapes
LOW_LIGHT_DARK = 70                  # Avg brightness < 70 → dim
LOW_LIGHT_DIM = 100                  # Avg brightness < 100 → slightly below normal
LOW_LIGHT_THRESHOLD = 100            # Legacy alias (kept for compatibility)

# ─── Depth / Motion Anti-Spoofing ────────────────────────────────────────────
DEPTH_CHECK_ENABLED = True           # Toggle depth/motion check on/off
DEPTH_TASKS_COUNT = 2                # Number of random tasks per session
DEPTH_TASK_TIMEOUT = 5               # Seconds per task to perform movement
DEPTH_SIZE_CHANGE_THRESHOLD = 0.12   # 12% face area change required (move closer/away)
DEPTH_POSITION_SHIFT_THRESHOLD = 10  # Pixels of nose shift required (turn/look)
DEPTH_ANGLE_CHANGE_THRESHOLD = 4     # Degrees of eye-line tilt required

# ─── Adaptive Learning ───────────────────────────────────────────────────────
ADAPTIVE_LOW_THRESHOLD = 0.40        # Matches between 40%–60% go to review queue
ADAPTIVE_HIGH_THRESHOLD = 0.60       # Same as CONFIDENCE_THRESHOLD

# ─── Camera ───────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0                     # Default camera index
FRAME_WIDTH = 640                    # Capture width
FRAME_HEIGHT = 480                   # Capture height
PROCESS_EVERY_N_FRAMES = 2           # Skip frames for performance

# ─── Ensure directories exist ─────────────────────────────────────────────────
for _dir in [KNOWN_FACES_DIR, ENCODINGS_DIR, UNKNOWN_FACES_DIR, DIGITAL_THEFT_DIR, LOGS_DIR, DATA_DIR]:
    os.makedirs(_dir, exist_ok=True)
