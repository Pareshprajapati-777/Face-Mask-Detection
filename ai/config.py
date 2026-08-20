import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "face_mask_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

# Inference Settings
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.55))
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Server Settings
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))
