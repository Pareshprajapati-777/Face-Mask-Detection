import cv2
import sys
from pathlib import Path

class FaceDetector:
    _instance = None
    _cascade = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceDetector, cls).__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        """Load bundled OpenCV Haar Cascade Face Detector strictly ONCE at startup."""
        base_dir = Path(__file__).resolve().parent
        # Prefer alt2 (significantly lower false positive rate for lights, doors, and background objects)
        cascade_path = base_dir / "cascades" / "haarcascade_frontalface_alt2.xml"
        if not cascade_path.exists():
            cascade_path = base_dir / "cascades" / "haarcascade_frontalface_default.xml"
        if not cascade_path.exists():
            try:
                cascade_path = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
            except Exception:
                pass

        if not cascade_path.exists() or not cascade_path.is_file():
            print(f"[ERROR] Haar Cascade XML missing at: {cascade_path}")
            sys.exit(1)

        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            print("[ERROR] Failed to load OpenCV Haar Cascade Classifier.")
            sys.exit(1)

        print(f"[SUCCESS] OpenCV Face Detector loaded successfully from: {cascade_path.name}")

    def detect_faces(self, gray_image, scale_factor=1.08, min_neighbors=7, min_size=(45, 45)):
        """Detect all visible faces in a grayscale image frame with precision settings."""
        if self._cascade is None or self._cascade.empty():
            return []

        return self._cascade.detectMultiScale(
            gray_image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

