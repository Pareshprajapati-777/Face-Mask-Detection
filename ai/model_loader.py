import json
import sys
from pathlib import Path
import tensorflow as tf
from ai.config import MODEL_PATH, CLASS_NAMES_PATH

class ModelLoader:
    _instance = None
    _model = None
    _class_names = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        """Load TensorFlow Keras model and class names JSON strictly ONCE."""
        if not MODEL_PATH.exists():
            print(f"[ERROR] Model file missing at: {MODEL_PATH}")
            print("Please train the model first: npm run train")
            sys.exit(1)

        if not CLASS_NAMES_PATH.exists():
            print(f"[ERROR] Class names JSON missing at: {CLASS_NAMES_PATH}")
            print("Please train the model first: npm run train")
            sys.exit(1)

        print(f"[INFO] Loading TensorFlow Keras Model from: {MODEL_PATH}")
        try:
            self._model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"[ERROR] Failed to load Keras model: {e}")
            sys.exit(1)

        with open(CLASS_NAMES_PATH, "r") as f:
            self._class_names = json.load(f)

        print(f"[SUCCESS] Model and class mapping loaded! Classes: {self._class_names}")

    @property
    def model(self):
        return self._model

    @property
    def class_names(self):
        return self._class_names

    @property
    def is_loaded(self):
        return self._model is not None and self._class_names is not None
