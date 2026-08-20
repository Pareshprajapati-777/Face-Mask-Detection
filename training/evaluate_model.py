import os
import json
from pathlib import Path
import sys

# Add parent to path so evaluate_model can import from training/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_HEIGHT = 224
IMG_WIDTH  = 224
BATCH_SIZE = 32

BASE_DIR    = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR  = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


def locate_dataset_root(dataset_dir: Path) -> Path:
    for item in dataset_dir.iterdir():
        if item.is_dir() and any(k in item.name.lower()
                                  for k in ["mask", "with_mask", "without_mask"]):
            return dataset_dir
    return dataset_dir


def evaluate():
    model_path  = MODELS_DIR / "face_mask_model.keras"
    config_path = MODELS_DIR / "class_names.json"

    if not model_path.exists():
        print(f"[ERROR] Model not found at: {model_path}")
        print("Run 'npm run train' first.")
        return

    if not config_path.exists():
        print(f"[ERROR] class_names.json not found at: {config_path}")
        return

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" MODEL EVALUATION & METRICS ")
    print("=" * 60)

    model = tf.keras.models.load_model(model_path)
    print(f"[SUCCESS] Loaded model from: {model_path}")

    with open(config_path, "r") as f:
        class_config = json.load(f)

    data_root = locate_dataset_root(DATASET_DIR)

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
    val_gen = val_datagen.flow_from_directory(
        str(data_root),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation',
        shuffle=False
    )

    print(f"\n[INFO] Evaluating on {val_gen.samples} validation images...")
    val_gen.reset()
    raw_preds = model.predict(val_gen, verbose=1)

    y_pred = np.argmax(raw_preds, axis=1)
    y_true = val_gen.classes

    class_names = [class_config[str(i)] for i in sorted(class_config.keys(), key=int)]
    cm = confusion_matrix(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 50)
    print(" METRICS SUMMARY ")
    print("=" * 50)
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  Precision : {precision * 100:.2f}%")
    print(f"  Recall    : {recall * 100:.2f}%")
    print(f"  F1 Score  : {f1 * 100:.2f}%")

    print("\n" + "-" * 60)
    print(" CLASSIFICATION REPORT ")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names,
                                digits=4, zero_division=0))

    # Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar=False, annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('PREDICTED', fontsize=12, fontweight='bold')
    plt.ylabel('ACTUAL', fontsize=12, fontweight='bold')
    plt.tight_layout()

    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"\n[INFO] Saved confusion matrix to: {cm_path}")


if __name__ == "__main__":
    evaluate()
