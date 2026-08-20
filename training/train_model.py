"""
==================================================
FACE MASK DETECTION - TRAINING PIPELINE
MobileNetV2 Transfer Learning + Fine-Tuning
==================================================
Architecture:
  MobileNetV2 (ImageNet pretrained, frozen) -> GlobalAveragePooling2D
  -> BatchNormalization -> Dense(256, relu) -> Dropout(0.5)
  -> Dense(128, relu) -> Dropout(0.3) -> Dense(2, softmax)

Phase 1: Train classifier head only (frozen base)
Phase 2: Fine-tune top 50 layers of MobileNetV2

Dataset: 35,000+ real Kaggle face images
Target:  >95% validation accuracy
==================================================
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
IMG_HEIGHT    = 224
IMG_WIDTH     = 224
BATCH_SIZE    = 32
EPOCHS_HEAD   = 10      # Phase 1: train classifier head only
EPOCHS_FINE   = 15      # Phase 2: fine-tune top layers
UNFREEZE_FROM = 100     # Unfreeze MobileNetV2 layers from this index onward

BASE_DIR    = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR  = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs" / "training"


# ─────────────────────────────────────────
# LOCATE DATASET ROOT
# ─────────────────────────────────────────
def locate_dataset_root(dataset_dir: Path) -> Path:
    for item in dataset_dir.iterdir():
        if item.is_dir() and any(k in item.name.lower() for k in ["mask", "with_mask", "without_mask"]):
            return dataset_dir
    return dataset_dir


# ─────────────────────────────────────────
# BUILD DATA GENERATORS (train + val + class weights)
# ─────────────────────────────────────────
def build_generators(data_root: Path):
    # Aggressive augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.75, 1.25],
        shear_range=0.1,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2
    )

    print("\n[INFO] Loading Training Dataset (80%)...")
    train_gen = train_datagen.flow_from_directory(
        str(data_root),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='training',
        shuffle=True,
        seed=42
    )

    print("[INFO] Loading Validation Dataset (20%)...")
    val_gen = val_datagen.flow_from_directory(
        str(data_root),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation',
        shuffle=False,
        seed=42
    )

    # ── Class weights for imbalanced datasets ──
    class_counts  = {c: 0 for c in train_gen.class_indices.values()}
    for cls_name, cls_idx in train_gen.class_indices.items():
        cls_path = data_root / cls_name
        if cls_path.exists():
            class_counts[cls_idx] = sum(1 for f in cls_path.iterdir()
                                        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])

    total = sum(class_counts.values())
    n_classes = len(class_counts)
    class_weights = {
        cls_idx: total / (n_classes * count) if count > 0 else 1.0
        for cls_idx, count in class_counts.items()
    }
    print(f"[INFO] Class weights: {class_weights}")

    return train_gen, val_gen, class_weights


# ─────────────────────────────────────────
# BUILD MODEL (MobileNetV2 transfer learning)
# ─────────────────────────────────────────
def build_model() -> tf.keras.Model:
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    base_model.trainable = False   # Freeze entire base for Phase 1

    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="input_image")

    # Preprocess for MobileNetV2 (scale [0,1] → [-1, 1])
    x = tf.keras.layers.Lambda(
        lambda img: (img * 2.0) - 1.0, name="mobilenet_preprocess"
    )(inputs)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="bn_1")(x)

    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4), name="dense_256")(x)
    x = layers.Dropout(0.5, name="dropout_1")(x)

    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(1e-4), name="dense_128")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)

    outputs = layers.Dense(2, activation='softmax', name="output_softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="FaceMaskDetector_MobileNetV2")
    return model, base_model


# ─────────────────────────────────────────
# SAVE CLASS MAPPING
# ─────────────────────────────────────────
def save_class_mapping(train_gen, models_dir: Path):
    raw = train_gen.class_indices
    class_names_map = {}
    print("\n" + "=" * 60)
    print(" CLASS MAPPING ")
    print("=" * 60)
    for folder_name, idx in raw.items():
        fn_lower = folder_name.lower()
        if "without" in fn_lower or "no" in fn_lower:
            label = "NO_MASK"
        else:
            label = "MASK"
        class_names_map[str(idx)] = label
        print(f"  Index {idx} → Folder '{folder_name}' → '{label}'")

    config_path = models_dir / "class_names.json"
    with open(config_path, "w") as f:
        json.dump(class_names_map, f, indent=4)
    print(f"[SUCCESS] Saved class mapping to: {config_path}")
    return class_names_map


# ─────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────
def make_callbacks(model_save_path: str, monitor_metric='val_accuracy', patience_es=5, patience_lr=3):
    return [
        callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor=monitor_metric,
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience_es,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1
        )
    ]


# ─────────────────────────────────────────
# PLOT TRAINING HISTORY
# ─────────────────────────────────────────
def plot_history(histories, outputs_dir: Path):
    # Merge Phase 1 + Phase 2 history
    combined = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    for h in histories:
        for k in combined:
            if k in h.history:
                combined[k].extend(h.history[k])

    epochs = range(1, len(combined['accuracy']) + 1)
    phase1_len = len(histories[0].history['accuracy'])

    plt.figure(figsize=(14, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, combined['accuracy'],     color='#22c55e', lw=2, label='Train Accuracy')
    plt.plot(epochs, combined['val_accuracy'], color='#3b82f6', lw=2, label='Val Accuracy')
    plt.axvline(x=phase1_len, color='#f59e0b', lw=1.5, linestyle='--', label='Fine-tune Start')
    plt.title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, combined['loss'],     color='#ef4444', lw=2, label='Train Loss')
    plt.plot(epochs, combined['val_loss'], color='#8b5cf6', lw=2, label='Val Loss')
    plt.axvline(x=phase1_len, color='#f59e0b', lw=1.5, linestyle='--', label='Fine-tune Start')
    plt.title('Training & Validation Loss', fontsize=13, fontweight='bold')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = outputs_dir / "accuracy_loss.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved accuracy & loss plot to: {plot_path}")


# ─────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────
def train():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    data_root = locate_dataset_root(DATASET_DIR)
    print(f"[INFO] Dataset root: {data_root}")

    train_gen, val_gen, class_weights = build_generators(data_root)
    save_class_mapping(train_gen, MODELS_DIR)

    model, base_model = build_model()
    model.summary()

    model_save_path = str(MODELS_DIR / "face_mask_model.keras")

    # ──────────────────────────
    # PHASE 1: Train Head Only
    # ──────────────────────────
    print("\n" + "=" * 60)
    print(" PHASE 1 — TRAINING CLASSIFIER HEAD (Base Frozen) ")
    print("=" * 60)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history1 = model.fit(
        train_gen,
        epochs=EPOCHS_HEAD,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=make_callbacks(model_save_path, patience_es=5, patience_lr=3)
    )

    print(f"\n[INFO] Phase 1 Best Val Accuracy: "
          f"{max(history1.history.get('val_accuracy', [0])):.4f}")

    # ──────────────────────────
    # PHASE 2: Fine-Tuning
    # ──────────────────────────
    print("\n" + "=" * 60)
    print(f" PHASE 2 — FINE-TUNING (Unfreeze layers {UNFREEZE_FROM}+) ")
    print("=" * 60)

    base_model.trainable = True
    for layer in base_model.layers[:UNFREEZE_FROM]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"[INFO] Unfrozen MobileNetV2 layers: {trainable_count}")

    # Reload best Phase 1 weights before fine-tuning
    model.load_weights(model_save_path)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),  # lower LR for fine-tuning
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_gen,
        epochs=EPOCHS_FINE,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=make_callbacks(model_save_path, patience_es=6, patience_lr=3)
    )

    print(f"\n[INFO] Phase 2 Best Val Accuracy: "
          f"{max(history2.history.get('val_accuracy', [0])):.4f}")

    print(f"\n[SUCCESS] Best model saved to: {model_save_path}")

    plot_history([history1, history2], OUTPUTS_DIR)


if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    train()
