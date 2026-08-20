import os
import sys
import shutil
import hashlib
from pathlib import Path
from PIL import Image

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def analyze_dataset():
    """Inspect dataset, report counts per class, clean corrupted files, and verify structure."""
    base_dir = Path(__file__).resolve().parent.parent
    dataset_dir = base_dir / "dataset"
    outputs_dir = base_dir / "outputs" / "training"
    quarantine_dir = base_dir / "outputs" / "corrupted"

    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" DATASET VALIDATION AND ANALYSIS ")
    print("=" * 60)

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found at: {dataset_dir}")
        print("Please place image class subdirectories inside 'dataset/'.")
        return

    class_paths = {}
    for p in dataset_dir.glob("*"):
        if p.is_dir():
            p_name = p.name.lower()
            if "without" in p_name or "no_mask" in p_name or "without_mask" in p_name:
                class_paths["NO_MASK"] = p
            elif "with" in p_name or "mask" in p_name:
                class_paths["MASK"] = p

    if len(class_paths) < 2:
        print(f"[ERROR] Found insufficient class folders in {dataset_dir}.")
        print("Expected folders such as 'with_mask' and 'without_mask'.")
        return

    print("\n[SUCCESS] Class Directories Identified:")
    for cls_name, cls_path in class_paths.items():
        print(f"  - Class '{cls_name}': {cls_path}")

    class_counts = {}
    corrupted_count = 0
    total_valid = 0

    for cls_name, cls_path in class_paths.items():
        files = [f for f in cls_path.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
        valid_count = 0
        for fpath in files:
            try:
                with Image.open(fpath) as img:
                    img.verify()
                valid_count += 1
                total_valid += 1
            except Exception as e:
                corrupted_count += 1
                print(f"[CORRUPTED FILE] {fpath.name}: {e}")
                quarantine_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(fpath), str(quarantine_dir / fpath.name))
        class_counts[cls_name] = valid_count

    print("\n" + "=" * 40)
    print(" DATASET SUMMARY ")
    print("=" * 40)
    for cls_name, count in class_counts.items():
        print(f"  {cls_name:<10}: {count} images")
    print(f"  Total Valid: {total_valid} images")
    print(f"  Corrupted Cleaned: {corrupted_count} images")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    analyze_dataset()
