import os
import sys
import zipfile
from pathlib import Path

"""
==================================================
KAGGLE DATASET IDENTIFIER

henrylydecker/face-masks
==================================================
"""

KAGGLE_DATASET_ID = "henrylydecker/face-masks"

def download_and_extract():
    """Download Kaggle dataset using local authentication and extract to dataset/."""
    base_dir = Path(__file__).resolve().parent.parent
    dataset_dir = base_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" KAGGLE DATASET DOWNLOAD & EXTRACTION ")
    print("=" * 60)
    print(f"[INFO] Target Dataset: {KAGGLE_DATASET_ID}")

    # Check for existing ZIP file
    zip_files = list(dataset_dir.glob("*.zip")) + list(base_dir.glob("*.zip"))
    if zip_files:
        zip_path = zip_files[0]
        print(f"[INFO] Found ZIP archive: {zip_path}")
        print(f"[INFO] Extracting to: {dataset_dir}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            print("[SUCCESS] Dataset archive extracted successfully!")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to extract {zip_path}: {e}")

    # Attempt Kaggle API download
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("[SUCCESS] Kaggle credentials authenticated!")
        print(f"[INFO] Downloading dataset files to '{dataset_dir}'...")
        api.dataset_download_files(KAGGLE_DATASET_ID, path=str(dataset_dir), unzip=True)
        print("[SUCCESS] Kaggle dataset downloaded and extracted!")
        return True
    except Exception as err:
        print(f"[WARNING] Kaggle API download notice: {err}")
        print("Fallback: Place Kaggle zip file inside 'dataset/' or set up ~/.kaggle/kaggle.json")
        return False

if __name__ == "__main__":
    download_and_extract()
