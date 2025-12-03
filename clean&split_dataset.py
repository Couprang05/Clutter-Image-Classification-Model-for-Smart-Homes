"""
Steps performed:
1) Reads mit_indoor_clutter_dataset.csv (image_path, clutter_level)
2) Validate images (remove missing / corrupted)
3) Produce cleaned CSV: mit_indoor_clutter_dataset_clean.csv
4) Add numeric label column (label: 0=low,1=medium,2=high)
5) Stratified split into train/val/test (70/15/15)
6) Save train.csv, val.csv, test.csv under dataset/processed/

"""

import os
import sys
import shutil
import random
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Path to the CSV you generated earlier (text labels)
INPUT_CSV = Path("dataset/processed/mit_indoor_clutter_dataset.csv")

# Root folder for images is used to check existence; script trusts image_path column but will
# treat it relative to project root if a relative path is present.
ROOT_DIR = Path(".").resolve()  # adjust if needed

# Output processed directory (will be created if missing)
PROCESSED_DIR = Path("dataset/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Filenames to be saved
CLEAN_CSV = PROCESSED_DIR / "mit_indoor_clutter_dataset_clean.csv"
LABELLED_CSV = PROCESSED_DIR / "mit_indoor_clutter_dataset_clean_labelled.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

# Label mapping
LABEL_MAP = {"low": 0, "medium": 1, "high": 2}
# -------------------------------------------------------

def is_image_valid(image_path: Path) -> tuple[bool, str]:
    """
    Returns (True, "") if valid image. Otherwise (False, reason).
    """
    try:
        # Use PIL verify to check integrity
        with Image.open(image_path) as im:
            im.verify()
            # Optionally, reopen to check size (some broken files pass verify)
        # Reopen to get size (safer)
        with Image.open(image_path) as im:
            w, h = im.size
            if w < 16 or h < 16:
                return False, f"too_small ({w}x{h})"
    except FileNotFoundError:
        return False, "missing"
    except UnidentifiedImageError:
        return False, "unidentified"
    except Exception as e:
        return False, f"error:{str(e)}"
    return True, ""

def load_input_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: Input CSV not found at: {path.resolve()}")
        sys.exit(1)
    df = pd.read_csv(path)
    if "image_path" not in df.columns or "clutter_level" not in df.columns:
        print("ERROR: INPUT CSV must contain columns: 'image_path' and 'clutter_level'")
        sys.exit(1)
    return df

def resolve_path(p: str) -> Path:
    p = str(p)
    # If absolute path, return as-is
    p_path = Path(p)
    if p_path.is_absolute():
        return p_path
    # else assume relative to project root
    return (ROOT_DIR / p_path).resolve()

def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    good_rows = []
    bad_rows = []
    print(f"Validating {len(df)} image files ...")
    for _, r in tqdm(df.iterrows(), total=len(df)):
        img_p = str(r["image_path"])
        resolved = resolve_path(img_p)
        valid, reason = is_image_valid(resolved)
        if valid:
            # save absolute resolved path (safer for downstream)
            good_rows.append({"image_path": str(resolved), "clutter_level": r["clutter_level"]})
        else:
            bad_rows.append({"image_path": str(resolved), "clutter_level": r["clutter_level"], "reason": reason})
    print(f"Validation complete: good={len(good_rows)}, bad={len(bad_rows)}")
    if len(bad_rows) > 0:
        bad_log = PROCESSED_DIR / "bad_files_log.csv"
        pd.DataFrame(bad_rows).to_csv(bad_log, index=False)
        print(f"List of bad/missing images saved to: {bad_log}")
    df_good = pd.DataFrame(good_rows)
    return df_good

def add_numeric_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # sanitize textual labels to lower and strip
    df["clutter_level"] = df["clutter_level"].astype(str).str.lower().str.strip()
    # Check for unexpected labels
    unexpected = sorted(set(df["clutter_level"].unique()) - set(LABEL_MAP.keys()))
    if unexpected:
        print(f"WARNING: Found unexpected clutter labels: {unexpected}")
        print("These will be mapped to 'low' by default. (Edit script if different behavior desired.)")
        df["clutter_level"] = df["clutter_level"].apply(lambda x: x if x in LABEL_MAP else "low")
    df["label"] = df["clutter_level"].map(LABEL_MAP)
    return df

def stratified_split_and_save(df: pd.DataFrame):
    # Ensure there are at least one sample per class
    counts = df["label"].value_counts().to_dict()
    print("Class counts (label -> count):", counts)
    if min(counts.values()) < 3:
        print("WARNING: Very small class counts found. Stratified splitting may fail or be unstable.")
    # First split: train / temp (70 / 30)
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    # Second split: temp -> val/test (50/50 of temp => each 15% of full)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED)

    # Save
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    print(f"Saved: {TRAIN_CSV} ({len(train_df)})")
    print(f"Saved: {VAL_CSV}   ({len(val_df)})")
    print(f"Saved: {TEST_CSV}  ({len(test_df)})")

def main():
    print("=== dataset_clean_split.py started ===")
    df_in = load_input_csv(INPUT_CSV)
    print(f"Loaded input CSV with {len(df_in)} rows.")
    # Step 1: validate images
    df_clean = validate_and_clean(df_in)
    # Save intermediate cleaned CSV
    df_clean.to_csv(CLEAN_CSV, index=False)
    print(f"Cleaned CSV saved to: {CLEAN_CSV} ({len(df_clean)})")
    # Step 2: add numeric labels
    df_labelled = add_numeric_label(df_clean)
    df_labelled.to_csv(LABELLED_CSV, index=False)
    print(f"Labelled CSV saved to: {LABELLED_CSV} ({len(df_labelled)})")
    # Step 3: stratified split
    stratified_split_and_save(df_labelled)
    print("=== done ===")

if __name__ == "__main__":
    main()