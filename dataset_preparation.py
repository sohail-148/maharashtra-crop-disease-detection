"""
Phase 6 — Dataset Preparation: Tomato (T1)

Creates a stratified 70/15/15 train/validation/test split
for the Tomato PlantVillage dataset.

Rules:
  - No images are copied, moved, or modified.
  - No augmented images are written to disk.
  - Output is three CSV files under splits/tomato/:
        train.csv
        val.csv
        test.csv
  Each CSV has columns: file_path, class_label, class_index

Reproducible via RANDOM_SEED = 42.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "tomato_plantvillage"
SPLITS_DIR  = ROOT / "splits" / "tomato"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15   # remainder after train+val

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Step 1 — Discover images and classes
# ---------------------------------------------------------------------------

def discover_dataset(dataset_dir: Path) -> pd.DataFrame:
    """
    Walk each immediate sub-directory of dataset_dir.
    Each sub-directory name is the class label.
    Returns a DataFrame with columns: file_path, class_label
    """
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found:\n  {dataset_dir}")
        sys.exit(1)

    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

    if not class_dirs:
        print(f"ERROR: No class sub-directories found in:\n  {dataset_dir}")
        sys.exit(1)

    rows = []
    print("\n" + "=" * 60)
    print("TOMATO DATASET — CLASS INVENTORY")
    print("=" * 60)
    print(f"{'Class':<50} {'Images':>7}")
    print("-" * 60)

    for class_dir in class_dirs:
        images = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not images:
            print(f"  WARNING: No images in {class_dir.name}, skipping.")
            continue

        print(f"{class_dir.name:<50} {len(images):>7,}")

        for img_path in images:
            rows.append({
                "file_path":   str(img_path),
                "class_label": class_dir.name,
            })

    df = pd.DataFrame(rows)

    print("-" * 60)
    print(f"{'TOTAL':<50} {len(df):>7,}")
    print(f"{'Classes found':<50} {df['class_label'].nunique():>7}")
    print("=" * 60)

    return df

# ---------------------------------------------------------------------------
# Step 2 — Build class index mapping
# ---------------------------------------------------------------------------

def build_class_index(df: pd.DataFrame) -> dict:
    """Map each class label to an integer index (sorted alphabetically)."""
    classes = sorted(df["class_label"].unique())
    return {label: idx for idx, label in enumerate(classes)}

# ---------------------------------------------------------------------------
# Step 3 — Stratified split
# ---------------------------------------------------------------------------

def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train / val / test using stratified sampling.
    Ratios: 70% / 15% / 15%
    """
    # First cut: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),   # 0.30
        stratify=df["class_label"],
        random_state=RANDOM_SEED,
    )

    # Second cut: val vs test  (equal halves of the 30%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),   # 0.50 of the 30%
        stratify=temp_df["class_label"],
        random_state=RANDOM_SEED,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Step 4 — Verify split integrity
# ---------------------------------------------------------------------------

def verify_splits(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> None:
    """
    Checks:
      1. No image appears in more than one split.
      2. Total image count is preserved.
      3. Class proportions are roughly maintained in each split.
    """
    print("\n" + "=" * 60)
    print("SPLIT INTEGRITY CHECKS")
    print("=" * 60)

    # Check 1 — no duplicates across splits
    train_files = set(train_df["file_path"])
    val_files   = set(val_df["file_path"])
    test_files  = set(test_df["file_path"])

    overlap_tv = train_files & val_files
    overlap_tt = train_files & test_files
    overlap_vt = val_files   & test_files

    if overlap_tv or overlap_tt or overlap_vt:
        print("FAIL — overlapping files detected:")
        if overlap_tv: print(f"  Train ∩ Val  : {len(overlap_tv)} files")
        if overlap_tt: print(f"  Train ∩ Test : {len(overlap_tt)} files")
        if overlap_vt: print(f"  Val   ∩ Test : {len(overlap_vt)} files")
        sys.exit(1)
    else:
        print("PASS — no image appears in more than one split")

    # Check 2 — total count preserved
    combined = len(train_df) + len(val_df) + len(test_df)
    if combined != len(full_df):
        print(f"FAIL — image count mismatch: {combined} vs {len(full_df)}")
        sys.exit(1)
    else:
        print(f"PASS — total image count preserved ({combined:,})")

    # Check 3 — class coverage
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = set(full_df["class_label"].unique()) - set(split_df["class_label"].unique())
        if missing:
            print(f"WARN  — {split_name} is missing classes: {missing}")
        else:
            print(f"PASS — all 10 classes present in {split_name}")

    print("=" * 60)

# ---------------------------------------------------------------------------
# Step 5 — Print per-class distribution table
# ---------------------------------------------------------------------------

def print_distribution(
    full_df:  pd.DataFrame,
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> None:
    """Print a per-class breakdown across all three splits."""
    print("\n" + "=" * 78)
    print("PER-CLASS SPLIT DISTRIBUTION")
    print("=" * 78)
    header = f"{'Class':<50} {'Total':>6}  {'Train':>6}  {'Val':>5}  {'Test':>5}"
    print(header)
    print("-" * 78)

    for cls in sorted(full_df["class_label"].unique()):
        total = (full_df["class_label"]  == cls).sum()
        trn   = (train_df["class_label"] == cls).sum()
        val   = (val_df["class_label"]   == cls).sum()
        tst   = (test_df["class_label"]  == cls).sum()
        print(f"{cls:<50} {total:>6,}  {trn:>6,}  {val:>5,}  {tst:>5,}")

    print("-" * 78)
    print(
        f"{'TOTAL':<50} {len(full_df):>6,}  "
        f"{len(train_df):>6,}  {len(val_df):>5,}  {len(test_df):>5,}"
    )
    print("=" * 78)

    # Percentage check
    print(f"\n  Train : {len(train_df)/len(full_df)*100:.1f}%  "
          f"Val : {len(val_df)/len(full_df)*100:.1f}%  "
          f"Test : {len(test_df)/len(full_df)*100:.1f}%")

# ---------------------------------------------------------------------------
# Step 6 — Save CSVs
# ---------------------------------------------------------------------------

def save_splits(
    train_df:     pd.DataFrame,
    val_df:       pd.DataFrame,
    test_df:      pd.DataFrame,
    class_index:  dict,
) -> None:
    """Add class_index column and write the three CSV files."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_df = split_df.copy()
        out_df["class_index"] = out_df["class_label"].map(class_index)

        # Column order
        out_df = out_df[["file_path", "class_label", "class_index"]]

        out_path = SPLITS_DIR / f"{split_name}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}  ({len(out_df):,} rows)")

    # Also save class index mapping
    mapping_path = SPLITS_DIR / "class_index.csv"
    mapping_df = pd.DataFrame(
        [{"class_label": k, "class_index": v} for k, v in class_index.items()]
    ).sort_values("class_index")
    mapping_df.to_csv(mapping_path, index=False)
    print(f"  Saved: {mapping_path}  ({len(mapping_df)} classes)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\nPhase 6 — Dataset Preparation: Tomato (T1)")
    print(f"Dataset : {DATASET_DIR}")
    print(f"Output  : {SPLITS_DIR}")
    print(f"Split   : {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)}")
    print(f"Seed    : {RANDOM_SEED}")

    # 1. Discover
    full_df = discover_dataset(DATASET_DIR)

    # 2. Class index
    class_index = build_class_index(full_df)

    # 3. Split
    train_df, val_df, test_df = stratified_split(full_df)

    # 4. Verify
    verify_splits(full_df, train_df, val_df, test_df)

    # 5. Print distribution
    print_distribution(full_df, train_df, val_df, test_df)

    # 6. Save
    print("\nSaving split CSVs...")
    save_splits(train_df, val_df, test_df, class_index)

    print("\nDone. No images were copied or modified.")


if __name__ == "__main__":
    main()
