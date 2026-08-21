"""
Phase 6 — Dataset Preparation: All Six Experiments

Creates stratified 70/15/15 train/validation/test splits
for every crop dataset.

Rules:
  - No images are copied, moved, or modified.
  - No augmented images are written to disk.
  - Output is four CSV files per dataset under splits/<name>/:
        train.csv, val.csv, test.csv, class_index.csv
  - Each CSV has columns: file_path, class_label, class_index
  - Datasets are kept strictly separate (no merging).
  - Source class folder names are used as-is (no renaming).

Reproducible via RANDOM_SEED = 42.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------------------------------------------------------------------
# Dataset registry
#
# Each entry: (experiment_id, dataset_folder, output_name, nested)
#   nested=False  → class folders are direct children of dataset_folder
#   nested=True   → class folders may be one level deeper (sugarcane_large)
# ---------------------------------------------------------------------------

DATASETS = [
    ("T1",  "tomato_plantvillage",   "tomato",               False),
    ("G1",  "grape_niphad",          "grape_niphad",          False),
    ("G2",  "grape_2024",            "grape_2024",            False),
    ("C1",  "chilli_cold",           "chilli_cold",           False),
    ("S1",  "sugarcane_maharashtra", "sugarcane_maharashtra", False),
    ("S2",  "sugarcane_large",       "sugarcane_large",       True),
]

# ---------------------------------------------------------------------------
# Step 1 — Discover images and classes
# ---------------------------------------------------------------------------

def discover_dataset(dataset_dir: Path, nested: bool) -> pd.DataFrame:
    """
    Collect all images organised by class.

    nested=False: class folders are immediate children of dataset_dir.
    nested=True:  walk up to two levels to find folders that contain images
                  (handles sugarcane_large's Diseases/ sub-group).
    """
    if not dataset_dir.exists():
        print(f"  ERROR: directory not found: {dataset_dir}")
        sys.exit(1)

    if nested:
        # Collect every directory that directly contains image files
        class_dirs = sorted([
            d for d in dataset_dir.rglob("*")
            if d.is_dir() and any(
                f.suffix.lower() in IMAGE_EXTENSIONS
                for f in d.iterdir() if f.is_file()
            )
        ], key=lambda d: d.name)
    else:
        class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

    rows = []
    print(f"\n  {'Class':<50} {'Images':>7}")
    print(f"  {'-'*58}")

    for class_dir in class_dirs:
        images = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not images:
            print(f"  WARNING: no images in '{class_dir.name}', skipping.")
            continue

        print(f"  {class_dir.name:<50} {len(images):>7,}")
        for img_path in images:
            rows.append({
                "file_path":   str(img_path),
                "class_label": class_dir.name,
            })

    df = pd.DataFrame(rows)
    print(f"  {'-'*58}")
    print(f"  {'TOTAL':<50} {len(df):>7,}")
    print(f"  {'Classes':<50} {df['class_label'].nunique():>7}")
    return df


# ---------------------------------------------------------------------------
# Step 2 — Class index mapping (alphabetical)
# ---------------------------------------------------------------------------

def build_class_index(df: pd.DataFrame) -> dict:
    return {label: idx for idx, label in enumerate(sorted(df["class_label"].unique()))}


# ---------------------------------------------------------------------------
# Step 3 — Stratified 70/15/15 split
# ---------------------------------------------------------------------------

def stratified_split(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["class_label"],
        random_state=RANDOM_SEED,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_df["class_label"],
        random_state=RANDOM_SEED,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Step 4 — Integrity verification
# ---------------------------------------------------------------------------

def verify_splits(full_df, train_df, val_df, test_df, experiment_id):
    errors = []

    # No cross-split overlap
    train_f = set(train_df["file_path"])
    val_f   = set(val_df["file_path"])
    test_f  = set(test_df["file_path"])
    if train_f & val_f:  errors.append(f"Train∩Val overlap: {len(train_f & val_f)} files")
    if train_f & test_f: errors.append(f"Train∩Test overlap: {len(train_f & test_f)} files")
    if val_f   & test_f: errors.append(f"Val∩Test overlap: {len(val_f & test_f)} files")

    # Total count preserved
    combined = len(train_df) + len(val_df) + len(test_df)
    if combined != len(full_df):
        errors.append(f"Count mismatch: {combined} vs {len(full_df)}")

    # All classes present in every split
    all_classes = set(full_df["class_label"].unique())
    for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
        missing = all_classes - set(sdf["class_label"].unique())
        if missing:
            errors.append(f"{name} missing classes: {missing}")

    if errors:
        print(f"\n  INTEGRITY FAILURES ({experiment_id}):")
        for e in errors:
            print(f"    ✗ {e}")
        sys.exit(1)
    else:
        print(f"  PASS — no overlap | count preserved ({combined:,}) | all classes in all splits")


# ---------------------------------------------------------------------------
# Step 5 — Per-class distribution table
# ---------------------------------------------------------------------------

def print_distribution(full_df, train_df, val_df, test_df):
    print(f"\n  {'Class':<50} {'Total':>6}  {'Train':>6}  {'Val':>5}  {'Test':>5}")
    print(f"  {'-'*76}")
    for cls in sorted(full_df["class_label"].unique()):
        total = (full_df["class_label"]  == cls).sum()
        trn   = (train_df["class_label"] == cls).sum()
        val   = (val_df["class_label"]   == cls).sum()
        tst   = (test_df["class_label"]  == cls).sum()
        print(f"  {cls:<50} {total:>6,}  {trn:>6,}  {val:>5,}  {tst:>5,}")
    print(f"  {'-'*76}")
    print(
        f"  {'TOTAL':<50} {len(full_df):>6,}  "
        f"{len(train_df):>6,}  {len(val_df):>5,}  {len(test_df):>5,}"
    )
    print(
        f"\n  Ratios →  Train: {len(train_df)/len(full_df)*100:.1f}%  "
        f"Val: {len(val_df)/len(full_df)*100:.1f}%  "
        f"Test: {len(test_df)/len(full_df)*100:.1f}%"
    )


# ---------------------------------------------------------------------------
# Step 6 — Save CSVs
# ---------------------------------------------------------------------------

def save_splits(train_df, val_df, test_df, class_index, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out_df = split_df.copy()
        out_df["class_index"] = out_df["class_label"].map(class_index)
        out_df = out_df[["file_path", "class_label", "class_index"]]
        out_path = out_dir / f"{split_name}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path.relative_to(ROOT)}  ({len(out_df):,} rows)")

    mapping_df = pd.DataFrame(
        [{"class_label": k, "class_index": v} for k, v in class_index.items()]
    ).sort_values("class_index")
    mapping_path = out_dir / "class_index.csv"
    mapping_df.to_csv(mapping_path, index=False)
    print(f"  Saved: {mapping_path.relative_to(ROOT)}  ({len(mapping_df)} classes)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_dataset(experiment_id, folder_name, output_name, nested, skip_if_exists=False):
    dataset_dir = ROOT / folder_name
    out_dir     = ROOT / "splits" / output_name

    # Optional skip for already-completed datasets
    if skip_if_exists and (out_dir / "train.csv").exists():
        print(f"\n[{experiment_id}] {output_name} — already split, skipping.")
        return None

    print(f"\n{'='*70}")
    print(f"[{experiment_id}]  {output_name}")
    print(f"{'='*70}")

    full_df     = discover_dataset(dataset_dir, nested)
    class_index = build_class_index(full_df)
    train_df, val_df, test_df = stratified_split(full_df)

    print(f"\n  Integrity checks:")
    verify_splits(full_df, train_df, val_df, test_df, experiment_id)

    print_distribution(full_df, train_df, val_df, test_df)

    print(f"\n  Saving CSVs to splits/{output_name}/")
    save_splits(train_df, val_df, test_df, class_index, out_dir)

    return {
        "experiment": experiment_id,
        "dataset":    output_name,
        "classes":    full_df["class_label"].nunique(),
        "total":      len(full_df),
        "train":      len(train_df),
        "val":        len(val_df),
        "test":       len(test_df),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", "-e",
        help="Run only this experiment ID (e.g. T1, G1). Omit to run all.",
        default=None,
    )
    args = parser.parse_args()

    print("\nPhase 6 — Dataset Preparation: All Experiments")
    print(f"Split: 70/15/15  |  Seed: {RANDOM_SEED}  |  No images copied.")

    summaries = []
    for (exp_id, folder, output, nested) in DATASETS:
        if args.experiment and exp_id != args.experiment:
            continue
        result = process_dataset(exp_id, folder, output, nested, skip_if_exists=False)
        if result:
            summaries.append(result)

    if summaries:
        print(f"\n\n{'='*70}")
        print("PHASE 6 — COMPLETE SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Exp':<4}  {'Dataset':<25} {'Classes':>7}  {'Total':>7}  {'Train':>7}  {'Val':>5}  {'Test':>5}")
        print(f"  {'-'*66}")
        grand_total = grand_train = grand_val = grand_test = 0
        for s in summaries:
            print(
                f"  {s['experiment']:<4}  {s['dataset']:<25} {s['classes']:>7}  "
                f"{s['total']:>7,}  {s['train']:>7,}  {s['val']:>5,}  {s['test']:>5,}"
            )
            grand_total += s["total"]
            grand_train += s["train"]
            grand_val   += s["val"]
            grand_test  += s["test"]
        print(f"  {'-'*66}")
        print(
            f"  {'TOTAL':<30} {grand_total:>7,}  "
            f"{grand_train:>7,}  {grand_val:>5,}  {grand_test:>5,}"
        )
        print(f"{'='*70}")
        print("\nDone. No images were copied or modified.")


if __name__ == "__main__":
    main()
