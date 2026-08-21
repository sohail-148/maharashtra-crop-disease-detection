"""
Dataset splitting and standardization pipeline.

Creates stratified 70/15/15 train/validation/test splits
for all crop datasets without physically copying files.

Outputs:
- data_splits.json  : file paths for each split
- split_summary.csv : class-wise counts per split
"""

import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parent

# Dataset folders (rename these if your folder names differ)
DATASETS = {
    "Tomato": ROOT / "tomato_plantvillage",
    "Grape_Niphad": ROOT / "grape_niphad",
    "Grape_2024": ROOT / "grape_2024",
    "Chilli": ROOT / "chilli_cold",
    "Sugarcane_Maharashtra": ROOT / "sugarcane_maharashtra",
    "Sugarcane_Large": ROOT / "sugarcane_large",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42


def collect_images(dataset_path):
    """Collect all image files organized by class folder."""
    class_images = {}

    if not dataset_path.exists():
        print(f"WARNING: Dataset not found: {dataset_path}")
        return class_images

    for class_folder in sorted(dataset_path.rglob("*")):
        if not class_folder.is_dir():
            continue

        image_files = [
            str(f.resolve())
            for f in class_folder.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if image_files:
            class_images[class_folder.name] = image_files

    return class_images


def create_stratified_split(class_images, dataset_name):
    """Create stratified train/val/test splits for one dataset."""
    records = []

    for class_name, image_paths in class_images.items():
        for img_path in image_paths:
            records.append({
                "dataset": dataset_name,
                "class": class_name,
                "file": img_path,
            })

    df = pd.DataFrame(records)

    if df.empty:
        print(f"WARNING: No images found for {dataset_name}")
        return None, None, None

    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=df["class"],
        random_state=RANDOM_SEED,
    )

    # Second split: val vs test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=temp_df["class"],
        random_state=RANDOM_SEED,
    )

    return train_df, val_df, test_df


def main():
    all_splits = {}
    summary_rows = []

    for dataset_name, dataset_path in DATASETS.items():
        print(f"\nProcessing: {dataset_name}")

        class_images = collect_images(dataset_path)

        if not class_images:
            continue

        total_images = sum(len(imgs) for imgs in class_images.values())
        print(f"  Total images: {total_images}")
        print(f"  Classes: {len(class_images)}")

        train_df, val_df, test_df = create_stratified_split(class_images, dataset_name)

        if train_df is None:
            continue

        all_splits[dataset_name] = {
            "train": train_df.to_dict(orient="records"),
            "validation": val_df.to_dict(orient="records"),
            "test": test_df.to_dict(orient="records"),
        }

        # Build summary
        for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
            class_counts = split_df["class"].value_counts().to_dict()
            for class_name, count in class_counts.items():
                summary_rows.append({
                    "dataset": dataset_name,
                    "split": split_name,
                    "class": class_name,
                    "count": count,
                })

        print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Save splits
    splits_path = ROOT / "data_splits.json"
    with open(splits_path, "w") as f:
        json.dump(all_splits, f, indent=2)
    print(f"\nSaved splits to: {splits_path}")

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_path = ROOT / "split_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    pivot = summary_df.pivot_table(
        index=["dataset", "class"],
        columns="split",
        values="count",
        fill_value=0,
    ).reset_index()

    print(pivot.to_string(index=False))

    print("\n" + "=" * 70)
    print("TOTAL PER SPLIT")
    print("=" * 70)
    totals = summary_df.groupby(["dataset", "split"])["count"].sum().unstack(fill_value=0)
    print(totals.to_string())


if __name__ == "__main__":
    main()