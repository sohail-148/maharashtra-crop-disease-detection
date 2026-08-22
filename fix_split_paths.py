"""
fix_split_paths.py — Rewrite Windows absolute paths in split CSVs for AWS

Run ONCE on the AWS instance after uploading the splits/ directory.
Rewrites file_path column in every CSV in-place so they point to the
correct Linux location.

Usage:
    python fix_split_paths.py --dataset-root /data/datasets

After running, the split CSVs will contain paths like:
    /data/datasets/grape_niphad/Healthy Leaves/img.jpg

The original Windows paths (D:\\CropDiseaseProject\\...) will be gone.
This script only touches the splits/ directory — datasets are not affected.

NOTE: train_experiment.py also does this rewriting in-memory at runtime,
so running this script is OPTIONAL.  It is useful if you want the fixed
CSVs permanently on disk (e.g. for inspection or re-running without flags).
"""

import os
import sys
import argparse
import pandas as pd

SPLIT_DIRS = [
    "grape_niphad",
    "grape_2024",
    "chilli_cold",
    "sugarcane_maharashtra",
    "sugarcane_large",
]
SPLIT_FILES = ["train.csv", "val.csv", "test.csv"]


def fix_paths_in_df(df: pd.DataFrame, new_root: str) -> tuple[pd.DataFrame, int]:
    """
    Replace Windows absolute prefix in file_path column.
    Returns (fixed_df, count_of_changed_rows).
    """
    changed = 0

    def _fix(path: str) -> str:
        nonlocal changed
        # Normalise to forward slashes
        p = path.replace("\\", "/")
        # Find the dataset folder — scan for known folder names
        for folder in SPLIT_DIRS:
            marker = "/" + folder + "/"
            idx = p.find(marker)
            if idx != -1:
                relative = p[idx + 1:]   # e.g. grape_niphad/Healthy Leaves/img.jpg
                new_path = os.path.join(new_root, relative).replace("\\", "/")
                if new_path != path:
                    changed += 1
                return new_path
        return p  # unchanged if no known folder found

    df = df.copy()
    df["file_path"] = df["file_path"].apply(_fix)
    return df, changed


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite Windows paths in split CSVs for Linux/AWS"
    )
    parser.add_argument(
        "--dataset-root", "-d",
        required=True,
        help="Absolute path on Linux that contains the dataset folders, "
             "e.g. /data/datasets",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        help="Path to the splits/ directory. Defaults to splits/ next to this script.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change but do not write files.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    splits_root = args.splits_dir or os.path.join(script_dir, "splits")

    if not os.path.isdir(splits_root):
        print(f"ERROR: splits directory not found: {splits_root}")
        sys.exit(1)

    print(f"\nDataset root  : {args.dataset_root}")
    print(f"Splits dir    : {splits_root}")
    print(f"Dry run       : {args.dry_run}")
    print()

    total_changed = 0

    for exp_dir in SPLIT_DIRS:
        for fname in SPLIT_FILES:
            fpath = os.path.join(splits_root, exp_dir, fname)
            if not os.path.exists(fpath):
                print(f"  SKIP (not found): {fpath}")
                continue

            df = pd.read_csv(fpath)
            if "file_path" not in df.columns:
                print(f"  SKIP (no file_path column): {fpath}")
                continue

            fixed_df, n = fix_paths_in_df(df, args.dataset_root)
            total_changed += n

            if n == 0:
                print(f"  OK (already correct): {exp_dir}/{fname}")
                continue

            # Show a sample before/after
            sample_before = df["file_path"].iloc[0]
            sample_after  = fixed_df["file_path"].iloc[0]
            print(f"  {exp_dir}/{fname}  ({n} paths)")
            print(f"    before: {sample_before}")
            print(f"    after : {sample_after}")

            if not args.dry_run:
                fixed_df.to_csv(fpath, index=False)
                print(f"    → written.")

    print(f"\nTotal paths changed: {total_changed}")
    if args.dry_run:
        print("Dry run — no files written.")
    else:
        print("Done. All split CSVs updated.")


if __name__ == "__main__":
    main()
