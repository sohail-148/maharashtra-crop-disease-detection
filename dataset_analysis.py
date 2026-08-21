from pathlib import Path
from PIL import Image
import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parent

# Dataset folders
DATASETS = {
    "Tomato": ROOT / "tomato_plantvillage",
    "Grape_Niphad": ROOT / "grape_niphad",
    "Grape_2024": ROOT / "grape_2024",
    "Chilli": ROOT / "chilli_cold",
    "Sugarcane_Maharashtra": ROOT / "sugarcane_maharashtra",
    "Sugarcane_Large": ROOT / "sugarcane_large",
}

# Image extensions we will accept
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

records = []

for dataset_name, dataset_path in DATASETS.items():
    if not dataset_path.exists():
        print(f"\nWARNING: Dataset not found: {dataset_name}")
        print(f"Path: {dataset_path}")
        continue

    print(f"\nScanning: {dataset_name}")

    class_folders = [
    folder for folder in dataset_path.rglob("*")
    if folder.is_dir()
    and any(
        file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        for file in folder.iterdir()
    )
]

    for class_folder in sorted(class_folders):
        image_files = [
            file for file in class_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for image_file in image_files:
            records.append({
                "dataset": dataset_name,
                "class": class_folder.name,
                "file": str(image_file),
            })

        print(f"  {class_folder.name}: {len(image_files)} images")

# Create dataframe
df = pd.DataFrame(records)

if df.empty:
    print("\nNo images were found.")
    raise SystemExit(1)

# Save detailed file list
df.to_csv(ROOT / "dataset_file_list.csv", index=False)

# Create summary
summary = (
    df.groupby(["dataset", "class"])
    .size()
    .reset_index(name="image_count")
)

summary.to_csv(ROOT / "dataset_summary.csv", index=False)

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(summary.to_string(index=False))

print("\n" + "=" * 60)
print("TOTAL IMAGES BY DATASET")
print("=" * 60)

dataset_totals = df.groupby("dataset").size().sort_values(ascending=False)
print(dataset_totals)

print("\n" + "=" * 60)
print(f"TOTAL IMAGES ACROSS ALL DATASETS: {len(df)}")
print("=" * 60)