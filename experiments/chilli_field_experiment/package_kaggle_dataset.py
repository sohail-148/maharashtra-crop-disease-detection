"""
package_kaggle_dataset.py — Packages all data required for Kaggle GPU execution of Chilli Experiment 4
Creates: experiments/chilli_field_experiment/chilli_exp4_kaggle_dataset.zip
"""
import os
import zipfile
import pandas as pd

PROJECT_ROOT = r"D:\CropDiseaseProject"
EXP_DIR = os.path.join(PROJECT_ROOT, "experiments", "chilli_field_experiment")
OUT_ZIP = os.path.join(EXP_DIR, "chilli_exp4_kaggle_dataset.zip")

manifest_path = os.path.join(EXP_DIR, "field_data_manifest.csv")
val_path = os.path.join(PROJECT_ROOT, "splits", "chilli_cold", "val.csv")
test_path = os.path.join(PROJECT_ROOT, "splits", "chilli_cold", "test.csv")
ext_path = os.path.join(PROJECT_ROOT, "results", "model_robustness_audit", "external_provenance.csv")
rw_path = os.path.join(PROJECT_ROOT, "results", "model_robustness_audit", "realworld_provenance.csv")

print("Packaging Kaggle dataset...")
with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
    # 1. Add CSV manifests
    z.write(manifest_path, "field_data_manifest.csv")
    z.write(val_path, "val.csv")
    z.write(test_path, "test.csv")
    z.write(ext_path, "external_provenance.csv")
    z.write(rw_path, "realworld_provenance.csv")
    print("  Added CSV manifests.")
    
    # 2. Add train images from manifest
    train_df = pd.read_csv(manifest_path)
    added_paths = set()
    for _, row in train_df.iterrows():
        p = row["image_path"]
        rel = row["relative_path"].replace("\\", "/")
        if p not in added_paths and os.path.exists(p):
            z.write(p, rel)
            added_paths.add(p)
    print(f"  Added {len(added_paths)} training images.")
    
    # 3. Add val & test images
    for df, name in [(pd.read_csv(val_path), "validation"), (pd.read_csv(test_path), "test")]:
        cnt = 0
        for _, row in df.iterrows():
            p = row["file_path"]
            rel = os.path.relpath(p, PROJECT_ROOT).replace("\\", "/")
            if p not in added_paths and os.path.exists(p):
                z.write(p, rel)
                added_paths.add(p)
                cnt += 1
        print(f"  Added {cnt} {name} images.")
        
    # 4. Add Track B & Track C Chilli eval images
    ext_df = pd.read_csv(ext_path)
    ch_ext = ext_df[ext_df["crop"].str.lower() == "chilli"]
    b_cnt = 0
    for _, row in ch_ext.iterrows():
        p = row["image_path"]
        rel = row["relative_path"].replace("\\", "/")
        if p not in added_paths and os.path.exists(p):
            z.write(p, rel)
            added_paths.add(p)
            b_cnt += 1
    print(f"  Added {b_cnt} Track B evaluation images.")
    
    rw_df = pd.read_csv(rw_path)
    ch_rw = rw_df[rw_df["crop"].str.lower() == "chilli"]
    c_cnt = 0
    for _, row in ch_rw.iterrows():
        p = row["image_path"]
        rel = row["relative_path"].replace("\\", "/")
        if p not in added_paths and os.path.exists(p):
            z.write(p, rel)
            added_paths.add(p)
            c_cnt += 1
    print(f"  Added {c_cnt} Track C evaluation images.")

size_mb = os.path.getsize(OUT_ZIP) / (1024 * 1024)
print(f"\nDataset packaged successfully: {OUT_ZIP} ({size_mb:.1f} MB)")
