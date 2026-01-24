import os
import argparse
import shutil
from pathlib import Path
import pandas as pd

def copy_baseline_into_augmented():
    # Delete all files but keep folder
    folder = Path("data/processed/augmented/train/malignant")
    for file in folder.glob("*"):
        if file.is_file():
            file.unlink()

    # Copy baseline with 'real_' prefix
    source = Path("data/processed/baseline/train/malignant")
    dest = Path("data/processed/augmented/train/malignant")
    dest.mkdir(parents=True, exist_ok=True)

    for img in source.glob("*.jpg"):
        shutil.copy(img, dest / f"{img.stem}.jpg")
    
    print(f"✓ Copied {len(list(source.glob('*.jpg')))} real baseline images")

def copy_synthetic_into_augmented(gan_version):
    # Copy synthetic images (already have 'synthetic_malignant_' prefix from generate_samples)
    source = Path(f"data/synthetic/{gan_version}/generation")
    dest = Path("data/processed/augmented/train/malignant")
    dest.mkdir(parents=True, exist_ok=True)

    for img in source.glob("*.jpg"):
        shutil.copy(img, dest / img.name)
    
    print(f"✓ Copied {len(list(source.glob('*.jpg')))} synthetic images")


def copy_val_test_csvs():
    """Copy val.csv and test.csv from baseline to augmented (they remain the same)"""
    for split in ['val', 'test']:
        source_csv = Path(f"data/processed/baseline/{split}/{split}.csv")
        dest_dir = Path(f"data/processed/augmented/{split}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_csv = dest_dir / f"{split}.csv"
        
        shutil.copy(source_csv, dest_csv)
        print(f"✓ Copied {split}.csv from baseline to augmented")


def create_augmented_train_csv(gan_version):
    """Create train.csv for augmented dataset with baseline + synthetic images"""
    # Read baseline train.csv
    baseline_csv = Path("data/processed/baseline/train/train.csv")
    df_baseline = pd.read_csv(baseline_csv)
    
    # Get list of synthetic images
    synthetic_dir = Path(f"data/synthetic/{gan_version}/generation")
    synthetic_images = list(synthetic_dir.glob("synthetic_malignant_*.jpg"))
    
    # Create entries for synthetic images (without extension, label = Malignant)
    synthetic_entries = []
    for img_path in synthetic_images:
        img_name = img_path.stem  # Remove .png extension
        synthetic_entries.append({'img_name': img_name, 'target': 'Malignant'})
    
    df_synthetic = pd.DataFrame(synthetic_entries)
    
    # Combine baseline and synthetic
    df_augmented = pd.concat([df_baseline, df_synthetic], ignore_index=True)
    
    # Save to augmented train.csv
    output_csv = Path("data/processed/augmented/train/train.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_augmented.to_csv(output_csv, index=False)
    
    print(f"✓ Created train.csv with {len(df_baseline)} baseline + {len(df_synthetic)} synthetic = {len(df_augmented)} total entries")

    
#TO RUN **AFTER** generate_samples.py script
def main():
    parser = argparse.ArgumentParser(
        description='COPYING IMAGES FROM SYNTHETIC AND BASELINE TO POPULATE AUGMENTED'
    )

    parser.add_argument('--version', 
        type=str, 
        required=True,
        help='Specify the gan version output to analyze')
    
    args = parser.parse_args()
    
    print(f"Populating augmented dataset with GAN version: {args.version}\n")
    
    copy_baseline_into_augmented()
    copy_synthetic_into_augmented(args.version)
    
    # Copy CSV files
    copy_val_test_csvs()
    create_augmented_train_csv(args.version)
    
    print("\n✓ Augmented dataset populated successfully!")
    
    # Show final count
    augmented_path = Path("data/processed/augmented/train/malignant")
    total_count = len(list(augmented_path.glob("*.jpg")))
    print(f"Total images in augmented/train/malignant: {total_count}")

if __name__ == '__main__':
    main()