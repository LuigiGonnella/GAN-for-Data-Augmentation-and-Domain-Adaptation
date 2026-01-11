import os
import argparse
import shutil
from pathlib import Path

def copy_baseline_into_augmented():
    # Delete all files but keep folder
    folder = Path("data/processed/augmented/train/malignant")
    for file in folder.glob("*"):
        if file.is_file():
            file.unlink()

    # Simple copy
    source = Path("data/processed/baseline/train/malignant")
    dest = Path("data/processed/augmented/train/malignant")
    dest.mkdir(parents=True, exist_ok=True)

    for img in source.glob("*.png"):
        shutil.copy(img, dest / img.name)

    # Copy with prefix
    for img in source.glob("*.png"):
        shutil.copy(img, dest / f"real_{img.name}")

def copy_synthetic_into_augmented(gan_version):
            
    # Simple copy
    source = Path(f"data/synthetic/{gan_version}/generation")
    dest = Path("data/processed/augmented/train/malignant")
    dest.mkdir(parents=True, exist_ok=True)

    for img in source.glob("*.png"):
        shutil.copy(img, dest / img.name)

    # Copy with prefix
    for img in source.glob("*.png"):
        shutil.copy(img, dest / f"{img.name}")

    
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
    
    copy_baseline_into_augmented()

    copy_synthetic_into_augmented(args.version)

    