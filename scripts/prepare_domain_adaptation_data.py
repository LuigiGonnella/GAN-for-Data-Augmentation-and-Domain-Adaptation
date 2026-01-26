import shutil
from pathlib import Path
import os
import argparse
import logging
import pandas as pd
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_domain_adaptation_datasets(
    baseline_dir,
    synthetic_malignant_dir,
    output_dir='data/processed/domain_adaptation',
    balance_source_train=False,
    balance_source_val=False
):
    """Prepare domain adaptation datasets using existing baseline splits.
    
    Structure:
        - Source Train: Real benign (from baseline/train CSV) + 80% synthetic malignant
        - Source Val: Real benign (from baseline/val CSV) + 20% synthetic malignant
        - Target: All test data (from baseline/test CSV) for adaptation
    
    Args:
        baseline_dir: Path to data/processed/baseline with train/, val/, test/ subdirs and CSVs
        synthetic_malignant_dir: Directory with GAN-generated malignant images
        output_dir: Output directory for organized datasets
        balance_source_train: If True, subsample train benign to match synthetic malignant count
        balance_source_val: If True, subsample val benign to match synthetic malignant count
    """
    
    logger.info("Starting domain adaptation dataset preparation...")
    logger.info("Using existing baseline train/val/test splits")
    
    baseline_path = Path(baseline_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    source_train_dir = output_path / 'source_synthetic' / 'train'
    source_train_dir.mkdir(parents=True, exist_ok=True)
    (source_train_dir / 'benign').mkdir(exist_ok=True)
    (source_train_dir / 'malignant').mkdir(exist_ok=True)
    
    source_val_dir = output_path / 'source_synthetic' / 'val'
    source_val_dir.mkdir(parents=True, exist_ok=True)
    (source_val_dir / 'benign').mkdir(exist_ok=True)
    (source_val_dir / 'malignant').mkdir(exist_ok=True)
    
    target_test_dir = output_path / 'target_real' / 'test'
    target_test_dir.mkdir(parents=True, exist_ok=True)
    (target_test_dir / 'benign').mkdir(exist_ok=True)
    (target_test_dir / 'malignant').mkdir(exist_ok=True)
    
    # =======================
    # 1. Split Synthetic Malignant 80/20
    # =======================
    logger.info("Splitting synthetic malignant images (80% train, 20% val)...")
    synthetic_files = list(Path(synthetic_malignant_dir).glob('*.jpg')) + \
                     list(Path(synthetic_malignant_dir).glob('*.png'))
    
    if len(synthetic_files) == 0:
        raise ValueError(f"No images found in {synthetic_malignant_dir}")
    
    random.seed(42)
    random.shuffle(synthetic_files)
    num_train_synthetic = int(0.8 * len(synthetic_files))
    synthetic_train_files = synthetic_files[:num_train_synthetic]
    synthetic_val_files = synthetic_files[num_train_synthetic:]
    
    logger.info(f"Synthetic malignant split: {len(synthetic_train_files)} train, {len(synthetic_val_files)} val")
    
    # =======================
    # 2. Process Source TRAIN (Real Benign + Synthetic Malignant)
    # =======================
    logger.info("\nProcessing SOURCE TRAIN...")
    train_csv_path = baseline_path / 'train' / 'train.csv'
    train_df = pd.read_csv(train_csv_path)
    train_benign_df = train_df[train_df['target'] == 'Benign']
    
    logger.info(f"Found {len(train_benign_df)} benign images in baseline train split")
    
    # Balance if requested
    benign_train_to_copy = train_benign_df
    if balance_source_train and len(train_benign_df) > num_train_synthetic:
        logger.info(f"Balancing source train: subsampling {len(train_benign_df)} benign to {num_train_synthetic}")
        benign_train_to_copy = train_benign_df.sample(n=num_train_synthetic, random_state=42)
    
    # Copy benign images (check both flat structure and benign/ subfolder)
    benign_train_count = 0
    for img_name in benign_train_to_copy['img_name']:
        # Try benign subfolder first (if exists)
        src_file = baseline_path / 'train' / 'benign' / f"{img_name}.jpg"
        if not src_file.exists():
            # Try flat structure
            src_file = baseline_path / 'train' / f"{img_name}.jpg"
        
        if src_file.exists():
            dst = source_train_dir / 'benign' / src_file.name
            if not dst.exists():
                shutil.copy2(src_file, dst)
                benign_train_count += 1
        else:
            logger.warning(f"Training image not found: {img_name}")
    
    logger.info(f"Copied {benign_train_count} benign images to source train")
    
    # Copy synthetic malignant for train
    for img_file in synthetic_train_files:
        dst = source_train_dir / 'malignant' / img_file.name
        if not dst.exists():
            shutil.copy2(img_file, dst)
    
    logger.info(f"Copied {len(synthetic_train_files)} synthetic malignant images to source train")
    
    # =======================
    # 3. Process Source VAL (Real Benign + Synthetic Malignant)
    # =======================
    logger.info("\nProcessing SOURCE VAL...")
    val_csv_path = baseline_path / 'val' / 'val.csv'
    val_df = pd.read_csv(val_csv_path)
    val_benign_df = val_df[val_df['target'] == 'Benign']
    
    logger.info(f"Found {len(val_benign_df)} benign images in baseline val split")
    
    # Balance if requested
    benign_val_to_copy = val_benign_df
    if balance_source_val and len(val_benign_df) > num_train_synthetic:
        logger.info(f"Balancing source val: subsampling {len(val_benign_df)} benign to {len(synthetic_val_files)}")
        benign_val_to_copy = val_benign_df.sample(n=len(synthetic_val_files), random_state=42)
    
    # Copy benign images (check both flat structure and benign/ subfolder)
    benign_val_count = 0
    for img_name in benign_val_to_copy['img_name']:
        # Try flat structure first (val typically has no subfolders)
        src_file = baseline_path / 'val' / f"{img_name}.jpg"
        if not src_file.exists():
            # Try benign subfolder as fallback
            src_file = baseline_path / 'val' / 'benign' / f"{img_name}.jpg"
        
        if src_file.exists():
            dst = source_val_dir / 'benign' / src_file.name
            if not dst.exists():
                shutil.copy2(src_file, dst)
                benign_val_count += 1
        else:
            logger.warning(f"Validation image not found: {img_name}")
    
    logger.info(f"Copied {benign_val_count} benign images to source val")
    
    # Copy synthetic malignant for val
    for img_file in synthetic_val_files:
        dst = source_val_dir / 'malignant' / img_file.name
        if not dst.exists():
            shutil.copy2(img_file, dst)
    
    logger.info(f"Copied {len(synthetic_val_files)} synthetic malignant images to source val")
    
    # =======================
    # 4. Process Target TEST (All Real Test Data)
    # =======================
    logger.info("\nProcessing TARGET TEST (for adaptation)...")
    test_csv_path = baseline_path / 'test' / 'test.csv'
    test_df = pd.read_csv(test_csv_path)
    test_benign_df = test_df[test_df['target'] == 'Benign']
    test_malignant_df = test_df[test_df['target'] == 'Malignant']
    
    logger.info(f"Found {len(test_benign_df)} benign, {len(test_malignant_df)} malignant in baseline test split")
    
    # Copy benign test images (check both flat structure and benign/ subfolder)
    benign_test_count = 0
    for img_name in test_benign_df['img_name']:
        # Try flat structure first (test typically has no subfolders)
        src_file = baseline_path / 'test' / f"{img_name}.jpg"
        if not src_file.exists():
            # Try benign subfolder as fallback
            src_file = baseline_path / 'test' / 'benign' / f"{img_name}.jpg"
        
        if src_file.exists():
            dst = target_test_dir / 'benign' / src_file.name
            if not dst.exists():
                shutil.copy2(src_file, dst)
                benign_test_count += 1
        else:
            logger.warning(f"Test benign image not found: {img_name}")
    
    logger.info(f"Copied {benign_test_count} benign images to target test")
    
    # Copy malignant test images (check both flat structure and malignant/ subfolder)
    malignant_test_count = 0
    for img_name in test_malignant_df['img_name']:
        # Try flat structure first (test typically has no subfolders)
        src_file = baseline_path / 'test' / f"{img_name}.jpg"
        if not src_file.exists():
            # Try malignant subfolder as fallback
            src_file = baseline_path / 'test' / 'malignant' / f"{img_name}.jpg"
        
        if src_file.exists():
            dst = target_test_dir / 'malignant' / src_file.name
            if not dst.exists():
                shutil.copy2(src_file, dst)
                malignant_test_count += 1
        else:
            logger.warning(f"Test malignant image not found: {img_name}")
    
    logger.info(f"Copied {malignant_test_count} malignant images to target test")
    
    # =======================
    # Final Summary
    # =======================
    logger.info("\n" + "="*70)
    logger.info("Domain Adaptation Dataset Preparation Complete!")
    logger.info("="*70)
    logger.info("SOURCE TRAIN (real benign + synthetic malignant):")
    logger.info(f"  - Benign: {len(list((source_train_dir / 'benign').glob('*')))} images")
    logger.info(f"  - Malignant (synthetic): {len(list((source_train_dir / 'malignant').glob('*')))} images")
    logger.info("\nSOURCE VAL (real benign + synthetic malignant):")
    logger.info(f"  - Benign: {len(list((source_val_dir / 'benign').glob('*')))} images")
    logger.info(f"  - Malignant (synthetic): {len(list((source_val_dir / 'malignant').glob('*')))} images")
    logger.info("\nTARGET TEST (all real test data for adaptation):")
    logger.info(f"  - Benign: {len(list((target_test_dir / 'benign').glob('*')))} images")
    logger.info(f"  - Malignant: {len(list((target_test_dir / 'malignant').glob('*')))} images")
    logger.info("="*70)
    
    return {
        'source_train_dir': str(source_train_dir),
        'source_val_dir': str(source_val_dir),
        'target_test_dir': str(target_test_dir)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare domain adaptation datasets using existing baseline splits'
    )
    parser.add_argument(
        '--baseline-dir',
        type=str,
        required=True,
        help='Path to data/processed/baseline directory (contains train/, val/, test/ with CSVs)'
    )
    parser.add_argument(
        '--synthetic-malignant',
        type=str,
        required=True,
        help='Path to directory containing GAN-generated malignant images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/domain_adaptation',
        help='Output directory for organized datasets (default: data/processed/domain_adaptation)'
    )
    parser.add_argument(
        '--balance-source-train',
        action='store_true',
        default=False,
        help='Balance source train by subsampling benign to match synthetic malignant count'
    )
    parser.add_argument(
        '--balance-source-val',
        action='store_true',
        default=False,
        help='Balance source val by subsampling benign to match synthetic malignant count'
    )
    
    args = parser.parse_args()
    
    prepare_domain_adaptation_datasets(
        baseline_dir=args.baseline_dir,
        synthetic_malignant_dir=args.synthetic_malignant,
        output_dir=args.output_dir,
        balance_source_train=args.balance_source_train,
        balance_source_val=args.balance_source_val
    )
