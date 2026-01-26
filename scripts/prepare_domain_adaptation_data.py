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
    synthetic_malignant_dir,
    real_benign_train_dir,
    test_images_dir,
    output_dir='data/processed/domain_adaptation',
    balance_source=False,
    balance_target=False
):
    """Prepare domain adaptation datasets.
    
    Args:
        synthetic_malignant_dir: Directory with GAN-generated malignant images
        real_benign_train_dir: Directory with real benign training images
        test_images_dir: Directory containing all test images
        output_dir: Output directory for organized datasets
        balance_source: If True, subsample benign to match malignant count
        balance_target: If True, subsample target to have equal classes
    """
    
    logger.info("Starting domain adaptation dataset preparation...")
    
    source_train_dir = Path(output_dir) / 'source_synthetic' / 'train'
    source_train_dir.mkdir(parents=True, exist_ok=True)
    (source_train_dir / 'benign').mkdir(exist_ok=True)
    (source_train_dir / 'malignant').mkdir(exist_ok=True)
    
    target_test_dir = Path(output_dir) / 'target_real' / 'test'
    target_test_dir.mkdir(parents=True, exist_ok=True)
    (target_test_dir / 'benign').mkdir(exist_ok=True)
    (target_test_dir / 'malignant').mkdir(exist_ok=True)
    
    # Get synthetic malignant count first for balancing
    logger.info("Counting synthetic malignant images...")
    synthetic_files = list(Path(synthetic_malignant_dir).glob('*.jpg')) + list(Path(synthetic_malignant_dir).glob('*.png'))
    num_synthetic = len(synthetic_files)
    logger.info(f"Found {num_synthetic} synthetic malignant images")
    
    # Handle source domain benign images with optional balancing
    logger.info("Processing real benign TRAIN images for source domain...")
    benign_train_files = list(Path(real_benign_train_dir).glob('*.jpg')) + list(Path(real_benign_train_dir).glob('*.png'))
    
    if balance_source and len(benign_train_files) > num_synthetic:
        logger.info(f"Balancing source domain: subsampling {len(benign_train_files)} benign to {num_synthetic}")
        random.seed(42)  # For reproducibility
        benign_train_files = random.sample(benign_train_files, num_synthetic)
    
    for img_file in benign_train_files:
        dst = source_train_dir / 'benign' / img_file.name
        if not dst.exists():
            shutil.copy2(img_file, dst)
    logger.info(f"Copied {len(benign_train_files)} benign train images to source domain")
    
    logger.info("Copying synthetic malignant images to source domain...")
    for img_file in synthetic_files:
        dst = source_train_dir / 'malignant' / img_file.name
        if not dst.exists():
            shutil.copy2(img_file, dst)
    logger.info(f"Copied {len(synthetic_files)} synthetic malignant images to source domain")
    
    test_csv_path = test_images_dir + '/test.csv'
    # Read test CSV to organize target domain
    logger.info(f"Reading test labels from {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)
    logger.info(f"Found {len(test_df)} test samples")
    
    # Separate benign and malignant from test set
    benign_test = test_df[test_df['target'] == 'Benign']
    malignant_test = test_df[test_df['target'] == 'Malignant']
    
    logger.info(f"Test set: {len(benign_test)} benign, {len(malignant_test)} malignant")
    
    # Balance target domain if requested
    if balance_target:
        min_count = min(len(benign_test), len(malignant_test))
        logger.info(f"Balancing target domain: subsampling both classes to {min_count}")
        benign_test = benign_test.sample(n=min_count, random_state=42)
        malignant_test = malignant_test.sample(n=min_count, random_state=42)
        logger.info(f"Balanced target: {len(benign_test)} benign, {len(malignant_test)} malignant")
    
    # Copy benign test images to target domain
    logger.info("Copying real benign TEST images to target domain...")
    benign_count = 0
    for img_name in benign_test['img_name']:
        # Try both .jpg and .png extensions
        img_file = None
        for ext in ['.jpg', '.png']:
            candidate = Path(test_images_dir) / f"{img_name}{ext}"
            if candidate.exists():
                img_file = candidate
                break
        
        if img_file:
            dst = target_test_dir / 'benign' / img_file.name
            if not dst.exists():
                shutil.copy2(img_file, dst)
                benign_count += 1
        else:
            logger.warning(f"Image not found: {img_name}")
    
    logger.info(f"Copied {benign_count} benign test images to target domain")
    
    # Copy malignant test images to target domain
    logger.info("Copying real malignant TEST images to target domain...")
    malignant_count = 0
    for img_name in malignant_test['img_name']:
        # Try both .jpg and .png extensions
        img_file = None
        for ext in ['.jpg', '.png']:
            candidate = Path(test_images_dir) / f"{img_name}{ext}"
            if candidate.exists():
                img_file = candidate
                break
        
        if img_file:
            dst = target_test_dir / 'malignant' / img_file.name
            if not dst.exists():
                shutil.copy2(img_file, dst)
                malignant_count += 1
        else:
            logger.warning(f"Image not found: {img_name}")
    
    logger.info(f"Copied {malignant_count} real malignant images to target domain")
    
    logger.info("\n" + "="*60)
    logger.info("Domain Adaptation Dataset Preparation Complete!")
    logger.info("="*60)
    logger.info("Source Domain (Training):")
    logger.info(f"  - Real Benign: {len(list((source_train_dir / 'benign').glob('*')))} images")
    logger.info(f"  - Synthetic Malignant: {len(list((source_train_dir / 'malignant').glob('*')))} images")
    logger.info("Target Domain (Testing):")
    logger.info(f"  - Real Benign: {len(list((target_test_dir / 'benign').glob('*')))} images")
    logger.info(f"  - Real Malignant: {len(list((target_test_dir / 'malignant').glob('*')))} images")
    logger.info("="*60)
    
    return {
        'source_dir': str(source_train_dir),
        'target_dir': str(target_test_dir)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare domain adaptation datasets for domain shift evaluation'
    )
    parser.add_argument(
        '--synthetic-malignant',
        type=str,
        required=True,
        help='Path to directory containing GAN-generated malignant images'
    )
    parser.add_argument(
        '--real-benign-train',
        type=str,
        required=True,
        help='Path to directory containing real benign training images'
    )
    parser.add_argument(
        '--test-images-dir',
        type=str,
        required=True,
        help='Path to directory containing all test images'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/domain_adaptation',
        help='Output directory for organized datasets'
    )
    parser.add_argument(
        '--balance-source',
        action='store_true',
        default=False,
        help='Balance source domain by subsampling benign to match malignant count'
    )
    parser.add_argument(
        '--no-balance-source',
        action='store_false',
        dest='balance_source',
        help='Do not balance source domain'
    )
    parser.add_argument(
        '--balance-target',
        action='store_true',
        default=False,
        help='Balance target domain by subsampling to smallest class (reduces test data)'
    )
    
    args = parser.parse_args()
    
    prepare_domain_adaptation_datasets(
        args.synthetic_malignant,
        args.real_benign_train,
        args.test_images_dir,
        args.output_dir,
        args.balance_source,
        args.balance_target
    )
