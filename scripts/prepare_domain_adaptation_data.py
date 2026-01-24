import shutil
from pathlib import Path
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def prepare_domain_adaptation_datasets(
    synthetic_malignant_dir,
    real_benign_dir,
    real_malignant_dir,
    output_dir='data/processed/domain_adaptation'
):
    
    logger.info("Starting domain adaptation dataset preparation...")
    
    source_train_dir = Path(output_dir) / 'source_synthetic' / 'train'
    source_train_dir.mkdir(parents=True, exist_ok=True)
    (source_train_dir / 'benign').mkdir(exist_ok=True)
    (source_train_dir / 'malignant').mkdir(exist_ok=True)
    
    target_test_dir = Path(output_dir) / 'target_real' / 'test'
    target_test_dir.mkdir(parents=True, exist_ok=True)
    (target_test_dir / 'benign').mkdir(exist_ok=True)
    (target_test_dir / 'malignant').mkdir(exist_ok=True)
    
    logger.info("Copying real benign images to source domain...")
    benign_files = list(Path(real_benign_dir).glob('*.jpg')) + list(Path(real_benign_dir).glob('*.png'))
    for img_file in benign_files:
        dst = source_train_dir / 'benign' / img_file.name
        if not dst.exists():
            shutil.copy2(img_file, dst)
    logger.info(f"Copied {len(benign_files)} benign images to source domain")
    
    logger.info("Copying synthetic malignant images to source domain...")
    synthetic_files = list(Path(synthetic_malignant_dir).glob('*.jpg')) + list(Path(synthetic_malignant_dir).glob('*.png'))
    for img_file in synthetic_files:
        dst = source_train_dir / 'malignant' / img_file.name
        if not dst.exists():
            shutil.copy2(img_file, dst)
    logger.info(f"Copied {len(synthetic_files)} synthetic malignant images to source domain")
    
    logger.info("Copying real benign images to target domain...")
    for img_file in benign_files:
        dst = target_test_dir / 'benign' / img_file.name
        if not dst.exists():
            shutil.copy2(img_file, dst)
    logger.info(f"Copied {len(benign_files)} benign images to target domain")
    
    logger.info("Copying real malignant images to target domain...")
    malignant_files = list(Path(real_malignant_dir).glob('*.jpg')) + list(Path(real_malignant_dir).glob('*.png'))
    for img_file in malignant_files:
        dst = target_test_dir / 'malignant' / img_file.name
        if not dst.exists():
            shutil.copy2(img_file, dst)
    logger.info(f"Copied {len(malignant_files)} real malignant images to target domain")
    
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
        '--real-benign',
        type=str,
        required=True,
        help='Path to directory containing real benign training images'
    )
    parser.add_argument(
        '--real-malignant',
        type=str,
        required=True,
        help='Path to directory containing real malignant images for testing'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/domain_adaptation',
        help='Output directory for organized datasets'
    )
    
    args = parser.parse_args()
    
    prepare_domain_adaptation_datasets(
        args.synthetic_malignant,
        args.real_benign,
        args.real_malignant,
        args.output_dir
    )
