"""Baseline Classifier Domain Shift Evaluation

This script evaluates GAN-generated image quality by measuring domain shift:
1. Trains a standard classifier (no domain adaptation) on synthetic data
2. Evaluates performance on both source (synthetic) and target (real) domains
3. Quantifies domain gap to assess GAN output quality

Purpose: Measure the problem (domain shift) as a baseline for comparison
with domain adaptation methods (e.g., DANN).

Small domain gap = High quality GAN outputs
Large domain gap = GAN needs improvement
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import yaml
import argparse
from pathlib import Path
import sys
import logging
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.classifier.classifier import Classifier
from evaluation.domain_shift_evaluation import DomainShiftEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_classifier_with_domain_shift_eval(
    config_path,
    source_dir='data/processed/domain_adaptation/source_synthetic/train',
    target_dir='data/processed/domain_adaptation/target_real/test',
    output_dir='results/domain_shift/baseline'
):
    """Train baseline classifier and evaluate domain shift for GAN quality assessment.
    
    Args:
        config_path: Path to training configuration YAML
        source_dir: Source domain (real benign + synthetic malignant)
        target_dir: Target domain (real benign + real malignant)
        output_dir: Directory for results and metrics
    
    Returns:
        model: Trained classifier
        source_metrics: Performance on source domain
        target_metrics: Performance on target domain  
        domain_gap: Domain shift quantification metrics
    """
    logger.info("="*70)
    logger.info("BASELINE CLASSIFIER - DOMAIN SHIFT EVALUATION")
    logger.info("Purpose: Evaluate GAN quality by measuring domain shift")
    logger.info("="*70)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training transform with augmentation to encourage learning real features
    # (not just GAN artifacts or positioning)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Test transform without augmentation (standard evaluation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    logger.info("\n" + "="*70)
    logger.info("DATA LOADING")
    logger.info("="*70)
    logger.info(f"Source domain (TRAINING): {source_dir}")
    logger.info("  - Real benign + Synthetic malignant (GAN-generated)")
    logger.info("  - Using augmentation to learn robust features (not GAN artifacts)")
    source_dataset = datasets.ImageFolder(source_dir, transform=train_transform)
    source_loader = DataLoader(
        source_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    logger.info(f"  - Total: {len(source_dataset)} samples")
    logger.info(f"  - Classes: {source_dataset.classes}")
    
    logger.info(f"\nTarget domain (TESTING): {target_dir}")
    logger.info("  - Real benign + Real malignant (held-out real data)")
    logger.info("  - No augmentation (standard evaluation)")
    target_dataset = datasets.ImageFolder(target_dir, transform=test_transform)
    target_loader = DataLoader(
        target_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    logger.info(f"  - Total: {len(target_dataset)} samples")
    logger.info(f"  - Classes: {target_dataset.classes}")
    
    model = Classifier(
        num_classes=config['model']['num_classes'],
        model_name=config['model']['architecture'],
        pretrained=config['model']['pretrained']
    ).to(device)
    
    # Freeze backbone and only train classification head
    # This prevents overfitting to GAN artifacts in deep features
    logger.info("\n" + "="*70)
    logger.info("MODEL CONFIGURATION")
    logger.info("="*70)
    logger.info("Freezing backbone layers (only training final classifier)")
    logger.info("  Reason: Use ImageNet features, learn only benign/malignant boundary")
    logger.info("  Benefit: Prevents overfitting to GAN-specific artifacts")
    model.freeze_layers_except_last()
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters that require gradients (final layer)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-5)
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training'].get('scheduler_step', 10),
        gamma=config['training'].get('scheduler_gamma', 0.1)
    )
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING ON SOURCE DOMAIN (Synthetic Data)")
    logger.info("="*70)
    num_epochs = config['training']['num_epochs']
    logger.info(f"Training for {num_epochs} epochs...")
    
    best_train_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in source_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        avg_train_loss = train_loss / len(source_loader)
        train_acc = train_correct / train_total
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] - "
                       f"Loss: {avg_train_loss:.4f}, "
                       f"Acc: {train_acc:.4f}")
        
        # Save best model based on training loss
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save(model.state_dict(), output_dir / 'best_model.pth')
    
    logger.info(f"\nTraining completed. Best loss: {best_train_loss:.4f}")
    logger.info(f"Model saved to: {output_dir / 'best_model.pth'}")
    
    # Load best model
    model.load_state_dict(torch.load(output_dir / 'best_model.pth'))
    
    logger.info("\n" + "="*70)
    logger.info("DOMAIN SHIFT EVALUATION")
    logger.info("="*70)
    logger.info("Evaluating classifier performance on both domains...")
    logger.info("This quantifies GAN output quality.\n")
    
    evaluator = DomainShiftEvaluator(model, device, output_dir=str(output_dir))
    source_metrics, target_metrics, domain_gap = evaluator.evaluate_domain_shift(
        source_loader, target_loader, criterion
    )
    
    # Create comprehensive summary
    logger.info("\n" + "="*70)
    logger.info("EVALUATION SUMMARY - GAN QUALITY ASSESSMENT")
    logger.info("="*70)
    logger.info(f"Source Domain Performance (trained on synthetic):")
    logger.info(f"  Accuracy: {source_metrics.get('accuracy', 0):.4f}")
    logger.info(f"  F1-Score: {source_metrics.get('f1', 0):.4f}")
    logger.info(f"\nTarget Domain Performance (tested on real):")
    logger.info(f"  Accuracy: {target_metrics.get('accuracy', 0):.4f}")
    logger.info(f"  F1-Score: {target_metrics.get('f1', 0):.4f}")
    logger.info(f"\nDomain Gap (Performance Drop):")
    logger.info(f"  Accuracy Gap: {domain_gap.get('accuracy_drop', 0):.4f}")
    logger.info(f"  F1-Score Gap: {domain_gap.get('f1_drop', 0):.4f}")
    
    # Interpretation
    acc_gap = abs(domain_gap.get('accuracy_drop', 0))
    logger.info("\n" + "="*70)
    logger.info("INTERPRETATION")
    logger.info("="*70)
    if acc_gap < 0.05:
        logger.info("Excellent: Very small domain gap (<5%)")
        logger.info("  GAN generates highly realistic images")
    elif acc_gap < 0.10:
        logger.info("Good: Small domain gap (5-10%)")
        logger.info("  GAN generates good quality images")
    elif acc_gap < 0.20:
        logger.info("Moderate: Noticeable domain gap (10-20%)")
        logger.info("  GAN quality could be improved")
    else:
        logger.info("Large: Significant domain gap (>20%)")
        logger.info("  GAN needs substantial improvement")
    logger.info("="*70)
    
    # Save summary to JSON (convert numpy arrays to lists)
    def convert_to_serializable(obj):
        """Convert numpy arrays to lists for JSON serialization"""
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    summary = {
        'source_metrics': convert_to_serializable(source_metrics),
        'target_metrics': convert_to_serializable(target_metrics),
        'domain_gap': convert_to_serializable(domain_gap),
        'interpretation': {
            'accuracy_gap_percentage': float(acc_gap * 100),
            'quality_assessment': 'excellent' if acc_gap < 0.05 else 'good' if acc_gap < 0.10 else 'moderate' if acc_gap < 0.20 else 'poor'
        }
    }
    
    with open(output_dir / 'baseline_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to: {output_dir / 'baseline_summary.json'}")
    
    return model, source_metrics, target_metrics, domain_gap


def main():
    parser = argparse.ArgumentParser(
        description='Baseline Domain Shift Evaluation - Assess GAN Quality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example usage:
  python baseline_classifier_domain_eval.py \\
    --config experiments/domain_shift_eval.yaml \\
    --source-dir data/processed/domain_adaptation/source_synthetic/train \\
    --target-dir data/processed/domain_adaptation/target_real/test \\
    --output-dir results/domain_shift/baseline

This script evaluates GAN-generated image quality by measuring domain shift.
Smaller performance gap = Better GAN quality.
        '''
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training configuration YAML file'
    )
    parser.add_argument(
        '--source-dir',
        type=str,
        default='data/processed/domain_adaptation/source_synthetic/train',
        help='Source domain: real benign + synthetic malignant (GAN output)'
    )
    parser.add_argument(
        '--target-dir',
        type=str,
        default='data/processed/domain_adaptation/target_real/test',
        help='Target domain: real benign + real malignant (ground truth)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/domain_shift/baseline',
        help='Output directory for model, metrics, and visualizations'
    )
    
    args = parser.parse_args()
    
    train_classifier_with_domain_shift_eval(
        args.config,
        args.source_dir,
        args.target_dir,
        args.output_dir
    )


if __name__ == '__main__':
    main()
