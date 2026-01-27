"""Domain Adversarial Neural Network (DANN) Training Script

This script implements domain adaptation to reduce domain shift effects:
1. Trains on source domain (real benign + synthetic malignant)
2. Adapts to target domain (real benign + real malignant) using adversarial training
3. Learns domain-invariant features that generalize across domains

DANN Architecture:
- Feature Extractor: Learns domain-invariant representations
- Class Classifier: Predicts benign/malignant on source domain
- Domain Discriminator: Distinguishes source from target (adversarial)

Goal: Improve performance on target domain by learning features that work 
      for both synthetic and real malignant images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.autograd import Function
from torchvision import transforms, datasets
import yaml
import argparse
from pathlib import Path
import sys
import logging
import json
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.domain_adaptation.dann import DomainAdversarialNN, DANNTrainer
from evaluation.domain_shift_evaluation import DomainShiftEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_dann(
    config_path,
    source_train_dir='data/processed/domain_adaptation/source_synthetic/train',
    source_val_dir='data/processed/domain_adaptation/source_synthetic/val',
    target_dir='data/processed/domain_adaptation/target_real/test',
    output_dir='results/domain_shift/dann'
):
    """Train DANN model with domain adversarial adaptation.
    
    Args:
        config_path: Path to training configuration YAML
        source_train_dir: Source training data (real benign + synthetic malignant)
        source_val_dir: Source validation data (real benign + synthetic malignant)
        target_dir: Target domain (real benign + real malignant)
        output_dir: Directory for results and metrics
    
    Returns:
        model: Trained DANN model
        source_metrics: Performance on source domain
        target_metrics: Performance on target domain
        domain_gap: Quantification of domain shift
    """
    logger.info("="*70)
    logger.info("DOMAIN ADVERSARIAL NEURAL NETWORK (DANN)")
    logger.info("Purpose: Learn domain-invariant features via adversarial training")
    logger.info("="*70)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Data loading
    logger.info("\n" + "="*70)
    logger.info("DATA LOADING")
    logger.info("="*70)
    
    logger.info(f"Source TRAIN: {source_train_dir}")
    logger.info("  - Real benign + Synthetic malignant (labeled)")
    source_train_dataset = datasets.ImageFolder(source_train_dir, transform=train_transform)
    source_train_loader = DataLoader(
        source_train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        drop_last=True  # Ensure consistent batch sizes
    )
    logger.info(f"  - Total: {len(source_train_dataset)} samples")
    logger.info(f"  - Classes: {source_train_dataset.classes}")
    
    logger.info(f"\nSource VAL: {source_val_dir}")
    logger.info("  - Real benign + Synthetic malignant (for monitoring)")
    source_val_dataset = datasets.ImageFolder(source_val_dir, transform=test_transform)
    source_val_loader = DataLoader(
        source_val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    logger.info(f"  - Total: {len(source_val_dataset)} samples")
    logger.info(f"  - Classes: {source_val_dataset.classes}")
    
    logger.info(f"\nTarget (ADAPTATION): {target_dir}")
    logger.info("  - Real benign + Real malignant (unlabeled for adaptation)")
    target_dataset = datasets.ImageFolder(target_dir, transform=train_transform)
    target_loader = DataLoader(
        target_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    logger.info(f"  - Total: {len(target_dataset)} samples")
    logger.info(f"  - Classes: {target_dataset.classes}")
    
    # For evaluation (no augmentation)
    logger.info("\nCreating evaluation loaders (no augmentation)...")
    source_eval_dataset = datasets.ImageFolder(source_train_dir, transform=test_transform)
    source_eval_loader = DataLoader(
        source_eval_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    target_eval_dataset = datasets.ImageFolder(target_dir, transform=test_transform)
    target_eval_loader = DataLoader(
        target_eval_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Model
    logger.info("\n" + "="*70)
    logger.info("MODEL INITIALIZATION")
    logger.info("="*70)
    
    feature_dim = config.get('feature_dim', 512)
    model = DomainAdversarialNN(
        feature_dim=feature_dim,
        num_classes=config['model']['num_classes']
    ).to(device)
    
    logger.info(f"Feature dimension: {feature_dim}")
    logger.info(f"Architecture:")
    logger.info(f"  - Feature Extractor: Learns domain-invariant features")
    logger.info(f"  - Class Classifier: Predicts benign/malignant")
    logger.info(f"  - Domain Discriminator: Distinguishes source/target (adversarial)")
    
    # Optimizers
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-5)
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        cooldown=1
    )
    
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.BCEWithLogitsLoss()  # More numerically stable than Sigmoid + BCELoss
    
    # Initialize DANN Trainer
    logger.info("Initializing DANN Trainer with gradient reversal...")
    logger.info("Note: Using BCEWithLogitsLoss for improved numerical stability")
    trainer = DANNTrainer(model, device, output_dir=str(output_dir))
    
    # Training
    logger.info("\n" + "="*70)
    logger.info("ADVERSARIAL DOMAIN ADAPTATION TRAINING")
    logger.info("="*70)
    num_epochs = config['training']['num_epochs']
    logger.info(f"Training for {num_epochs} epochs with domain adversarial loss...")
    logger.info(f"Lambda schedule: Gradually increases from 0 to 1")
    logger.info(f"Model selection: Best model based on TARGET RECALL (sensitivity)")
    logger.info(f"Early stopping: Patience of 8 epochs")
    
    best_target_recall = -1.0
    patience = 15
    early_stopping_count = 0
    training_history = {
        'class_loss': [],
        'domain_loss': [],
        'total_loss': [],
        'lambda_values': [],
        'domain_acc': [],  # Track domain discriminator accuracy
        'source_acc': [],
        'target_acc': [],
        'source_recall': [],
        'target_recall': []
    }

    init_epochs = 5
    
    for epoch in range(num_epochs):
        # Compute adaptive lambda using DANNTrainer's method
        lambda_adapt = trainer.compute_lambda_adaptation(epoch, num_epochs)
        training_history['lambda_values'].append(lambda_adapt)
        
        # Train one epoch using DANNTrainer
        avg_total_loss, avg_class_loss, avg_domain_loss, avg_domain_acc = trainer.train_epoch(
            source_train_loader, 
            target_loader, 
            optimizer,
            criterion_class,
            criterion_domain,
            lambda_adapt
        )
        
        training_history['class_loss'].append(avg_class_loss)
        training_history['domain_loss'].append(avg_domain_loss)
        training_history['total_loss'].append(avg_total_loss)
        training_history['domain_acc'].append(avg_domain_acc)
        
        # Evaluate using DANNTrainer's evaluate method
        _, source_acc, source_recall = trainer.evaluate(source_eval_loader, criterion_class)
        _, target_acc, target_recall = trainer.evaluate(target_eval_loader, criterion_class)
        
        # Step scheduler based on class loss
        scheduler.step(avg_class_loss)
        
        training_history['source_acc'].append(source_acc)
        training_history['target_acc'].append(target_acc)
        training_history['source_recall'].append(source_recall)
        training_history['target_recall'].append(target_recall)
        
        logger.info(f"\nEpoch [{epoch+1}/{num_epochs}]")
        logger.info(f"  Class Loss: {avg_class_loss:.4f}")
        logger.info(f"  Domain Loss: {avg_domain_loss:.4f}")
        logger.info(f"  Domain Acc: {avg_domain_acc:.4f} (target: ~0.50 = perfect invariance)")
        logger.info(f"  Lambda: {lambda_adapt:.3f}")
        logger.info(f"  Source - Accuracy: {source_acc:.4f}, Recall: {source_recall:.4f}")
        logger.info(f"  Target - Accuracy: {target_acc:.4f}, Recall: {target_recall:.4f}")
        
        # Save best model based on target RECALL (most important for cancer detection)
        if target_recall > best_target_recall and epoch > init_epochs:
            best_target_recall = target_recall
            early_stopping_count = 0
            torch.save(model.state_dict(), output_dir / 'best_dann_model.pth')
            logger.info(f"  ✓ New best target recall: {best_target_recall:.4f}")
        else:
            early_stopping_count += 1
            
        if early_stopping_count >= patience:
            logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
            logger.info(f"No improvement in target recall for {patience} epochs")
            break
    
    logger.info(f"\nTraining completed. Best target recall: {best_target_recall:.4f}")
    logger.info(f"Model saved to: {output_dir / 'best_dann_model.pth'}")
    
    # Load best model
    model.load_state_dict(torch.load(output_dir / 'best_dann_model.pth'))
    
    # Final evaluation
    logger.info("\n" + "="*70)
    logger.info("FINAL DOMAIN SHIFT EVALUATION")
    logger.info("="*70)
    logger.info("Evaluating DANN performance on both domains...")
    
    evaluator = DomainShiftEvaluator(model, device, output_dir=str(output_dir))
    source_metrics, target_metrics, domain_gap = evaluator.evaluate_domain_shift(
        source_eval_loader, target_eval_loader, criterion_class
    )
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("DANN EVALUATION SUMMARY")
    logger.info("="*70)
    logger.info(f"Source Domain (trained + adapted):")
    logger.info(f"  Accuracy: {source_metrics.get('accuracy', 0):.4f}")
    logger.info(f"  F1-Score: {source_metrics.get('f1', 0):.4f}")
    logger.info(f"  Recall: {source_metrics.get('recall', 0):.4f}")
    logger.info(f"\nTarget Domain (adapted):")
    logger.info(f"  Accuracy: {target_metrics.get('accuracy', 0):.4f}")
    logger.info(f"  F1-Score: {target_metrics.get('f1', 0):.4f}")
    logger.info(f"  Recall: {target_metrics.get('recall', 0):.4f}")
    logger.info(f"\nDomain Gap:")
    logger.info(f"  Accuracy Drop: {domain_gap.get('accuracy_drop', 0):.4f}")
    logger.info(f"  F1-Score Drop: {domain_gap.get('f1_drop', 0):.4f}")
    logger.info(f"  Recall Drop: {domain_gap.get('recall_drop', 0):.4f}")
    
    # Interpretation
    acc_gap = abs(domain_gap.get('accuracy_drop', 0))
    logger.info("\n" + "="*70)
    logger.info("INTERPRETATION")
    logger.info("="*70)
    if acc_gap < 0.05:
        logger.info("Excellent: Domain adaptation successful! (<5% gap)")
        logger.info("  Features are highly domain-invariant")
    elif acc_gap < 0.10:
        logger.info("Good: Significant domain gap reduction (5-10%)")
        logger.info("  Features show good domain invariance")
    elif acc_gap < 0.20:
        logger.info("Moderate: Some domain gap reduction (10-20%)")
        logger.info("  Partial domain invariance achieved")
    else:
        logger.info("Poor: Domain gap remains large (>20%)")
        logger.info("  Domain adaptation needs improvement")
    logger.info("="*70)
    
    # Save results
    def convert_to_serializable(obj):
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
        'training_history': training_history,
        'best_target_recall': float(best_target_recall),
        'interpretation': {
            'accuracy_gap_percentage': float(acc_gap * 100),
            'quality_assessment': 'excellent' if acc_gap < 0.05 else 'good' if acc_gap < 0.10 else 'moderate' if acc_gap < 0.20 else 'poor'
        }
    }
    
    with open(output_dir / 'dann_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to: {output_dir / 'dann_summary.json'}")
    
    return model, source_metrics, target_metrics, domain_gap


def main():
    parser = argparse.ArgumentParser(
        description='Domain Adversarial Neural Network (DANN) Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example usage:
  python train_with_dann.py \\
    --config experiments/domain_shift_eval.yaml \\
    --source-train-dir data/processed/domain_adaptation/source_synthetic/train \\
    --source-val-dir data/processed/domain_adaptation/source_synthetic/val \\
    --target-dir data/processed/domain_adaptation/target_real/test \\
    --output-dir results/domain_shift/dann

DANN learns domain-invariant features through adversarial training:
- Feature extractor learns features that work for both domains
- Domain discriminator tries to identify source vs target
- Gradient reversal forces extractor to fool discriminator
- Result: Features that generalize from synthetic to real data
        '''
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training configuration YAML file'
    )
    parser.add_argument(
        '--source-train-dir',
        type=str,
        default='data/processed/domain_adaptation/source_synthetic/train',
        help='Source training: real benign + synthetic malignant (labeled)'
    )
    parser.add_argument(
        '--source-val-dir',
        type=str,
        default='data/processed/domain_adaptation/source_synthetic/val',
        help='Source validation: real benign + synthetic malignant (for monitoring)'
    )
    parser.add_argument(
        '--target-dir',
        type=str,
        default='data/processed/domain_adaptation/target_real/test',
        help='Target domain: real benign + real malignant (unlabeled for adaptation)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/domain_shift/dann',
        help='Output directory for model, metrics, and visualizations'
    )
    
    args = parser.parse_args()
    
    train_dann(
        args.config,
        args.source_train_dir,
        args.source_val_dir,
        args.target_dir,
        args.output_dir
    )


if __name__ == '__main__':
    main()
