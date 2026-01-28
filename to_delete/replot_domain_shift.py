"""Re-evaluate Domain Shift with Corrected Threshold Selection

This script re-evaluates saved DANN models using corrected threshold selection logic.
It regenerates all plots and metrics with proper F1-based threshold optimization.

Usage:
    python training/replot_domain_shift.py \
        --model-path results/domain_shift/ALEXNET/best_dann_model.pth \
        --source-dir data/processed/domain_adaptation/source_synthetic/train \
        --target-dir data/processed/domain_adaptation/target_real/test \
        --output-dir results/domain_shift/ALEXNET/corrected_threshold
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
)
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.domain_adaptation.dann import DomainAdversarialNN


def corrected_threshold_selection(y_true, y_probs):
    """Find optimal threshold with corrected F1 calculation."""
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Compute F1 scores with proper zero-division handling
    f1_scores = np.zeros(len(precision_vals))
    valid_idx = (precision_vals + recall_vals) > 0
    f1_scores[valid_idx] = (2 * precision_vals[valid_idx] * recall_vals[valid_idx] / 
                            (precision_vals[valid_idx] + recall_vals[valid_idx]))
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    # Clamp to valid range
    optimal_threshold = np.clip(optimal_threshold, 0.0, 1.0)
    
    return optimal_threshold, precision_vals, recall_vals, thresholds


def evaluate_domain(model, loader, device, domain_name):
    """Evaluate model on a domain with corrected threshold selection."""
    model.eval()
    
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability of malignant
            
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Find optimal threshold with corrected logic
    optimal_threshold, precision_vals, recall_vals, pr_thresholds = corrected_threshold_selection(
        all_targets, all_probs
    )
    
    # Generate predictions with optimal threshold
    optimal_preds = (all_probs >= optimal_threshold).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(all_targets, optimal_preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except:
        roc_auc = 0
    
    # Calculate PR-AUC
    pr_auc = np.trapz(precision_vals[::-1], recall_vals[::-1]) if len(precision_vals) > 1 else 0
    
    metrics = {
        'domain': domain_name,
        'optimal_threshold': float(optimal_threshold),
        'accuracy': float(accuracy_score(all_targets, optimal_preds)),
        'precision': float(precision_score(all_targets, optimal_preds, zero_division=0)),
        'recall': float(recall_score(all_targets, optimal_preds, zero_division=0)),
        'f1': float(f1_score(all_targets, optimal_preds, zero_division=0)),
        'specificity': float(specificity),
        'sensitivity': float(sensitivity),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': cm.tolist(),
        'probs': all_probs,
        'targets': all_targets,
        'precision_vals': precision_vals,
        'recall_vals': recall_vals,
        'pr_thresholds': pr_thresholds
    }
    
    return metrics


def plot_evaluation(source_metrics, target_metrics, output_dir):
    """Create comprehensive evaluation plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 10)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. PR Curves
    ax = axes[0, 0]
    ax.plot(source_metrics['recall_vals'], source_metrics['precision_vals'], 
            'b-', linewidth=2, label=f'Source (AUC={source_metrics["pr_auc"]:.3f})')
    ax.plot(target_metrics['recall_vals'], target_metrics['precision_vals'], 
            'r-', linewidth=2, label=f'Target (AUC={target_metrics["pr_auc"]:.3f})')
    ax.scatter([source_metrics['recall']], [source_metrics['precision']], 
               s=100, c='blue', marker='*', label=f'Source Optimal (thr={source_metrics["optimal_threshold"]:.3f})', zorder=5)
    ax.scatter([target_metrics['recall']], [target_metrics['precision']], 
               s=100, c='red', marker='*', label=f'Target Optimal (thr={target_metrics["optimal_threshold"]:.3f})', zorder=5)
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves (Corrected Threshold)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. ROC Curves
    ax = axes[0, 1]
    # Compute ROC curves
    source_fpr, source_tpr, _ = roc_curve(source_metrics['targets'], source_metrics['probs'])
    target_fpr, target_tpr, _ = roc_curve(target_metrics['targets'], target_metrics['probs'])
    
    ax.plot(source_fpr, source_tpr, 'b-', linewidth=2, 
            label=f'Source (AUC={source_metrics["roc_auc"]:.3f})')
    ax.plot(target_fpr, target_tpr, 'r-', linewidth=2, 
            label=f'Target (AUC={target_metrics["roc_auc"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix - Source
    ax = axes[0, 2]
    cm_source = np.array(source_metrics['confusion_matrix'])
    sns.heatmap(cm_source, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    ax.set_title(f'Source Domain Confusion Matrix\n(threshold={source_metrics["optimal_threshold"]:.3f})', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    
    # 4. Confusion Matrix - Target
    ax = axes[1, 0]
    cm_target = np.array(target_metrics['confusion_matrix'])
    sns.heatmap(cm_target, annot=True, fmt='d', cmap='Reds', ax=ax,
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    ax.set_title(f'Target Domain Confusion Matrix\n(threshold={target_metrics["optimal_threshold"]:.3f})', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    
    # 5. Metrics Comparison Bar Chart
    ax = axes[1, 1]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    source_values = [source_metrics['accuracy'], source_metrics['precision'], 
                     source_metrics['recall'], source_metrics['f1'], source_metrics['specificity']]
    target_values = [target_metrics['accuracy'], target_metrics['precision'], 
                     target_metrics['recall'], target_metrics['f1'], target_metrics['specificity']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, source_values, width, label='Source', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, target_values, width, label='Target', color='indianred', alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Domain Comparison (Corrected Thresholds)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 6. Domain Gap Visualization
    ax = axes[1, 2]
    domain_gaps = {
        'Accuracy': source_metrics['accuracy'] - target_metrics['accuracy'],
        'Precision': source_metrics['precision'] - target_metrics['precision'],
        'Recall': source_metrics['recall'] - target_metrics['recall'],
        'F1-Score': source_metrics['f1'] - target_metrics['f1'],
        'Specificity': source_metrics['specificity'] - target_metrics['specificity']
    }
    
    colors = ['green' if v < 0 else 'red' for v in domain_gaps.values()]
    bars = ax.barh(list(domain_gaps.keys()), list(domain_gaps.values()), color=colors, alpha=0.7)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Source - Target (Gap)', fontsize=12)
    ax.set_title('Domain Shift Gap\n(Negative = Target Better)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (metric, value) in enumerate(domain_gaps.items()):
        ax.text(value, i, f' {value:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'domain_shift_corrected_threshold.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_dir / 'domain_shift_corrected_threshold.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Re-evaluate Domain Shift with Corrected Threshold Selection'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to saved DANN model checkpoint (.pth file)'
    )
    parser.add_argument(
        '--source-dir',
        type=str,
        default='data/processed/domain_adaptation/source_synthetic/train',
        help='Source domain directory (real benign + synthetic malignant)'
    )
    parser.add_argument(
        '--target-dir',
        type=str,
        default='data/processed/domain_adaptation/target_real/test',
        help='Target domain directory (real benign + real malignant)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for corrected plots (default: same as model dir + /corrected_threshold)'
    )
    parser.add_argument(
        '--feature-dim',
        type=int,
        default=512,
        help='Feature dimension of DANN model (default: 512)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation (default: 32)'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir is None:
        model_dir = Path(args.model_path).parent
        args.output_dir = model_dir / 'corrected_threshold'
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("RE-EVALUATION WITH CORRECTED THRESHOLD SELECTION")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Source: {args.source_dir}")
    print(f"Target: {args.target_dir}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading DANN model...")
    model = DomainAdversarialNN(feature_dim=args.feature_dim, num_classes=2).to(device)
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model.eval()
    print("✓ Model loaded\n")
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load data
    print("Loading datasets...")
    source_dataset = datasets.ImageFolder(args.source_dir, transform=transform)
    target_dataset = datasets.ImageFolder(args.target_dir, transform=transform)
    
    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=4)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, 
                               shuffle=False, num_workers=4)
    
    print(f"  Source: {len(source_dataset)} samples")
    print(f"  Target: {len(target_dataset)} samples\n")
    
    # Evaluate both domains
    print("Evaluating source domain with corrected threshold...")
    source_metrics = evaluate_domain(model, source_loader, device, 'source_synthetic')
    print(f"  Optimal threshold: {source_metrics['optimal_threshold']:.4f}")
    print(f"  Accuracy: {source_metrics['accuracy']:.4f}")
    print(f"  Recall: {source_metrics['recall']:.4f}")
    print(f"  F1-Score: {source_metrics['f1']:.4f}\n")
    
    print("Evaluating target domain with corrected threshold...")
    target_metrics = evaluate_domain(model, target_loader, device, 'target_real')
    print(f"  Optimal threshold: {target_metrics['optimal_threshold']:.4f}")
    print(f"  Accuracy: {target_metrics['accuracy']:.4f}")
    print(f"  Recall: {target_metrics['recall']:.4f}")
    print(f"  F1-Score: {target_metrics['f1']:.4f}\n")
    
    # Calculate domain gap
    domain_gap = {
        'accuracy_drop': source_metrics['accuracy'] - target_metrics['accuracy'],
        'f1_drop': source_metrics['f1'] - target_metrics['f1'],
        'recall_drop': source_metrics['recall'] - target_metrics['recall'],
        'precision_drop': source_metrics['precision'] - target_metrics['precision']
    }
    
    # Create plots
    print("Creating visualizations...")
    plot_evaluation(source_metrics, target_metrics, output_dir)
    
    # Save metrics
    summary = {
        'source_metrics': {k: v for k, v in source_metrics.items() 
                          if k not in ['probs', 'targets', 'precision_vals', 'recall_vals', 'pr_thresholds']},
        'target_metrics': {k: v for k, v in target_metrics.items() 
                          if k not in ['probs', 'targets', 'precision_vals', 'recall_vals', 'pr_thresholds']},
        'domain_gap': domain_gap
    }
    
    with open(output_dir / 'corrected_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Metrics saved to: {output_dir / 'corrected_metrics.json'}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY (CORRECTED THRESHOLDS)")
    print("="*70)
    print(f"\nSource Domain:")
    print(f"  Threshold: {source_metrics['optimal_threshold']:.4f}")
    print(f"  Accuracy:  {source_metrics['accuracy']:.4f}")
    print(f"  F1-Score:  {source_metrics['f1']:.4f}")
    print(f"  Recall:    {source_metrics['recall']:.4f}")
    print(f"\nTarget Domain:")
    print(f"  Threshold: {target_metrics['optimal_threshold']:.4f}")
    print(f"  Accuracy:  {target_metrics['accuracy']:.4f}")
    print(f"  F1-Score:  {target_metrics['f1']:.4f}")
    print(f"  Recall:    {target_metrics['recall']:.4f}")
    print(f"\nDomain Gap:")
    print(f"  Accuracy Drop: {domain_gap['accuracy_drop']:.4f}")
    print(f"  F1-Score Drop: {domain_gap['f1_drop']:.4f}")
    print(f"  Recall Drop:   {domain_gap['recall_drop']:.4f}")
    print("="*70)
    print("\n✓ Re-evaluation complete!")


if __name__ == '__main__':
    main()
