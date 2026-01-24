import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DomainShiftEvaluator:
    
    def __init__(self, model, device, output_dir='results/domain_shift'):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate_domain_shift(self, source_loader, target_loader, criterion):
        """
        Evaluate domain shift impact on classifier.
        
        Args:
            source_loader: DataLoader for source domain (synthetic malignant)
            target_loader: DataLoader for target domain (real malignant)
            criterion: Loss function
            
        Returns:
            source_metrics: Metrics on source domain
            target_metrics: Metrics on target domain
            domain_gap: Quantification of domain shift effect
        """
        
        logger.info("Evaluating model on source domain (synthetic)...")
        source_metrics = self._evaluate_domain(source_loader, criterion, domain_name='source_synthetic')
        
        logger.info("Evaluating model on target domain (real)...")
        target_metrics = self._evaluate_domain(target_loader, criterion, domain_name='target_real')
        
        domain_gap = self._calculate_domain_gap(source_metrics, target_metrics)
        
        logger.info("Creating visualizations...")
        self._visualize_results(source_metrics, target_metrics, domain_gap)
        
        logger.info("Saving detailed analysis...")
        self._save_analysis(source_metrics, target_metrics, domain_gap)
        
        return source_metrics, target_metrics, domain_gap
    
    def _evaluate_domain(self, loader, criterion, domain_name):
        self.model.eval()
        
        all_preds = []
        all_probs = []
        all_targets = []
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        
        cm = confusion_matrix(all_targets, all_preds)
        
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        try:
            roc_auc = roc_auc_score(all_targets, all_probs[:, 1])
        except:
            roc_auc = 0
        
        try:
            precision_vals, recall_vals, _ = precision_recall_curve(all_targets, all_probs[:, 1])
            pr_auc = auc(recall_vals, precision_vals)
        except:
            pr_auc = 0
        
        metrics = {
            'domain': domain_name,
            'loss': total_loss / num_batches,
            'accuracy': accuracy_score(all_targets, all_preds),
            'precision': precision_score(all_targets, all_preds, zero_division=0),
            'recall': recall_score(all_targets, all_preds, zero_division=0),
            'f1': f1_score(all_targets, all_preds, zero_division=0),
            'specificity': specificity,
            'sensitivity': sensitivity,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm,
            'preds': all_preds,
            'targets': all_targets,
            'probs': all_probs
        }
        
        return metrics
    
    def _calculate_domain_gap(self, source_metrics, target_metrics):
        gap = {
            'accuracy_drop': source_metrics['accuracy'] - target_metrics['accuracy'],
            'f1_drop': source_metrics['f1'] - target_metrics['f1'],
            'recall_drop': source_metrics['recall'] - target_metrics['recall'],
            'precision_drop': source_metrics['precision'] - target_metrics['precision'],
            'sensitivity_drop': source_metrics['sensitivity'] - target_metrics['sensitivity'],
            'specificity_drop': source_metrics['specificity'] - target_metrics['specificity'],
            'roc_auc_drop': source_metrics['roc_auc'] - target_metrics['roc_auc'],
        }
        return gap
    
    def _visualize_results(self, source_metrics, target_metrics, domain_gap):
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Domain Shift Analysis: Source (Synthetic) vs Target (Real)', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'sensitivity']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            source_val = source_metrics[metric]
            target_val = target_metrics[metric]
            
            x = np.arange(2)
            vals = [source_val, target_val]
            colors = ['#3498db', '#e74c3c']
            
            bars = ax.bar(x, vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Score', fontweight='bold')
            ax.set_title(f'{metric.capitalize()}', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(['Source\n(Synthetic)', 'Target\n(Real)'])
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            for i, (bar, val) in enumerate(zip(bars, vals)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'domain_shift_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        gap_keys = list(domain_gap.keys())
        gap_vals = list(domain_gap.values())
        colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in gap_vals]
        
        bars = ax.barh(gap_keys, gap_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xlabel('Performance Drop (Source - Target)', fontweight='bold')
        ax.set_title('Domain Shift Impact: Performance Degradation', fontweight='bold', fontsize=14)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax.grid(axis='x', alpha=0.3)
        
        for bar, val in zip(bars, gap_vals):
            ax.text(val + 0.01 if val > 0 else val - 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}', ha='left' if val > 0 else 'right', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'domain_gap_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Confusion Matrices: Domain Comparison', fontsize=14, fontweight='bold')
        
        sns.heatmap(source_metrics['confusion_matrix'], annot=True, fmt='d', 
                   ax=axes[0], cmap='Blues', cbar=True, square=True)
        axes[0].set_title('Source Domain\n(Real Benign + Synthetic Malignant)', fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        
        sns.heatmap(target_metrics['confusion_matrix'], annot=True, fmt='d',
                   ax=axes[1], cmap='Oranges', cbar=True, square=True)
        axes[1].set_title('Target Domain\n(Real Benign + Real Malignant)', fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {self.output_dir}")
    
    def _save_analysis(self, source_metrics, target_metrics, domain_gap):
        
        summary = {
            'source_domain': {
                'loss': float(source_metrics['loss']),
                'accuracy': float(source_metrics['accuracy']),
                'precision': float(source_metrics['precision']),
                'recall': float(source_metrics['recall']),
                'f1': float(source_metrics['f1']),
                'specificity': float(source_metrics['specificity']),
                'sensitivity': float(source_metrics['sensitivity']),
                'roc_auc': float(source_metrics['roc_auc']),
                'pr_auc': float(source_metrics['pr_auc']),
            },
            'target_domain': {
                'loss': float(target_metrics['loss']),
                'accuracy': float(target_metrics['accuracy']),
                'precision': float(target_metrics['precision']),
                'recall': float(target_metrics['recall']),
                'f1': float(target_metrics['f1']),
                'specificity': float(target_metrics['specificity']),
                'sensitivity': float(target_metrics['sensitivity']),
                'roc_auc': float(target_metrics['roc_auc']),
                'pr_auc': float(target_metrics['pr_auc']),
            },
            'domain_gap': {k: float(v) for k, v in domain_gap.items()}
        }
        
        with open(self.output_dir / 'domain_shift_analysis.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        results_df = pd.DataFrame({
            'Metric': ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'sensitivity', 'roc_auc', 'pr_auc'],
            'Source (Synthetic)': [
                source_metrics['accuracy'],
                source_metrics['precision'],
                source_metrics['recall'],
                source_metrics['f1'],
                source_metrics['specificity'],
                source_metrics['sensitivity'],
                source_metrics['roc_auc'],
                source_metrics['pr_auc']
            ],
            'Target (Real)': [
                target_metrics['accuracy'],
                target_metrics['precision'],
                target_metrics['recall'],
                target_metrics['f1'],
                target_metrics['specificity'],
                target_metrics['sensitivity'],
                target_metrics['roc_auc'],
                target_metrics['pr_auc']
            ]
        })
        
        results_df['Domain Gap'] = results_df['Source (Synthetic)'] - results_df['Target (Real)']
        results_df.to_csv(self.output_dir / 'domain_shift_metrics.csv', index=False)
        
        logger.info(f"Analysis saved to {self.output_dir}")
        logger.info("\n" + "="*70)
        logger.info("DOMAIN SHIFT EVALUATION SUMMARY")
        logger.info("="*70)
        logger.info("\nSource Domain (Real Benign + Synthetic Malignant):")
        logger.info(f"  Accuracy: {source_metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score: {source_metrics['f1']:.4f}")
        logger.info(f"  Recall:   {source_metrics['recall']:.4f}")
        logger.info(f"  ROC-AUC:  {source_metrics['roc_auc']:.4f}")
        logger.info("\nTarget Domain (Real Benign + Real Malignant):")
        logger.info(f"  Accuracy: {target_metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score: {target_metrics['f1']:.4f}")
        logger.info(f"  Recall:   {target_metrics['recall']:.4f}")
        logger.info(f"  ROC-AUC:  {target_metrics['roc_auc']:.4f}")
        logger.info("\nDomain Shift Impact:")
        logger.info(f"  Accuracy Drop:  {domain_gap['accuracy_drop']:.4f}")
        logger.info(f"  F1-Score Drop:  {domain_gap['f1_drop']:.4f}")
        logger.info(f"  Recall Drop:    {domain_gap['recall_drop']:.4f}")
        logger.info(f"  ROC-AUC Drop:   {domain_gap['roc_auc_drop']:.4f}")
        logger.info("="*70 + "\n")


def create_dataloaders(source_dir, target_dir, batch_size=32, num_workers=4):
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
    ])
    
    source_dataset = datasets.ImageFolder(source_dir, transform=transform)
    source_loader = DataLoader(source_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers)
    
    target_dataset = datasets.ImageFolder(target_dir, transform=transform)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, 
                              shuffle=False, num_workers=num_workers)
    
    logger.info(f"Source domain: {len(source_dataset)} samples")
    logger.info(f"Target domain: {len(target_dataset)} samples")
    
    return source_loader, target_loader
