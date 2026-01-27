"""
Replot classification metrics with corrected threshold selection.
This script recomputes and replotted PR, ROC curves and confusion matrices
with the fixed threshold selection logic.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import yaml
import os
import pandas as pd
from PIL import Image
import argparse
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.classifier.classifier import Classifier
from evaluation.classifier_metrics import evaluate, evaluate_with_threshold_tuning
from evaluation.plots import plot_cm, plot_precision_recall_ROC
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, auc, recall_score, precision_score
)


class DatasetCSV(Dataset):
    """Dataset that loads images and labels from CSV metadata"""
    def __init__(self, img_dir, csv_path, transform=None, has_subdirs=False):
        self.img_dir = img_dir
        self.transform = transform
        self.metadata = pd.read_csv(csv_path)
        self.label_map = {'Benign': 0, 'benign': 0, 'Malignant': 1, 'malignant': 1}
        self.has_subdirs = has_subdirs
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name = row['img_name'] + '.jpg'
        label = self.label_map[row['target']]
        
        if self.has_subdirs:
            subdir = 'benign' if label == 0 else 'malignant'
            img_path = os.path.join(self.img_dir, subdir, img_name)
        else:
            img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def replot_evaluation(
    model_path,
    data_path,
    output_dir='results/classifier_on_augmented_ALEXNET/tmp_plots',
    batch_size=32,
    model_name='alexnet'
):
    """
    Replot classification metrics with corrected threshold selection.
    
    Args:
        model_path: Path to saved model checkpoint
        data_path: Path to dataset
        output_dir: Output directory for plots
        batch_size: Batch size for evaluation
        model_name: Model architecture name
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created output directory: {output_dir}")
    
    # Load model
    print("\nLoading model...")
    model = Classifier(num_classes=2, model_name=model_name, pretrained=False)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()
    print(f"✓ Model loaded from {model_path}")
    
    # Transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load validation dataset
    print("\nLoading validation data...")
    val_csv = os.path.join(data_path, 'val', 'val.csv')
    val_dataset = DatasetCSV(os.path.join(data_path, 'val'), val_csv, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"✓ Validation set: {len(val_dataset)} samples")
    
    # Load test dataset
    print("Loading test data...")
    test_csv = os.path.join(data_path, 'test', 'test.csv')
    test_dataset = DatasetCSV(os.path.join(data_path, 'test'), test_csv, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"✓ Test set: {len(test_dataset)} samples")
    
    criterion = nn.CrossEntropyLoss()
    
    # ========== VALIDATION SET EVALUATION ==========
    print("\n" + "="*70)
    print("VALIDATION SET EVALUATION")
    print("="*70)
    
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.shape[0]
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    avg_loss = running_loss / len(all_labels)
    
    # Find optimal threshold with corrected logic
    print("\nComputing optimal threshold...")
    precision, recall, thresholds_pr = precision_recall_curve(all_labels, all_probs)
    
    # Avoid division by zero
    f1_scores = np.zeros(len(precision))
    valid_idx = (precision + recall) > 0
    f1_scores[valid_idx] = 2 * precision[valid_idx] * recall[valid_idx] / (precision[valid_idx] + recall[valid_idx])
    
    optimal_idx = np.argmax(f1_scores)
    if optimal_idx < len(thresholds_pr):
        optimal_threshold = thresholds_pr[optimal_idx]
    else:
        optimal_threshold = 0.5
    
    # Ensure threshold is in valid range
    optimal_threshold = np.clip(optimal_threshold, 0.0, 1.0)
    
    print(f"✓ Optimal threshold (max F1): {optimal_threshold:.4f}")
    print(f"  Max F1 score: {np.max(f1_scores):.4f}")
    
    # Generate predictions with optimal threshold
    optimal_preds = (all_probs >= optimal_threshold).astype(int)
    
    # Calculate metrics
    val_accuracy = accuracy_score(all_labels, optimal_preds)
    val_f1 = f1_score(all_labels, optimal_preds, zero_division=0)
    val_recall = recall_score(all_labels, optimal_preds, zero_division=0)
    val_precision = precision_score(all_labels, optimal_preds, zero_division=0)
    val_cm = confusion_matrix(all_labels, optimal_preds)
    val_roc_auc = roc_auc_score(all_labels, all_probs)
    
    print(f"\nValidation Metrics:")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {val_accuracy:.4f}")
    print(f"  Recall: {val_recall:.4f}")
    print(f"  Precision: {val_precision:.4f}")
    print(f"  F1-Score: {val_f1:.4f}")
    print(f"  ROC-AUC: {val_roc_auc:.4f}")
    print(f"\nConfusion Matrix:\n{val_cm}")
    
    # Plot PR and ROC curves
    print("\nPlotting PR and ROC curves...")
    plot_precision_recall_ROC(
        precision, recall, optimal_idx, optimal_threshold,
        all_labels, all_probs, val_roc_auc, str(output_dir)
    )
    print(f"✓ Saved to {output_dir}/pr_roc_curves.png")
    
    # ========== TEST SET EVALUATION ==========
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    
    running_loss = 0.0
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.shape[0]
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_labels = np.array(test_labels)
    test_probs = np.array(test_probs)
    test_loss = running_loss / len(test_labels)
    
    # Apply optimal threshold from validation set
    test_preds = (test_probs >= optimal_threshold).astype(int)
    
    # Calculate metrics
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    test_recall = recall_score(test_labels, test_preds, zero_division=0)
    test_precision = precision_score(test_labels, test_preds, zero_division=0)
    test_cm = confusion_matrix(test_labels, test_preds)
    test_roc_auc = roc_auc_score(test_labels, test_probs)
    
    print(f"\nTest Metrics:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  F1-Score: {test_f1:.4f}")
    print(f"  ROC-AUC: {test_roc_auc:.4f}")
    print(f"\nConfusion Matrix:\n{test_cm}")
    
    # Plot test confusion matrix
    print("\nPlotting test confusion matrix...")
    plot_cm(str(output_dir), test_cm)
    print(f"✓ Saved to {output_dir}/confusion_matrix.png")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"\nValidation Performance:")
    print(f"  Accuracy: {val_accuracy:.4f}, Recall: {val_recall:.4f}, Precision: {val_precision:.4f}, F1: {val_f1:.4f}")
    print(f"\nTest Performance:")
    print(f"  Accuracy: {test_accuracy:.4f}, Recall: {test_recall:.4f}, Precision: {test_precision:.4f}, F1: {test_f1:.4f}")
    print(f"\nAll plots saved to: {output_dir}")
    print("="*70)
    
    return {
        'optimal_threshold': optimal_threshold,
        'val_metrics': {
            'loss': avg_loss,
            'accuracy': val_accuracy,
            'recall': val_recall,
            'precision': val_precision,
            'f1': val_f1,
            'roc_auc': val_roc_auc,
            'confusion_matrix': val_cm
        },
        'test_metrics': {
            'loss': test_loss,
            'accuracy': test_accuracy,
            'recall': test_recall,
            'precision': test_precision,
            'f1': test_f1,
            'roc_auc': test_roc_auc,
            'confusion_matrix': test_cm
        }
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Replot classification metrics with corrected threshold selection'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='results/classifier_on_augmented_ALEXNET/no_ht/classifier.pth',
        help='Path to saved model checkpoint'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/augmented',
        help='Path to dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/classifier_on_augmented_ALEXNET/tmp_plots',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='alexnet',
        choices=['alexnet', 'resnet50'],
        help='Model architecture'
    )
    
    args = parser.parse_args()
    
    results = replot_evaluation(
        args.model_path,
        args.data_path,
        args.output_dir,
        args.batch_size,
        args.model_name
    )
