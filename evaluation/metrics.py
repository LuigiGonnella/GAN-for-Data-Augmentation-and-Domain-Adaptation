import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, auc, recall_score, precision_score
import matplotlib.pyplot as plt
import numpy as np
import os
from evaluation.plots import plot_precision_recall_ROC

def evaluate(model, dataloader, criterion, device, optimal_threshold=False):
    model.eval()
    running_loss = 0.0
    num_samples = 0
    val_corrects = 0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            bs = images.shape[0]
            num_samples += bs
            outputs = model(images)
            _, targets_pred = torch.max(outputs, 1)
            val_corrects += torch.sum(labels==targets_pred).item()


            loss = criterion(outputs, labels)
            running_loss += loss.item()* bs
            probs = torch.softmax(outputs, dim=1)[:, 1]  

            if optimal_threshold is False:
                preds = torch.argmax(outputs, dim=1)
            else:
                preds = (probs >= optimal_threshold).int()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = running_loss / num_samples
    accuracy = accuracy_score(all_labels, all_preds)
    
    
    
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)

    
    return avg_loss, accuracy, f1, recall, precision, roc_auc, cm, num_samples, val_corrects


def evaluate_with_threshold_tuning(model, dataloader, criterion, device, plot_dir):
    """
    Evaluate model on test set with threshold tuning and plotting.
    Finds optimal threshold on validation set, plots PR and ROC curves.
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
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
    
    # Find optimal threshold that maximizes F1 for medical imaging
    precision, recall, thresholds_pr = precision_recall_curve(all_labels, all_probs)
    f1 = precision * recall / (precision + recall)
    # For medical imaging, prioritize F1 to balance RECALL and PRECISION (choosing optimal th basing only on recall wold fic threshold=0.0 and the model would classify each sample as positive)
    optimal_idx = np.argmax(f1)  
    optimal_threshold = thresholds_pr[optimal_idx] if optimal_idx < len(thresholds_pr) else 0.5
    
    print(f"\n  Optimal threshold (max F1): {optimal_threshold:.3f}")
    
    # Generate predictions with optimal threshold
    optimal_preds = (all_probs >= optimal_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, optimal_preds)
    f1 = f1_score(all_labels, optimal_preds, zero_division=0)
    recall_val = recall_score(all_labels, optimal_preds, zero_division=0)
    precision_val = precision_score(all_labels, optimal_preds, zero_division=0)
    cm = confusion_matrix(all_labels, optimal_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    
    # Precision-Recall Curve and ROC Curve
    plot_precision_recall_ROC(precision, recall, optimal_idx, optimal_threshold, all_labels, all_probs, roc_auc, plot_dir)
    
    
    
    return avg_loss, accuracy, f1, recall_val, precision_val, roc_auc, cm, optimal_threshold
