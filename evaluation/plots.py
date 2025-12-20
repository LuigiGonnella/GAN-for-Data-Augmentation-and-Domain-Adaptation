import matplotlib.pyplot as plt
import os
from sklearn.metrics import auc,roc_curve
import numpy as np
import seaborn as sns

def plot_train_val_stats(plot_dir, train_losses_epochs, validation_losses_epochs, train_accuracy_epochs, validation_accuracy_epochs):
    loss_accuracy_path = os.path.join(plot_dir, 'loss_accuracy.png')
    _, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].plot(train_losses_epochs, c='blue', label='train_loss')
    ax[0].plot(validation_losses_epochs, c='orange', label='val_loss')
    ax[1].plot(train_accuracy_epochs, c='blue', label='train_accuracy')
    ax[1].plot(validation_accuracy_epochs, c='orange', label='val_accuracy')

    ax[0].set_xlabel('num epochs')
    ax[0].set_ylabel('loss')

    ax[1].set_xlabel('num epochs')
    ax[1].set_ylabel('accuracy')

    ax[0].legend()
    ax[1].legend()
    
    plt.tight_layout()
    plt.savefig(loss_accuracy_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_ROC(precision, recall, optimal_idx, optimal_threshold, all_labels, all_probs, roc_auc, plot_dir):
    # Precision-Recall Curve
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    avg_precision = auc(recall, precision)
    ax1.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP={avg_precision:.3f})')
    ax1.scatter(recall[optimal_idx], precision[optimal_idx], color='red', s=100, 
                label=f'Optimal (max recall)={optimal_threshold:.3f}', zorder=5)
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Curve', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # ROC Curve
    fpr, tpr, thresholds_roc = roc_curve(all_labels, all_probs)
    ax2.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC={roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    
    # Find threshold point on ROC curve closest to optimal
    optimal_idx_roc = np.argmin(np.abs(thresholds_roc - optimal_threshold))
    ax2.scatter(fpr[optimal_idx_roc], tpr[optimal_idx_roc], color='red', s=100,
                label=f'Optimal (max recall)={optimal_threshold:.3f}', zorder=5)
    
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('ROC Curve', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    curve_path = os.path.join(plot_dir, 'pr_roc_curves.png')
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved PR and ROC curves to: {curve_path}")

    return ax2

def plot_cm(plot_dir, test_cm):
    cm_path = os.path.join(plot_dir, 'confusion_matrix.png')
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(test_cm, annot=True, cbar=False, cmap='GnBu', annot_kws={'size':16}, fmt='g', 
                xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant']) 
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)

    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved Confusion Matrix to: {cm_path}")