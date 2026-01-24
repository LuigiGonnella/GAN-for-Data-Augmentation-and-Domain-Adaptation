import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_optimization_image(report_dir):
    """
    Generate optimization_report.png from the existing optimization_report.txt
    """
    
    # Define the metrics based on the txt file
    baseline_metrics = {
        'accuracy': 0.7953,
        'precision': 0.3363,
        'recall': 0.7600,
        'f1': 0.4663,
        'roc_auc': 0.8556,
        'val_loss': 0.2676
    }
    
    finetune_results = {
        'accuracy': 0.8651,
        'precision': 0.4560,
        'recall': 0.7600,
        'f1': 0.5700,
        'roc_auc': 0.9185,
        'val_loss': 0.2374
    }
    
    tuning_results = {
        'accuracy': 0.8949,
        'precision': 0.5376,
        'recall': 0.7633,
        'f1': 0.6309,
        'roc_auc': 0.9384,
        'val_loss': 0.1983
    }
    
    best_config = {
        'lr': 0.0001,
        'batch_size': 64,
        'weight_decay': 1e-05,
        'momentum': 0.8,
        'optimizer': 'RMSprop'
    }
    
    try:
        # Prepare metrics for each step
        steps = ['Baseline', 'Fine-tuned', 'Hyperparam Tuned']
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'val_loss']
        values = [
            [baseline_metrics['accuracy'], baseline_metrics['precision'], baseline_metrics['recall'], baseline_metrics['f1'], baseline_metrics['roc_auc'], baseline_metrics['val_loss']],
            [finetune_results['accuracy'], finetune_results['precision'], finetune_results['recall'], finetune_results['f1'], finetune_results['roc_auc'], finetune_results['val_loss']],
            [tuning_results['accuracy'], tuning_results['precision'], tuning_results['recall'], tuning_results['f1'], tuning_results['roc_auc'], tuning_results['val_loss']]
        ]

        df = pd.DataFrame([x for x in values if x is not None], columns=metrics, index=[s for s, v in zip(steps, values) if v is not None])
        
        # Set up figure with optimized spacing
        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, top=0.93, bottom=0.05, left=0.08, right=0.97)

        # Title
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        ax_title.set_title('FINAL OPTIMIZATION REPORT', fontsize=18, fontweight='bold', pad=5)

        # Bar chart for metrics
        ax1 = fig.add_subplot(gs[1, 0])
        df_plot = df.drop('val_loss', axis=1)
        df_plot.plot(kind='bar', ax=ax1)
        ax1.set_ylabel('Score', fontsize=10)
        ax1.set_title('Model Performance Metrics', fontsize=12, pad=8)
        ax1.legend(loc='upper left', fontsize=9, bbox_to_anchor=(1.02, 1), framealpha=0.9)
        ax1.set_xticklabels(df_plot.index, rotation=0, fontsize=10)
        ax1.tick_params(axis='y', labelsize=9)

        # Table for best hyperparameters
        ax2 = fig.add_subplot(gs[2, 0])
        ax2.axis('off')
        hp_table = [
            ['Learning Rate', best_config['lr']],
            ['Batch Size', best_config['batch_size']],
            ['Weight Decay', best_config['weight_decay']],
            ['Optimizer', best_config['optimizer']]
        ]
        table = ax2.table(cellText=hp_table, colLabels=['Hyperparameter', 'Value'], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax2.set_title('Best Hyperparameter Configuration', fontsize=12, pad=8)
        
        # Improvement summary - spans rows 1 and 2 in column 1
        ax3 = fig.add_subplot(gs[1:, 1])
        ax3.axis('off')
        improvement_finetune_f1 = (finetune_results['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100
        improvement_final_f1 = (tuning_results['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100

        improvement_finetune_recall = (finetune_results['recall'] - baseline_metrics['recall']) / baseline_metrics['recall'] * 100
        improvement_final_recall = (tuning_results['recall'] - baseline_metrics['recall']) / baseline_metrics['recall'] * 100

        summary_text_f1 = (
            f"F1 Score Improvement:\n"
            f"- Baseline: {baseline_metrics['f1']:.4f}\n"
            f"- Fine-tuned: {finetune_results['f1']:.4f} ({improvement_finetune_f1:+.2f}%)\n"
            f"- Hyperparam Tuned: {tuning_results['f1']:.4f} ({improvement_final_f1:+.2f}%)\n\n"
        )
        ax3.text(0, 0.95, summary_text_f1, fontsize=11, va='top', ha='left', wrap=True)

        # Add recall improvement summary in the same subplot
        summary_text_recall = (
            f"Recall Improvement:\n"
            f"- Baseline: {baseline_metrics['recall']:.4f}\n"
            f"- Fine-tuned: {finetune_results['recall']:.4f} ({improvement_finetune_recall:+.2f}%)\n"
            f"- Hyperparam Tuned: {tuning_results['recall']:.4f} ({improvement_final_recall:+.2f}%)\n"
        )
        ax3.text(0, 0.45, summary_text_recall, fontsize=11, va='top', ha='left', wrap=True)

        # Save image
        image_path = os.path.join(report_dir, "optimization_report.png")
        plt.savefig(image_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        print(f"✓ Report image saved: {image_path}")
        
    except Exception as e:
        print(f"[ERROR] Could not generate report image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Path to the report directory
    report_dir = "results/classifier_on_augmented_DCGAN/final_report"
    
    print(f"Generating optimization report image for: {report_dir}")
    generate_optimization_image(report_dir)
    print("Done!")
