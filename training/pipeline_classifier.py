import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_classifier import main as train_main
from finetune_classifier import run_fine_tuning
from hyperparameter_tuning import run_hyperparameter_tuning
from training.baseline_training_classifier import run_baseline
import yaml
import json
from datetime import datetime
import os

def run_full_pipeline(mode):
    
    # STEP 0: Freezing
    print("# STEP 0: Baseline Training (freeze)")

    baseline_metrics = run_baseline()
    
    print(f"\n✓ Baseline completed.\nResults:{baseline_metrics}")

    # STEP 1: Fine-Tuning 
    print("# STEP 1: Fine-Tuning")
    
    finetune_results = run_fine_tuning()
    
    print(f"\n✓ Fine-tuning completed.\nResults:{finetune_results}")
    
    if mode=='ht':
        # STEP 2: Fine-Tuning and Hyperparameter Tuning 
        print("# STEP 2: Hyperparameter Tuning with Best Strategy")
        
        best_config, tuning_results = run_hyperparameter_tuning()
        
        print(f"✓ Hyperparameter tuning completed.\nResults:{tuning_results}\n")
        print(f'BEST CONFIGURATION:/n{best_config}')
    else:
        best_config, tuning_results = False, False
    
    # STEP 3: Final Report
    print("# STEP 3: Final Report")
    
    generate_final_report(baseline_metrics, finetune_results, tuning_results, best_config)
    
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"End time: {datetime.now()}")



FINAL_METRICS = "Final Metrics:\n"
def generate_final_report(baseline_metrics, finetune_results, tuning_results, best_config):
    # --- Generate improved image report with graphs and schemas ---
    report_dir = "results/baseline/final_report"
    os.makedirs(report_dir, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        # Prepare metrics for each step
        steps = ['Baseline', 'Fine-tuned', 'Hyperparam Tuned']
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'val_loss']
        values = [
            [baseline_metrics['accuracy'], baseline_metrics['precision'], baseline_metrics['recall'], baseline_metrics['f1'], baseline_metrics['roc_auc'], baseline_metrics['val_loss']],
            [finetune_results['accuracy'], finetune_results['precision'], finetune_results['recall'], finetune_results['f1'], finetune_results['roc_auc'], finetune_results['val_loss']],
            [tuning_results['accuracy'], tuning_results['precision'], tuning_results['recall'], tuning_results['f1'], tuning_results['roc_auc'], tuning_results['val_loss']] if (tuning_results and best_config) is not False else None
        ]

        df = pd.DataFrame([x for x in values if x is not None], columns=metrics, index=[s for s, v in zip(steps, values) if v is not None])
        
        # Set up figure
        fig = plt.figure(constrained_layout=True, figsize=(14, 10))
        gs = fig.add_gridspec(3, 2)

        # Title
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')
        ax_title.set_title('FINAL OPTIMIZATION REPORT', fontsize=22, fontweight='bold', pad=20)

        # Bar chart for metrics
        ax1 = fig.add_subplot(gs[1, 0])
        df_plot = df.drop('val_loss', axis=1)
        df_plot.plot(kind='bar', ax=ax1)
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Metrics')
        ax1.legend(loc='lower right')
        ax1.set_xticklabels(df_plot.index, rotation=0)

        # Table for best hyperparameters or message if not performed
        ax2 = fig.add_subplot(gs[2, 0])
        ax2.axis('off')
        if best_config is not False:
            hp_table = [
                ['Learning Rate', best_config['learning_rate']],
                ['Batch Size', best_config['batch_size']],
                ['Weight Decay', best_config['weight_decay']],
                ['Optimizer', best_config['optimizer']]
            ]
            table = ax2.table(cellText=hp_table, colLabels=['Hyperparameter', 'Value'], loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.2)
        else:
            ax2.text(0.5, 0.5, 'No Hyperparameter Tuning performed', fontsize=14, ha='center', va='center')
            ax2.set_title('Best Hyperparameter Configuration', fontsize=14, pad=10)
        
        # Improvement summary
        ax3 = fig.add_subplot(gs[2, 1])
        ax3.axis('off')
        improvement_finetune_f1 = (finetune_results['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100
        improvement_final_f1 = (tuning_results['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100 if tuning_results else None

        improvement_finetune_recall = (finetune_results['recall'] - baseline_metrics['recall']) / baseline_metrics['recall'] * 100
        improvement_final_recall = (tuning_results['recall'] - baseline_metrics['recall']) / baseline_metrics['recall'] * 100 if tuning_results else None

        summary_text_f1 = (
            f"F1 Score Improvement:\n"
            f"- Baseline: {baseline_metrics['f1']:.4f}\n"
            f"- Fine-tuned: {finetune_results['f1']:.4f} ({improvement_finetune_f1:+.2f}%)\n"
        )
        if tuning_results:
            summary_text_f1 += f"- Hyperparam Tuned: {tuning_results['f1']:.4f} ({improvement_final_f1:+.2f}%)\n\n"
        else:
            summary_text_f1 += "- Hyperparam Tuned: No Hyperparameter Tuning performed\n\n"
        ax3.text(0, 1, summary_text_f1, fontsize=12, va='top', ha='left', wrap=True)

        # Add recall improvement summary in ax4, same style as ax3
        ax4 = fig.add_subplot(gs[3, :]) if hasattr(gs, '__getitem__') and gs.get_geometry()[0] > 3 else fig.add_subplot(111)
        ax4.axis('off')
        summary_text_recall = (
            f"Recall Improvement:\n"
            f"- Baseline: {baseline_metrics['recall']:.4f}\n"
            f"- Fine-tuned: {finetune_results['recall']:.4f} ({improvement_finetune_recall:+.2f}%)\n"
        )
        if tuning_results:
            summary_text_recall += f"- Hyperparam Tuned: {tuning_results['recall']:.4f} ({improvement_final_recall:+.2f}%)\n\n"
        else:
            summary_text_recall += "- Hyperparam Tuned: No Hyperparameter Tuning performed\n\n"
        ax4.text(0, 1, summary_text_recall, fontsize=12, va='top', ha='left', wrap=True)

        # Save image
        image_path = os.path.join(report_dir, "optimization_report.png")
        plt.savefig(image_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        print(f"✓ Improved report image saved: {image_path}")
    except Exception as e:
        print(f"[WARN] Could not generate improved report image: {e}")
    
    
    
    report_path = os.path.join(report_dir, "optimization_report.txt")
    


    # Write text report
    with open(report_path, 'w') as f:
        f.write("FINAL OPTIMIZATION REPORT\n")
        f.write("STEP 0: BASELINE TRAINING (freeze)\n")
        f.write(f"  - Accuracy: {baseline_metrics['accuracy']:.4f}\n")
        f.write(f"  - Precision: {baseline_metrics['precision']:.4f}\n")
        f.write(f"  - Recall: {baseline_metrics['recall']:.4f}\n")
        f.write(f"  - F1-Score: {baseline_metrics['f1']:.4f}\n")
        f.write(f"  - ROC-AUC: {baseline_metrics['roc_auc']:.4f}\n")
        f.write(f"  - Val Loss: {baseline_metrics['val_loss']:.4f}\n\n")
        f.write("STEP 1: FINE-TUNING\n")
        f.write(FINAL_METRICS)
        f.write(f"  - Accuracy: {finetune_results['accuracy']:.4f}\n")
        f.write(f"  - Precision: {finetune_results['precision']:.4f}\n")
        f.write(f"  - Recall: {finetune_results['recall']:.4f}\n")
        f.write(f"  - F1-Score: {finetune_results['f1']:.4f}\n")
        f.write(f"  - ROC-AUC: {finetune_results['roc_auc']:.4f}\n")
        f.write(f"  - Val Loss: {finetune_results['val_loss']:.4f}\n\n")

        if best_config is not False:
            f.write("STEP 2: HYPERPARAMETER TUNING\n")
            f.write(f"Best hyperparameter configuration:\n{best_config}\n")
            f.write(FINAL_METRICS)
            f.write(f"  - Accuracy: {tuning_results['accuracy'].item():.4f}\n")
            f.write(f"  - Precision: {tuning_results['precision'].item():.4f}\n")
            f.write(f"  - Recall: {tuning_results['recall'].item():.4f}\n")
            f.write(f"  - F1-Score: {tuning_results['f1']:.4f}\n")
            f.write(f"  - ROC-AUC: {tuning_results['roc_auc']:.4f}\n")
            f.write(f"  - Validation Loss: {tuning_results['val_loss']:.4f}\n\n")
            f.write("COMPARISON: Baseline vs Fine-tuned vs Fine Tuned and Hyperparameter Tuned\n")
            improvement_final_f1 = (tuning_results['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100
            improvement_final_recall = (tuning_results['recall'] - baseline_metrics['recall']) / baseline_metrics['recall'] * 100

        improvement_finetune_f1 = (finetune_results['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100
        improvement_finetune_recall = (finetune_results['recall'] - baseline_metrics['recall']) / baseline_metrics['recall'] * 100
        f.write(f"Baseline F1: {baseline_metrics['f1']:.4f}\n")
        f.write(f"Fine Tune F1: {finetune_results['f1']:.4f} ({improvement_finetune_f1:+.2f}%)\n")
        if best_config is not False:
            f.write(f"  After Hyperparameter Tuning F1: {tuning_results['f1']:.4f} ({improvement_final_f1:+.2f}%)\n")
        f.write(f"Baseline Recall: {baseline_metrics['recall']:.4f}\n")
        f.write(f"Fine Tune Recall: {finetune_results['recall']:.4f} ({improvement_finetune_recall:+.2f}%)\n")
        if best_config is not False:
            f.write(f"  After Hyperparameter Tuning Recall: {tuning_results['recall']:.4f} ({improvement_final_recall:+.2f}%)\n")

        if best_config is not False:
            f.write("Best hyperparameter configuration:\n")
            f.write(f"  - Learning Rate: {best_config['learning_rate']}\n")
            f.write(f"  - Batch Size: {best_config['batch_size']}\n")
            f.write(f"  - Weight Decay: {best_config['weight_decay']}\n")
            f.write(f"  - Momentum: {best_config['momentum']}\n")
            f.write(f"  - Optimizer: {best_config['optimizer']}\n")
            f.write(FINAL_METRICS)
            f.write(f"  - Accuracy: {tuning_results['accuracy']:.4f}\n")
            f.write(f"  - Precision: {tuning_results['precision']:.4f}\n")
            f.write(f"  - Recall: {tuning_results['recall']:.4f}\n")
            f.write(f"  - F1-Score: {tuning_results['f1']:.4f}\n")
            f.write(f"  - ROC-AUC: {tuning_results['roc_auc']:.4f}\n")
            f.write(f"  - Validation Loss: {tuning_results['val_loss']:.4f}\n\n")

    print(f"✓ Report saved: {report_path}")

    if best_config is not False:
        # Save summary as JSON
        summary_path = os.path.join(report_dir, "best_config.json")
        summary = {
            'best_hyperparameters': {
                'learning_rate': float(best_config['learning_rate']),
                'batch_size': int(best_config['batch_size']),
                'weight_decay': float(best_config['weight_decay']),
                'momentum': float(best_config['momentum']),
                'optimizer': best_config['optimizer'],
            },
            'final_metrics': {
                'accuracy': float(best_config['accuracy']),
                'precision': float(best_config['precision']),
                'recall': float(best_config['recall']),
                'f1_score': float(best_config['f1']),
                'roc_auc': float(best_config['roc_auc']),
                'val_loss': float(best_config['val_loss'])
            },
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Config saved: {summary_path}")

def validate_input():
        if len(sys.argv) == 2 and sys.argv[1].lower() == 'help':
            print("""
    Usage: python pipeline.py [mode]\n
    You can run this file in two ways:
    1. With no arguments: default behavior (ht)
    2. With one argument 'no_ht': disables hyperparameter tuning
    Example:
        python pipeline.py           # runs with hyperparameter tuning (ht)
        python pipeline.py no_ht     # runs without hyperparameter tuning
    """)
            sys.exit(0)
        if len(sys.argv) == 1:
            return 'ht'  # default mode
        if len(sys.argv) == 2:
            mode = sys.argv[1]
            if mode != 'no_ht':
                print(f"Error: Invalid argument '{mode}'. Only 'no_ht' is allowed. Use 'help' for usage.")
                sys.exit(1)
            return mode
        print("Error: Too many arguments. Use 'help' for usage.")
        sys.exit(1)

if __name__ == '__main__':

    #take as argument 'no_ht' or nothing
    mode = validate_input()

    print(f'RUNNING PIPELINE WITH MODE: {mode}\n')
    
    run_full_pipeline(mode)
    
    

    
