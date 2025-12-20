import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train_classifier import main as train_main
from finetune_classifier import run_fine_tuning
from hyperparameter_tuning import run_hyperparameter_tuning
import yaml
import json
from datetime import datetime

def run_full_pipeline():
    
    # STEP 0: Freezing
    print("# STEP 0: Baseline Training (no_freeze)")

    baseline_metrics = run_baseline_training()
    
    print(f"\n✓ Baseline completed.\nResults:{baseline_metrics}")

    # STEP 1: Fine-Tuning 
    print("# STEP 1: Fine-Tuning")
    
    finetune_results = run_fine_tuning()
    
    print(f"\n✓ Fine-tuning completed.\nResults:{finetune_results}")
    
    # STEP 2: Fine-Tuning and Hyperparameter Tuning 
    print("# STEP 2: Hyperparameter Tuning with Best Strategy")
    
    best_config, tuning_results = run_hyperparameter_tuning()
    
    print(f"✓ Hyperparameter tuning completed.\nResults:{tuning_results}\n")
    print(f'BEST CONFIGURATION:/n{best_config}')
    
    # STEP 3: Final Report
    print("# STEP 3: Final Report")
    
    generate_final_report(baseline_metrics, finetune_results, tuning_results, best_config)
    
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"End time: {datetime.now()}")

def run_baseline_training():

    with open('experiments/baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    
    metrics = train_main(config)
    
    return {
            'accuracy': metrics['accuracy'],
            'recall': metrics['recall'],
            'precision': metrics['precision'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc'],
            'val_loss': metrics['val_loss']
            }


FINAL_METRICS = "Final Metrics:\n"
def generate_final_report(baseline_metrics, finetune_results, tuning_results, best_config):
    
    report_dir = Path("results/final_report")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / "optimization_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("FINAL OPTIMIZATION REPORT\n")
        
        f.write("STEP 0: BASELINE TRAINING (freeze)\n")
        f.write()
        f.write(f"  - Accuracy: {baseline_metrics['accuracy']:.4f}\n")
        f.write(f"  - F1-Score: {baseline_metrics['f1']:.4f}\n")
        f.write(f"  - ROC-AUC: {baseline_metrics['roc_auc']:.4f}\n")
        f.write(f"  - Val Loss: {baseline_metrics['val_loss']:.4f}\n\n")
        
        f.write("STEP 1: FINE-TUNING\n")
        f.write(FINAL_METRICS)
        f.write(f"  - Accuracy: {finetune_results['accuracy']:.4f}\n")
        f.write(f"  - F1-Score: {finetune_results['f1']:.4f}\n")
        f.write(f"  - ROC-AUC: {finetune_results['roc_auc']:.4f}\n")
        f.write(f"  - Val Loss: {finetune_results['val_loss']:.4f}\n\n")
        
        f.write("STEP 2: HYPERPARAMETER TUNING\n")
        f.write(f"Best hyperparameter configuration:\n{best_config}\n")
        
        f.write(FINAL_METRICS)
        f.write(f"  - Accuracy: {tuning_results['accuracy']:.4f}\n")
        f.write(f"  - F1-Score: {tuning_results['f1']:.4f}\n")
        f.write(f"  - ROC-AUC: {tuning_results['roc_auc']:.4f}\n")
        f.write(f"  - Validation Loss: {tuning_results['val_loss']:.4f}\n\n")
        
        f.write("COMPARISON: Baseline vs Fine-tuned vs Fine Tuned and Hyperparameter Tuned\n")
        improvement_finetune = (finetune_results['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100
        improvement_final = (tuning_results['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100
        f.write(f"  Baseline F1: {baseline_metrics['f1']:.4f}\n")
        f.write(f" Fine Tune F1: {finetune_results:.4f} ({improvement_finetune:+.2f}%)\n")
        f.write(f"  After Hyperparameter Tuning F1: {tuning_results['f1']:.4f} ({improvement_final:+.2f}%)\n\n")
        
        f.write("Model location: results/hyperparameter_tuning/{}/classifier.pth\n".format(
            f"lr_{best_config['learning_rate']}_bs_{best_config['batch_size']}_wd_{best_config['weight_decay']}_{best_config['optimizer']}_cwr".replace('.', '_')
        ))
        f.write("Best hyperparameter configuration:\n")
        f.write(f"  - Learning Rate: {best_config['learning_rate']}\n")
        f.write(f"  - Batch Size: {best_config['batch_size']}\n")
        f.write(f"  - Weight Decay: {best_config['weight_decay']}\n")
        
        f.write(FINAL_METRICS)
        f.write(f"  - Accuracy: {tuning_results['accuracy']:.4f}\n")
        f.write(f"  - F1-Score: {tuning_results['f1']:.4f}\n")
        f.write(f"  - ROC-AUC: {tuning_results['roc_auc']:.4f}\n")
        f.write(f"  - Validation Loss: {tuning_results['val_loss']:.4f}\n\n")
        

    print(f"✓ Report saved: {report_path}")
    
    summary_path = report_dir / "best_config.json"
    summary = {
        'best_hyperparameters': {
            'learning_rate': float(best_config['learning_rate']),
            'batch_size': int(best_config['batch_size']),
            'weight_decay': float(best_config['weight_decay']),
        },
        'final_metrics': {
            'accuracy': float(best_config['accuracy']),
            'f1_score': float(best_config['f1']),
            'roc_auc': float(best_config['roc_auc']),
            'val_loss': float(best_config['val_loss'])
        },
        'model_path': f"results/hyperparameter_tuning/lr_{best_config['learning_rate']}_bs_{best_config['batch_size']}_wd_{best_config['weight_decay']}_{best_config['optimizer']}_cwr/classifier.pth".replace('.', '_')
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Config saved: {summary_path}")

if __name__ == '__main__':
    run_full_pipeline()
