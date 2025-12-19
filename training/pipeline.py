import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from train_classifier import main as train_main
from finetune_classifier import run_fine_tuning_comparison
from hyperparameter_tuning import run_hyperparameter_tuning
import yaml
import json
from datetime import datetime

def run_full_pipeline():
    
    # STEP 0: Baseline Training
    print("# STEP 0: Baseline Training (no_freeze)")

    baseline_metrics = run_baseline_training()
    
    print(f"\n✓ Baseline completed. F1: {baseline_metrics['f1']:.4f}, Accuracy: {baseline_metrics['accuracy']:.4f}")

    # STEP 1: Fine-tuning Comparison
    print("# STEP 1: Fine-Tuning Comparison")
    
    best_strategy, finetune_results = run_fine_tuning_comparison()
    
    print(f"\n✓ Fine-tuning completed. Best strategy: {best_strategy}")
    
    # STEP 2: Hyperparameter Tuning with Best Strategy
    print("# STEP 2: Hyperparameter Tuning with Best Strategy")
    
    best_config, tuning_results = run_hyperparameter_tuning(best_strategy)
    
    if best_config is None:
        print("Hyperparameter tuning failed!")
        return
    
    print(f"✓ Hyperparameter tuning completed.")
    
    # STEP 3: Final Report
    print("# STEP 3: Final Report")
    
    generate_final_report(baseline_metrics, best_strategy, finetune_results, best_config, tuning_results)
    
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"End time: {datetime.now()}")

def run_baseline_training():

    with open('experiments/baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['freezing_strategy'] = 'no_freeze'
    config['output_dir'] = 'results/baseline'
    
    metrics = train_main(config)
    
    return metrics

def generate_final_report(baseline_metrics, best_strategy, finetune_results, best_config, tuning_results):
    
    report_dir = Path("results/final_report")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = report_dir / "optimization_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("FINAL OPTIMIZATION REPORT\n")
        
        f.write("STEP 0: BASELINE TRAINING\n")
        f.write(f"Strategy: no_freeze (all ResNet50 layers trainable)\n")
        f.write(f"Metrics:\n")
        f.write(f"  - Accuracy: {baseline_metrics['accuracy']:.4f}\n")
        f.write(f"  - F1-Score: {baseline_metrics['f1']:.4f}\n")
        f.write(f"  - ROC-AUC: {baseline_metrics['roc_auc']:.4f}\n")
        f.write(f"  - Val Loss: {baseline_metrics['val_loss']:.4f}\n\n")
        
        f.write("STEP 1: FINE-TUNING COMPARISON\n")
        f.write("Tested strategies:\n")
        for _, row in finetune_results.iterrows():
            f.write(f"  - {row['strategy']}: F1={row['f1']:.4f}, Acc={row['accuracy']:.4f}, AUC={row['roc_auc']:.4f}\n")
        f.write(f"\n✓ Best strategy: {best_strategy}\n\n")
        
        f.write("STEP 2: HYPERPARAMETER TUNING\n")
        f.write("Best hyperparameter configuration:\n")
        f.write(f"  - Learning Rate: {best_config['learning_rate']}\n")
        f.write(f"  - Batch Size: {best_config['batch_size']}\n")
        f.write(f"  - Weight Decay: {best_config['weight_decay']}\n")
        f.write(f"  - Class Weight Ratio: 1:{best_config['class_weight_ratio']}\n\n")
        
        f.write("Final Metrics:\n")
        f.write(f"  - Accuracy: {best_config['accuracy']:.4f}\n")
        f.write(f"  - F1-Score: {best_config['f1']:.4f}\n")
        f.write(f"  - ROC-AUC: {best_config['roc_auc']:.4f}\n")
        f.write(f"  - Validation Loss: {best_config['val_loss']:.4f}\n\n")
        
        f.write("COMPARISON: Baseline vs Best Fine-tuned vs Final Tuned\n")
        best_finetune = finetune_results[finetune_results['strategy'] == best_strategy]['f1'].values[0]
        improvement_finetune = (best_finetune - baseline_metrics['f1']) / baseline_metrics['f1'] * 100
        improvement_final = (best_config['f1'] - baseline_metrics['f1']) / baseline_metrics['f1'] * 100
        f.write(f"  Baseline F1: {baseline_metrics['f1']:.4f}\n")
        f.write(f"  Best Fine-tune ({best_strategy}) F1: {best_finetune:.4f} ({improvement_finetune:+.2f}%)\n")
        f.write(f"  After Hyperparameter Tuning F1: {best_config['f1']:.4f} ({improvement_final:+.2f}%)\n\n")
        
        f.write("Model location: results/hyperparameter_tuning/{}/{}/classifier.pth\n".format(
            best_strategy,
            f"lr_{best_config['learning_rate']}_bs_{best_config['batch_size']}_wd_{best_config['weight_decay']}_cwr_{best_config['class_weight_ratio']}".replace('.', '_')
        ))
        f.write("Best hyperparameter configuration:\n")
        f.write(f"  - Learning Rate: {best_config['learning_rate']}\n")
        f.write(f"  - Batch Size: {best_config['batch_size']}\n")
        f.write(f"  - Weight Decay: {best_config['weight_decay']}\n")
        f.write(f"  - Class Weight Ratio: 1:{best_config['class_weight_ratio']}\n\n")
        
        f.write("Final Metrics:\n")
        f.write(f"  - Accuracy: {best_config['accuracy']:.4f}\n")
        f.write(f"  - F1-Score: {best_config['f1']:.4f}\n")
        f.write(f"  - ROC-AUC: {best_config['roc_auc']:.4f}\n")
        f.write(f"  - Validation Loss: {best_config['val_loss']:.4f}\n\n")
        
        f.write("COMPARISON: Fine-tuning Best vs Final Tuned\n")
        baseline_f1 = finetune_results[finetune_results['strategy'] == best_strategy]['f1'].values[0]
        improvement = (best_config['f1'] - baseline_f1) / baseline_f1 * 100
        f.write(f"  Fine-tuning F1: {baseline_f1:.4f}\n")
        f.write(f"  After tuning F1: {best_config['f1']:.4f}\n")
        f.write(f"  Improvement: {improvement:+.2f}%\n\n")
        
        f.write("Model location: results/hyperparameter_tuning/{}/{}/classifier.pth\n".format(
            best_strategy,
            f"lr_{best_config['learning_rate']}_bs_{best_config['batch_size']}_wd_{best_config['weight_decay']}_cwr_{best_config['class_weight_ratio']}".replace('.', '_')
        ))
    
    print(f"✓ Report saved: {report_path}")
    
    summary_path = report_dir / "best_config.json"
    summary = {
        'best_strategy': best_strategy,
        'best_hyperparameters': {
            'learning_rate': float(best_config['learning_rate']),
            'batch_size': int(best_config['batch_size']),
            'weight_decay': float(best_config['weight_decay']),
            'class_weight_ratio': int(best_config['class_weight_ratio'])
        },
        'final_metrics': {
            'accuracy': float(best_config['accuracy']),
            'f1_score': float(best_config['f1']),
            'roc_auc': float(best_config['roc_auc']),
            'val_loss': float(best_config['val_loss'])
        },
        'model_path': f"results/hyperparameter_tuning/{best_strategy}/lr_{best_config['learning_rate']}_bs_{best_config['batch_size']}_wd_{best_config['weight_decay']}_cwr_{best_config['class_weight_ratio']}/classifier.pth".replace('.', '_')
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Config saved: {summary_path}")

if __name__ == '__main__':
    run_full_pipeline()
