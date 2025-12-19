import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from train_classifier import main as train_main

def run_baseline():

    print("BASELINE TRAINING - no_freeze (all ResNet50 layers trainable)")
    
    with open('experiments/baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['freezing_strategy'] = 'no_freeze'
    config['output_dir'] = 'results/baseline'
    
    metrics = train_main(config)
    
    print("BASELINE COMPLETED")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Val Loss: {metrics['val_loss']:.4f}")
    print(f"Model saved: results/baseline/classifier.pth\n")
    
    return metrics

if __name__ == '__main__':
    run_baseline()
