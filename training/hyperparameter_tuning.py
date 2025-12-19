import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
import itertools
from train_classifier import main as train_main

HYPERPARAMETER_GRID = {
    'learning_rate': [0.001, 0.0001, 0.00001],
    'batch_size': [32, 64, 128],
    'weight_decay': [1e-5, 1e-4, 1e-6],
    'class_weight_ratio': [1, 3, 5, 10],  
}

def tune_with_hyperparams(best_strategy, hyperparams):
    
    lr, bs, wd, cw_ratio = hyperparams
    

    print(f"Hyperparameter Tuning")
    print(f"Strategy: {best_strategy}")
    print(f"LR: {lr}, BS: {bs}, WD: {wd}, Class Weight Ratio: 1:{cw_ratio}")
    
    with open('experiments/baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['freezing_strategy'] = best_strategy
    config['training']['lr'] = lr
    config['training']['batch_size'] = bs
    config['training']['weight_decay'] = wd
    config['class_weight_ratio'] = cw_ratio
    
    output_subdir = f"lr_{lr}_bs_{bs}_wd_{wd}_cwr_{cw_ratio}".replace('.', '_')
    config['output_dir'] = f"results/hyperparameter_tuning/{best_strategy}/{output_subdir}"
    
    try:
        metrics = train_main(config)
        
        return {
            'learning_rate': lr,
            'batch_size': bs,
            'weight_decay': wd,
            'class_weight_ratio': cw_ratio,
            'accuracy': metrics['accuracy'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc'],
            'val_loss': metrics['val_loss']
        }
    except Exception as e:
        print(f"Error with hyperparameter {hyperparams}: {e}")
        return None

def run_hyperparameter_tuning(best_strategy):
    
    print(f"# HYPERPARAMETER TUNING - BEST STRATEGY: {best_strategy}")
    
    lr_values = HYPERPARAMETER_GRID['learning_rate']
    bs_values = HYPERPARAMETER_GRID['batch_size']
    wd_values = HYPERPARAMETER_GRID['weight_decay']
    cw_values = HYPERPARAMETER_GRID['class_weight_ratio']
    
    all_combinations = list(itertools.product(lr_values, bs_values, wd_values, cw_values))
    total_combinations = len(all_combinations)
    
    print(f"Total combinations to test: {total_combinations}\n")
    
    results = []
    successful = 0
    
    for idx, combo in enumerate(all_combinations, 1):
        print(f"[{idx}/{total_combinations}] Testing: {combo}")
        result = tune_with_hyperparams(best_strategy, combo)
        
        if result is not None:
            results.append(result)
            successful += 1
        
        if idx % 5 == 0:
            print(f"Progress: {successful} successful, {idx - successful} failed\n")
    
    if not results:
        print("Error: No successful hyperparameter combinations found.")
        return None, None
    
    # Save all results
    results_df = pd.DataFrame(results)
    results_dir = Path(f"results/hyperparameter_tuning/{best_strategy}")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "tuning_results.csv"
    results_df.to_csv(csv_path, index=False)
    
    # Identify best configuration (highest F1)
    best_idx = results_df['f1'].idxmax()
    best_config = results_df.loc[best_idx]
    
    print(f"HYPERPARAMETER TUNING COMPLETED")
    print(f"\nBest configuration:")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  Weight Decay: {best_config['weight_decay']}")
    print(f"  Class Weight Ratio: 1:{best_config['class_weight_ratio']}")
    print(f"\nMetrics:")
    print(f"  Accuracy: {best_config['accuracy']:.4f}")
    print(f"  F1: {best_config['f1']:.4f}")
    print(f"  ROC-AUC: {best_config['roc_auc']:.4f}")
    print(f"  Val Loss: {best_config['val_loss']:.4f}")
    print(f"\nResults saved: {csv_path}\n")
    
    # Show top-5 configurations
    print(f"Top 5 configurations by F1-Score:")
    print(results_df.nlargest(5, 'f1')[['learning_rate', 'batch_size', 'weight_decay', 'class_weight_ratio', 'f1', 'roc_auc']].to_string(index=False))
    print()
    
    return best_config, results_df

if __name__ == '__main__':
    best_strategy = 'freeze_except_last'  # Default
    best_config, results = run_hyperparameter_tuning(best_strategy)
