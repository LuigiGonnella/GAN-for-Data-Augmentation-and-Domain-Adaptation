import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from train_classifier import main as train_main
import pandas as pd

FREEZE_STRATEGIES = {
    'freeze_except_last': {'freeze': True, 'layers': None},
    'freeze_last_2_blocks': {'freeze': True, 'layers': 2},
    'progressive': {'freeze': 'progressive', 'layers': None},
}

def fine_tune_with_strategy(strategy_name, strategy_config):
    
    print(f"Fine-tuning with: {strategy_name}")
    

    with open('experiments/baseline.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['freezing_strategy'] = strategy_name
    config['freeze_config'] = strategy_config
    config['output_dir'] = f"results/fine_tuning/{strategy_name}"
    
    metrics = train_main(config)
    
    return {
        'strategy': strategy_name,
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'roc_auc': metrics['roc_auc'],
        'val_loss': metrics['val_loss']
    }

def run_fine_tuning_comparison():
    
    results = []
    
    for strategy_name, strategy_config in FREEZE_STRATEGIES.items():
        try:
            result = fine_tune_with_strategy(strategy_name, strategy_config)
            results.append(result)
        except Exception as e:
            print(f"Error with {strategy_name}: {e}")
            continue
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/fine_tuning/comparison.csv', index=False)
    
    # Identify best strategy (best F1 score)
    best_idx = results_df['f1'].idxmax()
    best_strategy = results_df.loc[best_idx, 'strategy']
    
    print(f"BEST STRATEGY: {best_strategy}")
    print(results_df.to_string())
    
    return best_strategy, results_df

if __name__ == '__main__':
    best_strategy, results = run_fine_tuning_comparison()
    print(f"Best strategy: {best_strategy}")