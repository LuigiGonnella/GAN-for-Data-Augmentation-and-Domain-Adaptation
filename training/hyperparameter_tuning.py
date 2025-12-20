import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import pandas as pd
from train_classifier import main as train_main
import random

PARAM_DISTRIBUTION = {
    'weight_decay': [0, 1e-3, 1e-4, 1e-5],
    'batch_size': [16, 32, 64, 128],
    'lr': [1e-1, 1e-2, 1e-3, 1e-4],
    'momentum': [0.8, 0.9, 0.95],
    'optimizer': ['SGD', 'Adam', 'RMSprop', 'AdamW'] ,
}

N_ITERATIONS = 5 #simulate 5 iterations of RandomSearch

def tune_with_hyperparams(hyperparams):
        
    print(f"TESTING PARAMS: {hyperparams}")
    
    with open('experiments/baseline_ft_ht.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['training']['params'] = {**config['training']['params'], **hyperparams}    

    try:
        metrics = train_main(config)
        
        res = hyperparams
        res['accuracy']= metrics['accuracy']
        res['recall']= metrics['recall']
        res['precision']= metrics['precision']
        res['f1']= metrics['f1']
        res['roc_auc']= metrics['roc_auc']
        res['val_loss']= metrics['val_loss']

        return res
        
    except Exception as e:
        print(f"Error with hyperparameter {hyperparams}: {e}")
        return None

def run_hyperparameter_tuning():
    print("RUNNING HYPERPARAMETER TUNING\n")
    
    print(f"Total iterazion of RandomSearch: {N_ITERATIONS}\n")
    
    results = []
    
    for iter in range(N_ITERATIONS):
        #choose parameters for this run
        params = {
        'weight_decay': random.choice(PARAM_DISTRIBUTION['weight_decay']),
        'batch_size': random.choice(PARAM_DISTRIBUTION['batch_size']),
        'lr': random.choice(PARAM_DISTRIBUTION['lr']),
        'momentum': random.choice(PARAM_DISTRIBUTION['momentum']),
        'optimizer': random.choice(PARAM_DISTRIBUTION['optimizer']) 
        }

        print(f"PROCESSING ITERATION {iter}")
    
        result = tune_with_hyperparams(params)
        
        if result is not None:
            results.append(result)
    
    if not results:
        print("Error: No successful hyperparameter combinations found.")
        return None, None
    
    # Save all results
    results_df = pd.DataFrame(results)
    results_dir = Path("results/hyperparameter_tuning")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "tuning_results.csv"
    results_df.to_csv(csv_path, index=False)
    
    # Identify best configuration (highest Recall for medical imaging)
    best_idx = results_df['recall'].idxmax()
    best_config = results_df.loc[best_idx]
    
    print("HYPERPARAMETER TUNING COMPLETED.\n")
        
    
 
    
    return best_config, results_df

if __name__ == '__main__':
    best_config, results = run_hyperparameter_tuning()
    print(f'BEST CONFIG: {best_config}\nRESULTS: {results}')
