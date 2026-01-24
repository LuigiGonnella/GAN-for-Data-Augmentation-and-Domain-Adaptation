import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import argparse

import yaml
import pandas as pd
from train_classifier import main as train_main
import random
import os
PARAM_DISTRIBUTION = {
    'weight_decay': [0, 1e-3, 1e-4, 1e-5],
    'batch_size': [32, 64],
    'lr': [1e-1, 1e-2, 1e-3, 1e-4],
    'momentum': [0.8, 0.9, 0.95],
    'optimizer': ['SGD', 'Adam', 'RMSprop', 'AdamW'] ,
}

N_ITERATIONS = 10 #simulate 10 iterations of RandomSearch

BEST_CONFIG_EPOCHS = 10

def tune_with_hyperparams(hyperparams, config_path):
    
    print("BASELINE DATASET TRAINING - HYPERPARAMETER TUNING & FINETUNING'")
    print(f"TESTING PARAMS: {hyperparams}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['training']['params'] = {**config['training']['params'], **hyperparams}    

    
        metrics = train_main(config)
        
        res = hyperparams.copy()
        res['accuracy']= metrics['accuracy']
        res['recall']= metrics['recall']
        res['precision']= metrics['precision']
        res['f1']= metrics['f1']
        res['roc_auc']= metrics['roc_auc']
        res['val_loss']= metrics['val_loss']

        # Clean CUDA cache after each run
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        return res, config
        
    except Exception as e:
        print(f"Error with hyperparameter {hyperparams}: {repr(e)}")
        return None

def run_hyperparameter_tuning(config_path):
    print("RUNNING HYPERPARAMETER TUNING\n")
    
    print(f"Total iterations of RandomSearch: {N_ITERATIONS}\n")
    
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

        print(f"PROCESSING ITERATION {iter+1}")
    
        result, config = tune_with_hyperparams(params, config_path)

        # Clean CUDA cache after run
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        

        if result is not None:
            results.append(result)
    
    if not results:
        print("Error: No successful hyperparameter combinations found.")
        return None, None
    
    # Save all results
    results_df = pd.DataFrame(results)
    results_dir = config['output_dir']
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, "tuning_results.csv")
    results_df.to_csv(csv_path, index=False)
    
    # Identify best configuration (highest Recall for medical imaging)
    best_idx = results_df['recall'].idxmax()
    best_config = results_df.loc[best_idx]
    
    print("HYPERPARAMETER TUNING COMPLETED.\n")
        
    
 
    
    return best_config, results_df

def run_best_config(config_path):

    best_config, results = run_hyperparameter_tuning(config_path)
    print(f'BEST CONFIG: {best_config}\nRESULTS: {results}')

    # run complete training with best config
    params = {
        'lr': float(best_config['lr']),
        'batch_size': int(best_config['batch_size']),
        'weight_decay': float(best_config['weight_decay']),
        'momentum': float(best_config['momentum']),
        'optimizer': best_config['optimizer'],
    }

    print("FINETUNING WITH HYPERPARAMETER TUNING TRAINING - BEST CONFIGURATION TRAINING")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        config['training']['params']['epochs'] = BEST_CONFIG_EPOCHS
        config['training']['params'] = {**config['training']['params'], **params}
        config['best_config_run'] = True

        metrics = train_main(config)
        
        print("FINETUNING WITH HYPERPARAMETER TUNING TRAINING COMPLETED")
        return best_config, metrics
    except Exception as e:
        print(f'Error with file name {config_path}: {e}')
        return None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='BASELINE DATASET TRAINING - HYPERPARAMETER TUNING & FINETUNING'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to config YAML file'
    )
    args = parser.parse_args()
    
    best_config, results = run_best_config(args.config)
    print(f'\nBEST CONFIG: {best_config}\n')
    print(f'METRICS: {results}\n')
    