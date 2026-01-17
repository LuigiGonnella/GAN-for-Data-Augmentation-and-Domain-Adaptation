import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import argparse

import yaml
import pandas as pd
import random
import os

from training.train_dcgan import GANTrainer
PARAM_DISTRIBUTION = {
    'betas': [(0.5, 0.999), (0, 0.9), (0.3, 0.999)],
    'batch_size': [32, 64],
    'optimizer': ['Adam', 'AdamW'],
    'latent_dim': [100, 128, 256],
    'n_layers': [2, 3],
    'dropout': [0.1, 0.3],
    'n_critic': [1, 2]
}

N_ITERATIONS = 10 #simulate 10 iterations of RandomSearch

CONFIG_EPOCHS = 25

# Learning rate combinations to test (all combinations, not randomized)
LR_COMBINATIONS = [
    {'g_lr': 2e-4, 'd_lr': 2e-4},
    {'g_lr': 1e-4, 'd_lr': 2e-4},
    {'g_lr': 2e-4, 'd_lr': 1e-4},
    {'g_lr': 1e-4, 'd_lr': 1e-4},
    {'g_lr': 3e-4, 'd_lr': 3e-4},
]

def tune_lr(lr_params, config, id):
    """
    Tune learning rates for generator and discriminator
    
    Args:
        lr_params: Dict with 'g_lr' and 'd_lr' keys
        config_path: Path to base config file
    
    Returns:
        Tuple of (fid_score, config) or None on error
    """    
    config['training']['g_lr'] = lr_params['g_lr']
    config['training']['d_lr'] = lr_params['d_lr']
    config['training']['epochs'] = CONFIG_EPOCHS
    
    if not config['output']['sample_dir'].endswith('_lr'):
        config['output']['sample_dir'] += '_ht_lr'
        config['output']['metrics_dir'] += '_ht_lr'
        config['output']['checkpoint_dir'] += '_ht_lr'

    trainer = GANTrainer(config)
    fid_score = trainer.train(n_iter=id)
    
    return fid_score, config


def run_lr_tuning(config_path):
    """
    Test all learning rate combinations and return best configuration
    
    Args:
        config_path: Path to base config file
    
    Returns:
        Tuple of (best_g_lr, best_d_lr) with lowest FID score
    """
    print("\n" + "="*60)
    print("RUNNING LEARNING RATE TUNING")
    print("="*60)
    print(f"Testing {len(LR_COMBINATIONS)} learning rate combinations\n")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f'Exception ccurred: {e}')

    results = []
    
    for idx, lr_params in enumerate(LR_COMBINATIONS, 1):
        print(f"\n[{idx}/{len(LR_COMBINATIONS)}] Testing combination:")
        print(f"  Generator LR: {lr_params['g_lr']:.0e}")
        print(f"  Discriminator LR: {lr_params['d_lr']:.0e}")
        
        result = tune_lr(lr_params, config.copy(), idx)
        
        if result is not None:
            fid_score, config = result
            results.append((fid_score, lr_params['g_lr'], lr_params['d_lr']))
            print(f"  → FID Score: {fid_score:.4f}")
        else:
            print(f"  → Failed")
    
    if not results:
        print("\nError: No successful learning rate combinations found.")
        return None, None
    
    # Sort by FID score (lower is better)
    results.sort(key=lambda x: x[0])
    best_fid, best_g_lr, best_d_lr = results[0]
    
    # Save results to CSV
    results_data = []
    for fid_score, g_lr, d_lr in results:
        results_data.append({
            'fid_score': fid_score,
            'g_lr': g_lr,
            'd_lr': d_lr
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Create output directory
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        results_dir = Path(config['output']['metrics_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / "lr_tuning_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
    except Exception as e:
        print(f"\nWarning: Could not save results: {e}")
    
    print("\n" + "="*60)
    print("LEARNING RATE TUNING COMPLETED")
    print("="*60)
    print(f"\nBest learning rates (FID: {best_fid:.4f}):")
    print(f"  Generator LR: {best_g_lr:.0e}")
    print(f"  Discriminator LR: {best_d_lr:.0e}")
    print("\nAll results:")
    print(results_df.to_string(index=False))
    print("="*60 + "\n")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        loss_type = config['loss']['type']
    except:
        return None, None, None
    
    return loss_type, best_g_lr, best_d_lr

def tune_with_hyperparams(hyperparams, config, id):
    
    print(f"HYPERPARAMETER TUNING - ITERATION {id}")
    print(f"TESTING PARAMS: {hyperparams}")

    
    try:
        config['model']['generator']['latent_dim'] = hyperparams['latent_dim']    
        config['model']['discriminator']['n_layers'] = hyperparams['n_layers']
        config['model']['discriminator']['dropout'] = hyperparams['dropout']
        config['training']['n_critic'] = hyperparams['n_critic']
        config['training']['epochs'] = CONFIG_EPOCHS

        if not config['output']['sample_dir'].endswith('_ht'):
            config['output']['sample_dir'] += '_ht'
            config['output']['metrics_dir'] += '_ht'
            config['output']['checkpoint_dir'] += '_ht'

        trainer = GANTrainer(config)
        fid_scores = trainer.train(n_iter=id)

        return fid_scores, config
        
    except Exception as e:
        print(f"Error with hyperparameter {hyperparams}: {repr(e)}")
        return None

def run_hyperparameter_tuning(config):
    print("RUNNING HYPERPARAMETER TUNING\n")
    
    print(f"Total iterations of RandomSearch: {N_ITERATIONS}\n")
    
    results = []
    
    for iter in range(N_ITERATIONS):
        #choose parameters for this run
        params = {
        'batch_size': random.choice(PARAM_DISTRIBUTION['batch_size']),
        'latent_dim': random.choice(PARAM_DISTRIBUTION['latent_dim']),
        'n_layers': random.choice(PARAM_DISTRIBUTION['n_layers']),
        'dropout': random.choice(PARAM_DISTRIBUTION['dropout']),
        'n_critic': random.choice(PARAM_DISTRIBUTION['n_critic'])
        }

        print(f"PROCESSING ITERATION {iter+1}")
    
        fid_scores, config = tune_with_hyperparams(params, config.copy(), iter+1)
        

        if fid_scores is not None:
            results.append((fid_scores, config))  
            print(f"  → FID Score: {fid_scores:.4f}")


    if not results:
        print("Error: No successful hyperparameter combinations found.")
        return None, None
    
   
    results.sort(key=lambda x: x[0], reverse=False)
    res, best_config = results[0]

    print(f"\nBest FID Score: {res:.4f}")
    print(f"\nBest Parameters:")
    print(f"  Latent Dim: {best_config['model']['generator']['latent_dim']}")
    print(f"  N Layers: {best_config['model']['discriminator']['n_layers']}")
    print(f"  Dropout: {best_config['model']['discriminator']['dropout']}")
    print(f"  N Critic: {best_config['training']['n_critic']}")
    print(f"  Batch Size: {best_config['training']['batch_size']}")

    
    try:
        results_data = []
        for idx, (fid_score, config) in enumerate(results, 1):
            results_data.append({
                'iteration': idx,
                'fid_score': fid_score,
                'latent_dim': config['model']['generator']['latent_dim'],
                'n_layers': config['model']['discriminator']['n_layers'],
                'dropout': config['model']['discriminator']['dropout'],
                'n_critic': config['training']['n_critic'],
                'batch_size': config['training']['batch_size']
            })
        
        results_df = pd.DataFrame(results_data)
        results_dir = Path(best_config['output']['metrics_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / "hyperparameter_tuning_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Detailed results saved to {csv_path}")
    except Exception as e:
        print(f"Warning: Could not save detailed results: {e}")

    print("HYPERPARAMETER TUNING COMPLETED.\n")
    
    return best_config, res

def run_best_config(type, config_path, g_lr, d_lr):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f'Exception occurred: {e}')

    config['loss']['type'] = type
    config['output']['sample_dir'] += f'{type}_ht'
    config['output']['metrics_dir'] += f'{type}_ht'
    config['output']['checkpoint_dir'] += f'_{type}_ht'
    config['training']['g_lr'] = g_lr
    config['training']['d_lr'] = d_lr

    best_config, best_fid = run_hyperparameter_tuning(config)

    if best_config is None:
        print("Error: Could not run hyperparameter tuning")
        return
    
    print(f'\nBEST CONFIG FID: {best_fid:.4f}')

    best_config['training']['epochs'] = 300

    if '_ht' in best_config['output']['sample_dir']:
        best_config['output']['sample_dir'] = best_config['output']['sample_dir'].replace('_ht', '_final')
    else:
        best_config['output']['sample_dir'] += '_final'
    
    if '_ht' in best_config['output']['metrics_dir']:
        best_config['output']['metrics_dir'] = best_config['output']['metrics_dir'].replace('_ht', '_final')
    else:
        best_config['output']['metrics_dir'] += '_final'
    
    if '_ht' in best_config['output']['checkpoint_dir']:
        best_config['output']['checkpoint_dir'] = best_config['output']['checkpoint_dir'].replace('_ht', '_final')
    else:
        best_config['output']['checkpoint_dir'] += '_final'

    print("FINETUNING WITH HYPERPARAMETER TUNING TRAINING - BEST CONFIGURATION TRAINING")

   
    trainer = GANTrainer(best_config)
    fid = trainer.train()
    
    print(f"Final FID Score: {fid:.4f}")
    print(f"Results saved to: {best_config['output']['sample_dir']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='BASELINE DATASET TRAINING - HYPERPARAMETER TUNING'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to config YAML file with a specified loss'
    )
    parser.add_argument(
        '--only_lr',
        action='store_true',
        help='Only run learning rate tuning and exit (skip hyperparameter tuning)'
    )
    parser.add_argument(
        '--not_lr',
        action='store_true',
        help='Skip learning rate tuning and use learning rates from config file'
    )
    args = parser.parse_args()

    if args.not_lr:
        # Skip LR tuning, use values from config
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            type = config['loss']['type']
            g_lr = config['training']['g_lr']
            d_lr = config['training']['d_lr']
            print(f"Skipping LR tuning. Using config values: g_lr={g_lr:.0e}, d_lr={d_lr:.0e}")
        except Exception as e:
            print(f'Error reading config: {e}')
            sys.exit(1)
    else:
        type, g_lr, d_lr = run_lr_tuning(args.config)

        if g_lr is None:
            print("Error: Learning rate tuning failed")
            sys.exit(1)

        if args.only_lr:
            print("\n" + "="*60)
            print("Learning rate tuning completed. Exiting (--only_lr flag set).")
            print("="*60)
            sys.exit(0)

    config_path = 'experiments/dcgan_ht.yaml'
    
    results = run_best_config(type, config_path, g_lr, d_lr)
    