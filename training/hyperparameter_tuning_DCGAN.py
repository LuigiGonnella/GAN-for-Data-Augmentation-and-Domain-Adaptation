import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import argparse

import yaml
import pandas as pd
import random
import os
import copy

from training.train_dcgan import GANTrainer
from evaluation.gan_classifier_evaluation import (
    CLASSIFIER_PATH,
    evaluate_and_plot_classifier,
    compute_combined_score
)

PARAM_DISTRIBUTION = {
    'batch_size': [32, 64],
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

def tune_lr(lr_params, config, id, use_classifier=True, classifier_config=None):
    """
    Tune learning rates for generator and discriminator
    
    Args:
        lr_params: Dict with 'g_lr' and 'd_lr' keys
        config: Config dict
        id: Iteration ID
        use_classifier: Whether to evaluate with classifier
        classifier_config: Classifier config dict or path to yaml file
    
    Returns:
        Tuple of (fid_score, classifier_metrics, config) or None on error
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
    
    # Evaluate with classifier if requested and path exists
    classifier_metrics = None
    if use_classifier and os.path.exists(CLASSIFIER_PATH):
        generator_path = os.path.join(config['output']['checkpoint_dir'], f'final_generator_iter_{id}.pth')
        if os.path.exists(generator_path):
            try:
                plot_dir = os.path.join(config['output']['metrics_dir'], 'plots')
                classifier_metrics = evaluate_and_plot_classifier(
                    generator_path, config, plot_dir, id, classifier_config=classifier_config
                )
                print(f"  Classifier metrics - Precision: {classifier_metrics['precision']:.4f}, Recall: {classifier_metrics['recall']:.4f}")
            except Exception as e:
                print(f"  Warning: Classifier evaluation failed: {e}")
    
    return fid_score, classifier_metrics, config


def run_lr_tuning(config_path, use_classifier=True, classifier_config=None):
    """
    Test all learning rate combinations and return best configuration
    
    Args:
        config_path: Path to base config file
        use_classifier: Whether to use classifier evaluation
        classifier_config: Classifier config dict or path to yaml file
    
    Returns:
        Tuple of (best_g_lr, best_d_lr) based on combined metrics
    """
    print("\n" + "="*60)
    print("RUNNING LEARNING RATE TUNING")
    if use_classifier and os.path.exists(CLASSIFIER_PATH):
        print("Using FID + Classifier Metrics for selection")
    else:
        print("Using FID only for selection")
    print("="*60)
    print(f"Testing {len(LR_COMBINATIONS)} learning rate combinations\n")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f'Exception occurred: {e}')

    results = []
    
    for idx, lr_params in enumerate(LR_COMBINATIONS, 1):
        print(f"\n[{idx}/{len(LR_COMBINATIONS)}] Testing combination:")
        print(f"  Generator LR: {lr_params['g_lr']:.0e}")
        print(f"  Discriminator LR: {lr_params['d_lr']:.0e}")
        
        result = tune_lr(lr_params, copy.deepcopy(config), idx, use_classifier, classifier_config)
        
        if result is not None:
            fid_score, classifier_metrics, config = result
            results.append((fid_score, classifier_metrics, lr_params['g_lr'], lr_params['d_lr']))
            print(f"  → FID Score: {fid_score:.4f}")
        else:
            print(f"  → Failed")
    
    if not results:
        print("\nError: No successful learning rate combinations found.")
        return None, None
    
    # Sort by combined score: prioritize recall (if available), then FID
    if use_classifier and results[0][1] is not None:
        # Get all FID scores for normalization
        fid_scores = [r[0] for r in results]
        
        # Calculate combined scores using the utility function
        scored_results = []
        for fid, clf_metrics, g_lr, d_lr in results:
            combined_score = compute_combined_score(
                fid, clf_metrics['recall'], fid_scores,
                recall_weight=0.6, fid_weight=0.4
            )
            scored_results.append((combined_score, fid, clf_metrics, g_lr, d_lr))
        
        scored_results.sort(key=lambda x: x[0])
        _, best_fid, best_clf_metrics, best_g_lr, best_d_lr = scored_results[0]
        
        print("\nUsing combined scoring: -0.6*recall + 0.4*normalized_fid")
    else:
        # Fall back to FID only
        results.sort(key=lambda x: x[0])
        best_fid, best_clf_metrics, best_g_lr, best_d_lr = results[0]
    
    # Save results to CSV
    results_data = []
    for fid_score, clf_metrics, g_lr, d_lr in results:
        row = {
            'fid_score': fid_score,
            'g_lr': g_lr,
            'd_lr': d_lr
        }
        if clf_metrics:
            row.update({
                'classifier_precision': clf_metrics['precision'],
                'classifier_recall': clf_metrics['recall'],
                'classifier_accuracy': clf_metrics['accuracy'],
                'classifier_f1': clf_metrics['f1']
            })
        results_data.append(row)
    
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
    if best_clf_metrics:
        print(f"\nClassifier Performance on Generated Samples:")
        print(f"  Recall: {best_clf_metrics['recall']:.4f}")
        print(f"  Precision: {best_clf_metrics['precision']:.4f}")
        print(f"  F1-Score: {best_clf_metrics['f1']:.4f}")
        print(f"  Accuracy: {best_clf_metrics['accuracy']:.4f}")
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

def tune_with_hyperparams(hyperparams, config, id, use_classifier=True, classifier_config=None):
    
    print(f"HYPERPARAMETER TUNING - ITERATION {id}")
    print(f"TESTING PARAMS: {hyperparams}")

    
    try:
        config['model']['generator']['latent_dim'] = hyperparams['latent_dim']    
        config['model']['discriminator']['n_layers'] = hyperparams['n_layers']
        config['model']['discriminator']['dropout'] = hyperparams['dropout']
        config['training']['batch_size'] = hyperparams['batch_size']
        config['training']['n_critic'] = hyperparams['n_critic']
        config['training']['epochs'] = CONFIG_EPOCHS

        if not config['output']['sample_dir'].endswith('_ht'):
            config['output']['sample_dir'] += '_ht'
            config['output']['metrics_dir'] += '_ht'
            config['output']['checkpoint_dir'] += '_ht'

        trainer = GANTrainer(config)
        fid_scores = trainer.train(n_iter=id)

        # Evaluate with classifier if requested
        classifier_metrics = None
        if use_classifier and os.path.exists(CLASSIFIER_PATH):
            generator_path = os.path.join(config['output']['checkpoint_dir'], f'final_generator_iter_{id}.pth')
            if os.path.exists(generator_path):
                try:
                    plot_dir = os.path.join(config['output']['metrics_dir'], 'plots')
                    classifier_metrics = evaluate_and_plot_classifier(
                        generator_path, config, plot_dir, id, classifier_config=classifier_config
                    )
                    print(f"  Classifier metrics - Precision: {classifier_metrics['precision']:.4f}, Recall: {classifier_metrics['recall']:.4f}")
                except Exception as e:
                    print(f"  Warning: Classifier evaluation failed: {e}")

        return fid_scores, classifier_metrics, config
        
    except Exception as e:
        print(f"Error with hyperparameter {hyperparams}: {repr(e)}")
        return None

def run_hyperparameter_tuning(config, use_classifier=True, classifier_config=None):
    print("RUNNING HYPERPARAMETER TUNING\n")
    if use_classifier and os.path.exists(CLASSIFIER_PATH):
        print("Using FID + Classifier Metrics for selection")
    else:
        print("Using FID only for selection")
    
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
    
        result = tune_with_hyperparams(params, copy.deepcopy(config), iter+1, use_classifier, classifier_config)
        

        if result is not None:
            fid_scores, classifier_metrics, config = result
            results.append((fid_scores, classifier_metrics, config))  
            print(f"  → FID Score: {fid_scores:.4f}")


    if not results:
        print("Error: No successful hyperparameter combinations found.")
        return None, None
    
    # Sort by combined score if classifier metrics available
    if use_classifier and results[0][1] is not None:
        # Get all FID scores for normalization
        fid_scores = [r[0] for r in results]
        
        # Calculate combined scores using the utility function
        scored_results = []
        for fid, clf_metrics, config in results:
            combined_score = compute_combined_score(
                fid, clf_metrics['recall'], fid_scores,
                recall_weight=0.6, fid_weight=0.4
            )
            scored_results.append((combined_score, fid, clf_metrics, config))
        
        scored_results.sort(key=lambda x: x[0])
        _, res, best_clf_metrics, best_config = scored_results[0]
    else:
        results.sort(key=lambda x: x[0], reverse=False)
        res, best_clf_metrics, best_config = results[0]

    print(f"\nBest FID Score: {res:.4f}")
    if best_clf_metrics:
        print(f"\nClassifier Performance on Generated Samples:")
        print(f"  Recall: {best_clf_metrics['recall']:.4f}")
        print(f"  Precision: {best_clf_metrics['precision']:.4f}")
        print(f"  F1-Score: {best_clf_metrics['f1']:.4f}")
    print(f"\nBest Parameters:")
    print(f"  Latent Dim: {best_config['model']['generator']['latent_dim']}")
    print(f"  N Layers: {best_config['model']['discriminator']['n_layers']}")
    print(f"  Dropout: {best_config['model']['discriminator']['dropout']}")
    print(f"  N Critic: {best_config['training']['n_critic']}")
    print(f"  Batch Size: {best_config['training']['batch_size']}")

    
    try:
        results_data = []
        for idx, (fid_score, clf_metrics, config) in enumerate(results, 1):
            row = {
                'iteration': idx,
                'fid_score': fid_score,
                'latent_dim': config['model']['generator']['latent_dim'],
                'n_layers': config['model']['discriminator']['n_layers'],
                'dropout': config['model']['discriminator']['dropout'],
                'n_critic': config['training']['n_critic'],
                'batch_size': config['training']['batch_size']
            }
            if clf_metrics:
                row.update({
                    'classifier_precision': clf_metrics['precision'],
                    'classifier_recall': clf_metrics['recall'],
                    'classifier_accuracy': clf_metrics['accuracy'],
                    'classifier_f1': clf_metrics['f1']
                })
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        results_dir = Path(best_config['output']['metrics_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = results_dir / "hyperparameter_tuning_results.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"Detailed results saved to {csv_path}")
    except Exception as e:
        print(f"Warning: Could not save detailed results: {e}")

    print("HYPERPARAMETER TUNING COMPLETED.\n")
    
    return best_config

def run_best_config(type, config_path, g_lr, d_lr, use_classifier=True, classifier_config=None):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f'Exception occurred: {e}')

    config['loss']['type'] = type
    
    # Append _ht suffix to output directories if not already present
    if not config['output']['sample_dir'].endswith('_ht'):
        config['output']['sample_dir'] += '_ht'
    if not config['output']['metrics_dir'].endswith('_ht'):
        config['output']['metrics_dir'] += '_ht'
    if not config['output']['checkpoint_dir'].endswith('_ht'):
        config['output']['checkpoint_dir'] += '_ht'
    
    config['training']['g_lr'] = g_lr
    config['training']['d_lr'] = d_lr

    best_config = run_hyperparameter_tuning(config, use_classifier=use_classifier, classifier_config=classifier_config)

    if best_config is None:
        print("Error: Could not run hyperparameter tuning")
        return
    
    print(f'\nBEST CONFIG: {best_config}')

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
    
    # Evaluate final generator with classifier
    if use_classifier and os.path.exists(CLASSIFIER_PATH):
        generator_path = os.path.join(best_config['output']['checkpoint_dir'], 'final_generator.pth')
        if os.path.exists(generator_path):
            try:
                print("\n" + "="*60)
                print("EVALUATING FINAL GENERATOR WITH CLASSIFIER")
                print("="*60)
                plot_dir = os.path.join(best_config['output']['metrics_dir'], 'plots')
                classifier_metrics = evaluate_and_plot_classifier(
                    generator_path, best_config, plot_dir, 'final',
                    classifier_config=classifier_config
                )
                print(f"\nFinal Generator - Classifier Performance:")
                print(f"  Recall:    {classifier_metrics['recall']:.4f}")
                print(f"  Precision: {classifier_metrics['precision']:.4f}")
                print(f"  F1-Score:  {classifier_metrics['f1']:.4f}")
                print(f"  Accuracy:  {classifier_metrics['accuracy']:.4f}")
                print("="*60)
            except Exception as e:
                print(f"\nWarning: Final classifier evaluation failed: {e}")
    
    print(f"\nFinal FID Score: {fid:.4f}")
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
    parser.add_argument(
        '--no_classifier',
        action='store_true',
        help='Disable classifier-based evaluation (use FID only)'
    )
    parser.add_argument(
        '--classifier_config',
        type=str,
        default=None,
        help='Path to classifier config YAML file (defaults to built-in config)'
    )
    args = parser.parse_args()
    
    # Check if classifier is available
    use_classifier = not args.no_classifier and os.path.exists(CLASSIFIER_PATH)
    if not args.no_classifier and not os.path.exists(CLASSIFIER_PATH):
        print(f"Warning: Classifier not found at {CLASSIFIER_PATH}")
        print("Falling back to FID-only evaluation")

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
        type, g_lr, d_lr = run_lr_tuning(args.config, use_classifier=use_classifier, classifier_config=args.classifier_config)

        if g_lr is None:
            print("Error: Learning rate tuning failed")
            sys.exit(1)

        if args.only_lr:
            print("\n" + "="*60)
            print("Learning rate tuning completed. Exiting (--only_lr flag set).")
            print("="*60)
            sys.exit(0)
    
    results = run_best_config(type, args.config, g_lr, d_lr, use_classifier=use_classifier, classifier_config=args.classifier_config)
    