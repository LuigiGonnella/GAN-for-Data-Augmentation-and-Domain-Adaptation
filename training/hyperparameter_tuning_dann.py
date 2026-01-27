"""Hyperparameter Tuning for Domain Adversarial Neural Network (DANN)

This script performs random search hyperparameter tuning for DANN training:
1. Randomly samples hyperparameter combinations
2. Trains DANN with each combination for a few epochs
3. Selects best configuration based on target recall (or F1)
4. Performs full training with best configuration

Key DANN-specific hyperparameters:
- feature_dim: Dimensionality of domain-invariant features
- learning_rate: Learning rate for all components
- weight_decay: L2 regularization strength
- batch_size: Batch size for training
- optimizer: Optimizer choice (Adam, AdamW, SGD)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import argparse
import yaml
import pandas as pd
import random
import os
import torch

# Import the main training function to reuse code
from train_with_dann import train_dann

# Hyperparameter search space for DANN
PARAM_DISTRIBUTION = {
    'feature_dim': [256, 512, 1024],
    'weight_decay': [0, 1e-4, 1e-5, 1e-6],
    'batch_size': [32, 64],
    'lr': [1e-3, 1e-4, 5e-4],
    'optimizer': ['Adam', 'AdamW', 'RMSProp'],
}

N_ITERATIONS = 10  # Number of random search iterations
TUNING_EPOCHS = 5  # Quick epochs for hyperparameter evaluation
BEST_CONFIG_EPOCHS = 50  # Full training with best config


def train_dann_with_hyperparams(hyperparams, config_path, 
                                  source_train_dir, source_val_dir, 
                                  target_dir, output_dir):
    """Train DANN with specific hyperparameters for tuning.
    
    This function modifies the config file with tuning hyperparameters
    and calls the main train_dann function to reuse existing code.
    
    Args:
        hyperparams: Dictionary of hyperparameters to test
        config_path: Base configuration file path
        source_train_dir: Source training data directory
        source_val_dir: Source validation data directory
        target_dir: Target domain data directory
        output_dir: Output directory for this run
        
    Returns:
        Dictionary with hyperparameters and resulting metrics
    """
    print(f"\nTesting hyperparameters: {hyperparams}")
    
    try:
        # Load and modify config with tuning hyperparameters
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['training']['learning_rate'] = hyperparams['lr']
        config['training']['batch_size'] = hyperparams['batch_size']
        config['training']['weight_decay'] = hyperparams['weight_decay']
        config['training']['num_epochs'] = TUNING_EPOCHS
        config['feature_dim'] = hyperparams['feature_dim']
        
        # Save temporary config for this run
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        temp_config_path = output_path / 'temp_config.yaml'
        
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Train using the main train_dann function to reuse all logic
        _, source_metrics, target_metrics, domain_gap = train_dann(
            str(temp_config_path),
            source_train_dir,
            source_val_dir,
            target_dir,
            str(output_path)
        )
        
        # Prepare results
        result = hyperparams.copy()
        result['target_accuracy'] = target_metrics.get('accuracy', 0)
        result['target_recall'] = target_metrics.get('recall', 0)
        result['target_f1'] = target_metrics.get('f1', 0)
        result['source_accuracy'] = source_metrics.get('accuracy', 0)
        result['source_recall'] = source_metrics.get('recall', 0)
        result['source_f1'] = source_metrics.get('f1', 0)
        result['accuracy_gap'] = domain_gap.get('accuracy_drop', 0)
        result['f1_gap'] = domain_gap.get('f1_drop', 0)
        result['recall_gap'] = domain_gap.get('recall_drop', 0)
        
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Results - Target Recall: {result['target_recall']:.4f}, Target F1: {result['target_f1']:.4f}")
        
        return result
        
    except Exception as e:
        print(f"Error with hyperparameters {hyperparams}: {repr(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_hyperparameter_tuning(config_path, source_train_dir, source_val_dir, 
                                target_dir, output_base_dir):
    """Run random search hyperparameter tuning.
    
    Args:
        config_path: Base configuration file
        source_train_dir: Source training data directory
        source_val_dir: Source validation data directory
        target_dir: Target domain directory
        output_base_dir: Base output directory for all runs
        
    Returns:
        best_config: Best hyperparameter configuration
        results_df: DataFrame with all tuning results
    """
    print("="*70)
    print("DANN HYPERPARAMETER TUNING - RANDOM SEARCH")
    print("="*70)
    print(f"Total iterations: {N_ITERATIONS}")
    print(f"Tuning epochs per iteration: {TUNING_EPOCHS}\n")
    
    results = []
    
    for iteration in range(N_ITERATIONS):
        # Sample random hyperparameters
        params = {
            'feature_dim': random.choice(PARAM_DISTRIBUTION['feature_dim']),
            'weight_decay': random.choice(PARAM_DISTRIBUTION['weight_decay']),
            'batch_size': random.choice(PARAM_DISTRIBUTION['batch_size']),
            'lr': random.choice(PARAM_DISTRIBUTION['lr']),
            'optimizer': random.choice(PARAM_DISTRIBUTION['optimizer']),
        }
        
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration + 1}/{N_ITERATIONS}")
        print(f"{'='*70}")
        
        output_dir = os.path.join(output_base_dir, f"tuning_iter_{iteration + 1}")
        
        result = train_dann_with_hyperparams(
            params, config_path,
            source_train_dir, source_val_dir,
            target_dir, output_dir
        )
        
        if result is not None:
            results.append(result)
        
        # Clean CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if not results:
        print("\nError: No successful hyperparameter combinations found.")
        return None, None
    
    # Save all results
    results_df = pd.DataFrame(results)
    results_dir = Path(output_base_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = results_dir / "dann_tuning_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nAll tuning results saved to: {csv_path}")
    
    # Select best configuration based on target F1 (balances precision and recall)
    best_idx = results_df['target_f1'].idxmax()
    best_config = results_df.loc[best_idx]
    
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("="*70)
    print(f"Best configuration (by target F1):")
    print(best_config)
    print("="*70)
    
    return best_config, results_df


def train_best_config(config_path, best_config, source_train_dir, source_val_dir,
                       target_dir, output_dir):
    """Train DANN with best configuration for full epochs.
    
    This function reuses the main train_dann function with the best
    hyperparameters identified during tuning.
    
    Args:
        config_path: Base configuration file
        best_config: Best hyperparameter configuration from tuning
        source_train_dir: Source training data directory
        source_val_dir: Source validation data directory
        target_dir: Target domain directory
        output_dir: Output directory for best model
        
    Returns:
        Dictionary with final metrics
    """
    print("\n" + "="*70)
    print("TRAINING WITH BEST CONFIGURATION")
    print("="*70)
    print(f"Training for {BEST_CONFIG_EPOCHS} epochs with best hyperparameters\n")
    
    # Load and modify config with best hyperparameters
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['training']['learning_rate'] = float(best_config['lr'])
    config['training']['batch_size'] = int(best_config['batch_size'])
    config['training']['weight_decay'] = float(best_config['weight_decay'])
    config['training']['num_epochs'] = BEST_CONFIG_EPOCHS
    config['feature_dim'] = int(best_config['feature_dim'])
    
    # Save best config
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config_save_path = output_path / 'best_config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Best configuration saved to: {config_save_path}")
    
    # Train with best config using the main train_dann function (reuses all code)
    _, source_metrics, target_metrics, domain_gap = train_dann(
        str(config_save_path),
        source_train_dir,
        source_val_dir,
        target_dir,
        str(output_path)
    )
    
    final_metrics = {
        'source_accuracy': source_metrics.get('accuracy', 0),
        'source_recall': source_metrics.get('recall', 0),
        'source_f1': source_metrics.get('f1', 0),
        'target_accuracy': target_metrics.get('accuracy', 0),
        'target_recall': target_metrics.get('recall', 0),
        'target_f1': target_metrics.get('f1', 0),
        'accuracy_gap': domain_gap.get('accuracy_drop', 0),
        'f1_gap': domain_gap.get('f1_drop', 0),
        'recall_gap': domain_gap.get('recall_drop', 0),
    }
    
    print("\n" + "="*70)
    print("BEST CONFIGURATION TRAINING COMPLETED")
    print("="*70)
    print(f"Final Target Metrics:")
    print(f"  Accuracy: {final_metrics['target_accuracy']:.4f}")
    print(f"  Recall: {final_metrics['target_recall']:.4f}")
    print(f"  F1-Score: {final_metrics['target_f1']:.4f}")
    print("="*70)
    
    return final_metrics


def main():
    parser = argparse.ArgumentParser(
        description='DANN Hyperparameter Tuning with Random Search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Example usage:
  python hyperparameter_tuning_dann.py \\
    --config experiments/domain_shift_eval.yaml \\
    --source-train-dir data/processed/domain_adaptation/source_synthetic/train \\
    --source-val-dir data/processed/domain_adaptation/source_synthetic/val \\
    --target-dir data/processed/domain_adaptation/target_real/test \\
    --output-dir results/dann_tuning

This script performs random search over DANN hyperparameters,
identifies the best configuration, and trains a final model.
        '''
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to base configuration YAML file'
    )
    parser.add_argument(
        '--source-train-dir',
        type=str,
        default='data/processed/domain_adaptation/source_synthetic/train',
        help='Source training data directory'
    )
    parser.add_argument(
        '--source-val-dir',
        type=str,
        default='data/processed/domain_adaptation/source_synthetic/val',
        help='Source validation data directory'
    )
    parser.add_argument(
        '--target-dir',
        type=str,
        default='data/processed/domain_adaptation/target_real/test',
        help='Target domain data directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/dann_tuning',
        help='Base output directory for tuning runs'
    )
    
    args = parser.parse_args()
    
    # Run hyperparameter tuning
    best_config, results_df = run_hyperparameter_tuning(
        args.config,
        args.source_train_dir,
        args.source_val_dir,
        args.target_dir,
        args.output_dir
    )
    
    if best_config is None:
        print("Hyperparameter tuning failed. Exiting.")
        return
    
    # Train with best configuration
    best_output_dir = os.path.join(args.output_dir, 'best_model')
    final_metrics = train_best_config(
        args.config,
        best_config,
        args.source_train_dir,
        args.source_val_dir,
        args.target_dir,
        best_output_dir
    )
    
    # Save final summary
    summary = {
        'best_hyperparameters': best_config.to_dict(),
        'final_metrics': final_metrics
    }
    
    summary_path = Path(args.output_dir) / 'tuning_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f)
    
    print(f"\nFinal summary saved to: {summary_path}")
    print("\n" + "="*70)
    print("DANN HYPERPARAMETER TUNING PIPELINE COMPLETED")
    print("="*70)


if __name__ == '__main__':
    main()
