import sys
import torch
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from train_classifier import main as train_main

def run_baseline(config_path):

    print("BASELINE MODEL (freezing for resnet) TRAINING")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        
        metrics = train_main(config)
        
        print("✓ TRAINING COMPLETED")
        
        return metrics
    
    except Exception as e:
        print(f'Error with file name {config_path}: {e}')
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='BASELINE DATASET TRAINING'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to config YAML file'
    )
    args = parser.parse_args()
    
    result = run_baseline(args.config)
    print(result)
