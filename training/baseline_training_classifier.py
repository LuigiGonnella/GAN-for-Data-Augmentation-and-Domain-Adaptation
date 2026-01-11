import sys
import torch
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from train_classifier import main as train_main

def run_baseline():

    print("BASELINE DATASET TRAINING - all layer freezed except for FCL head")

    parser = argparse.ArgumentParser(
        description='BASELINE DATASET TRAINING - all layer freezed except for FCL head'
    )

    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to config YAML file')
    
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        
        metrics = train_main(config)
        
        print("BASELINE COMPLETED")
        
        return metrics
    
    except Exception as e:
        print(f'Error with file name experiments/classifier_baseline_freeze.yaml: {e}')
        return None

if __name__ == '__main__':
    result = run_baseline()
    print(result)
