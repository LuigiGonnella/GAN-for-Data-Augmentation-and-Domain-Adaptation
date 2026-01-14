import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import argparse

import yaml
from train_classifier import main as train_main
import pandas as pd

def run_fine_tuning(config_path):

    print("FINETUNING TRAINING - earlier layers freezed and increasing lr")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        metrics = train_main(config)

        print("FINETUNING COMPLETED")

        return metrics
    
    except Exception as e:
        print(f'Error with file name experiments/classifier_baseline_ft.yaml: {e}')
        return None
   


#to run only finetuning 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='BASELINE DATASET TRAINING - FINETUNING TRAINING - earlier layers freezed and increasing lr'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to config YAML file'
    )
    args = parser.parse_args()
    
    result = run_fine_tuning(args.config)
    print(result)


