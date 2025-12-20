import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from train_classifier import main as train_main

def run_baseline():

    print("BASELINE TRAINING - all layer freezed except for FCL head")
    
    try:
        with open('experiments/baseline_freeze.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        
        metrics = train_main(config)
        
        print("BASELINE COMPLETED")
        
        return metrics
    
    except Exception as e:
        print(f'Error with file name experiments/baseline_freeze.yaml: {e}')
        return None

if __name__ == '__main__':
    result = run_baseline()
    print(result)
