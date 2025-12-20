import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from train_classifier import main as train_main
import pandas as pd

def run_fine_tuning():

    print("FINETUNING TRAINING - earlier layers freezed and increasing lr")


    try:
        with open('experiments/baseline_ft.yaml', 'r') as f:
            config = yaml.safe_load(f)
            metrics = train_main(config)

        print("FINETUNING COMPLETED")

        return metrics
    
    except Exception as e:
        print(f'Error with file name experiments/baseline_ft.yaml: {e}')
        return None
   


#to run only finetuning 
if __name__ == '__main__':
    result = run_fine_tuning()
    print(result)


