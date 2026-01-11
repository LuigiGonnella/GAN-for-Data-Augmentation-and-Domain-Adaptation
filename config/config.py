from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

class Config():
    # Data paths
    RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'images'
    RAW_METADATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'metadata.csv' 
    BASELINE_PATH =  PROJECT_ROOT / 'data' / 'processed' / 'baseline' 
    AUGMENTED_PATH =  PROJECT_ROOT / 'data' / 'processed' / 'augmented' 


config = Config()