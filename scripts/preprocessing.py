import sys
import yaml
import os
import shutil
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from config.config import config



config_path = 'config/preprocessing.yaml'
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)



def load_and_preprocess_data(malignant_count, benign_count):
    df = pd.read_csv(config.RAW_METADATA_PATH)
    
    #mantain only interesting columns
    df = df[['isic_id', 'diagnosis_1']].rename(columns={'isic_id': 'img_name', 'diagnosis_1': 'target'})

    #mantain only benign and malign
    df = df[df['target'].isin(['Benign', 'Malignant'])]

    #target
    #Benign       15990
    #Malignant     8473

    #mantain only 1k malignant and 10k benign
    benign = df[df['target'] == 'Benign'].copy()
    benign = benign.iloc[:benign_count, :]

    malignant = df[df['target'] == 'Malignant'].copy()
    malignant = malignant.iloc[:malignant_count, :]

    df = pd.concat([benign, malignant], axis=0, ignore_index=True)
    
    return df



def copy_images(images, src_folder, dst_folder):
    #delete files from dst_folder
    for file in dst_folder.glob("*"):
        if file.is_file():
            file.unlink()
            
    for img in images:
        img_path = img + '.jpg'
        shutil.copy(os.path.join(src_folder, img_path), dst_folder)
    
    return len(images)


def populate_baseline_and_augmented(seed, df, val_ratio, test_ratio):

    #get test - (train + val) split
    X, x_test, Y, y_test = train_test_split(df['img_name'], df['target'], test_size=test_ratio, random_state=seed, stratify=df['target']) #mantain same imbalance
    
    #split (train + val) further to get train - val
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=val_ratio/(1-test_ratio), random_state=seed, stratify=Y) #mantain same imbalance

    #create new dataframes
    df_test = pd.DataFrame({'img_name': x_test, 'target': y_test})
    df_train = pd.DataFrame({'img_name': x_train, 'target': y_train})
    df_val = pd.DataFrame({'img_name': x_val, 'target': y_val})

    #obtain images as lists
    train_benign_images = list(df_train[df_train['target']=='Benign']['img_name'])
    train_malignant_images = list(df_train[df_train['target']=='Malignant']['img_name'])

    test_images = list(df_test['img_name'])

    val_images = list(df_val['img_name'])

    #obtain dir paths and copy images
    baseline_train_benign_path = os.path.join(config.BASELINE_PATH, 'train', 'benign')
    baseline_len_benign_train = copy_images(train_benign_images, config.RAW_DATA_PATH, baseline_train_benign_path)

    augmented_train_benign_path = os.path.join(config.AUGMENTED_PATH, 'train', 'benign')
    augmented_len_benign_train = copy_images(train_benign_images, config.RAW_DATA_PATH, augmented_train_benign_path)

    print(f'COPIED {baseline_len_benign_train} BENIGN IMAGES IN baseline/train/benign FOLDER')
    print(f'COPIED {augmented_len_benign_train} BENIGN IMAGES IN augmented/train/benign FOLDER')

    train_malignant_path = os.path.join(config.BASELINE_PATH, 'train', 'malignant')
    len_malignant_train = copy_images(train_malignant_images, config.RAW_DATA_PATH, train_malignant_path)
    
    print(f'COPIED {len_malignant_train} MALIGNANT IMAGES IN train/malignant FOLDER')

    baseline_test_path = os.path.join(config.BASELINE_PATH, 'test')
    baseline_len_test = copy_images(test_images, config.RAW_DATA_PATH, baseline_test_path)

    augmented_test_path = os.path.join(config.AUGMENTED_PATH, 'test')
    augmented_len_test = copy_images(test_images, config.RAW_DATA_PATH, augmented_test_path)

    print(f'COPIED {baseline_len_test} TEST IMAGES IN baseline/test FOLDER')
    print(f'COPIED {augmented_len_test} TEST IMAGES IN augmented/test FOLDER')

    
    baseline_val_path = os.path.join(config.BASELINE_PATH, 'val')
    baseline_len_val = copy_images(val_images, config.RAW_DATA_PATH, baseline_val_path)

    augmented_val_path = os.path.join(config.AUGMENTED_PATH, 'val')
    augmented_len_val = copy_images(val_images, config.RAW_DATA_PATH, augmented_val_path)

    print(f'COPIED {baseline_len_val} VAL IMAGES IN baseline/val FOLDER')
    print(f'COPIED {augmented_len_val} VAL IMAGES IN augmented/val FOLDER')

    return df_test, df_train, df_val




    



#load metadata
df = load_and_preprocess_data(cfg['dataset']['malignant_count'], cfg['dataset']['benign_count'])

#populate baseline/train, baseline/test and baseline/val
df_test, df_train, df_val = populate_baseline_and_augmented(cfg['dataset']['random_seed'], df, cfg['dataset']['val_ratio'], cfg['dataset']['test_ratio'])

#save metadata for classifier training
df_test.to_csv(config.BASELINE_PATH / 'test' / 'test.csv', index=False)
df_train.to_csv(config.BASELINE_PATH / 'train' / 'train.csv', index=False)
df_val.to_csv(config.BASELINE_PATH / 'val' / 'val.csv', index=False)


