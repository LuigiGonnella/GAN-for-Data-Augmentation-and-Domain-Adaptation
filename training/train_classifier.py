import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import yaml
import os
import pandas as pd
from PIL import Image
from pathlib import Path
from models.classifier.classifier import Classifier
from evaluation.metrics import evaluate


class DatasetCSV(Dataset):
    """Dataset that loads images and labels from CSV metadata"""
    def __init__(self, img_dir, csv_path, transform=None, has_subdirs=False):
        self.img_dir = img_dir
        self.transform = transform
        self.metadata = pd.read_csv(csv_path)
        self.label_map = {'Benign': 0, 'benign': 0, 'Malignant': 1, 'malignant': 1}
        self.has_subdirs = has_subdirs  # True if images are in benign/malignant subdirs
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name = row['img_name'] + '.jpg'
        label = self.label_map[row['target']]
        
    
        if self.has_subdirs:
            subdir = 'benign' if label == 0 else 'malignant'
            img_path = os.path.join(self.img_dir, subdir, img_name)
        else:
            img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def test_model(model, config, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_path = config.get('data_path', 'data/processed/baseline')
    test_csv = os.path.join(data_path, 'test', 'test.csv')
    test_dataset = DatasetCSV(os.path.join(data_path, 'test'), test_csv, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0) 

    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, test_f1, test_roc_auc, test_cm = evaluate(model, test_loader, criterion, device)
    return test_loss, test_accuracy, test_f1, test_roc_auc, test_cm




def main(config=None):
    # Load config
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    elif config is None:
        with open('experiments/baseline.yaml', 'r') as f:
            config = yaml.safe_load(f)

    freezing_strategy = config.get('freezing_strategy', 'freeze_except_last')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_path = config.get('data_path', 'data/processed/baseline')
    
    train_csv = os.path.join(data_path, 'train', 'train.csv')
    train_dataset = DatasetCSV(os.path.join(data_path, 'train'), train_csv, transform=transform, has_subdirs=True)
    
    val_csv = os.path.join(data_path, 'val', 'val.csv')
    val_dataset = DatasetCSV(os.path.join(data_path, 'val'), val_csv, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=0)

    model = Classifier(num_classes=2, model_name=config['model']['name'], pretrained=True)
    

    if freezing_strategy == 'no_freeze':
        model.unfreeze_all_layers()
    elif freezing_strategy == 'freeze_except_last':
        model.freeze_layers_except_last()
    elif freezing_strategy == 'freeze_last_2_blocks':
        model.freeze_up_to_layer(layer_num=2)
    elif freezing_strategy == 'progressive':
        model.freeze_layers_except_last()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['lr'], weight_decay=config.get('weight_decay', 1e-5))
    
    class_weight_ratio = config.get('class_weight_ratio', 1)
    if class_weight_ratio > 1:
        weights = torch.tensor([1.0, float(class_weight_ratio)], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0
    
    for epoch in range(config['training']['epochs']):
        print(f'\nStarting epoch {epoch+1}/{config["training"]["epochs"]}...')
        model.train()
        running_loss = 0.0
        batch_count = 0
        for images, labels in train_loader:
            batch_count += 1
            if batch_count == 1:
                print(f'First batch loaded successfully! Size: {images.shape}')
            if batch_count % 10 == 0:
                print(f'Processed {batch_count} batches...')
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss, accuracy, f1, roc_auc, cm = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{config["training"]["epochs"]}, Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}')
        
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            output_dir = config.get('output_dir', 'results/baseline')
            os.makedirs(output_dir, exist_ok=True)
            model_save_path = os.path.join(output_dir, 'classifier.pth')
            torch.save(model.state_dict(), model_save_path)
    
    print(f'Training completed. Best val accuracy: {best_val_accuracy:.4f}')
    
    output_dir = config.get('output_dir', 'results/baseline')
    model_save_path = os.path.join(output_dir, 'classifier.pth')
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_accuracy, test_f1, test_roc_auc, test_cm = test_model(model, config, device)
    print(f'Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}, ROC-AUC: {test_roc_auc:.4f}')
    print(f'Confusion Matrix:\n{test_cm}')
    
    return {
        'accuracy': test_accuracy,
        'f1': test_f1,
        'roc_auc': test_roc_auc,
        'val_loss': test_loss,
        'best_val_accuracy': best_val_accuracy
    }

if __name__ == '__main__':
    main()