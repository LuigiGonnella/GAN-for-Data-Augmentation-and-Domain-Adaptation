import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import yaml
import os
import pandas as pd
from PIL import Image
from pathlib import Path
from models.classifier.classifier import Classifier
from evaluation.classifier_metrics import evaluate, evaluate_with_threshold_tuning
from evaluation.plots import plot_train_val_stats, plot_cm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, auc, recall_score, precision_score



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

def test_model(model, config, device, optimal_threshold):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_path = config.get('data_path', 'data/processed/baseline')
    test_csv = os.path.join(data_path, 'test', 'test.csv')
    test_dataset = DatasetCSV(os.path.join(data_path, 'test'), test_csv, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['params']['batch_size'], shuffle=False, num_workers=0) 

    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy, test_f1, test_recall, test_precision, test_roc_auc, test_cm, _, _ = evaluate(
        model, test_loader, criterion, device, optimal_threshold
    )
    return test_loss, test_accuracy, test_f1, test_recall, test_precision, test_roc_auc, test_cm


def preprocess(config):
    # Load config
    if not isinstance(config, dict):
        return print('Please provide a correct training configuration')

    # Training transforms with data augmentation to prevent overfitting
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Validation/test transforms without augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_path = config.get('data_path', 'data/processed/baseline')
    
    train_csv = os.path.join(data_path, 'train', 'train.csv')
    train_dataset = DatasetCSV(os.path.join(data_path, 'train'), train_csv, transform=train_transform, has_subdirs=True)
    
    val_csv = os.path.join(data_path, 'val', 'val.csv')
    val_dataset = DatasetCSV(os.path.join(data_path, 'val'), val_csv, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['params']['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['params']['batch_size'], shuffle=False, num_workers=0)

    pretrained = True if config['model']['pretrained'] else False
    model = Classifier(num_classes=2, model_name=config['model']['name'], pretrained=pretrained)

    return train_loader, val_loader, model

def define_strategy(config, model):
    scheduler = None
    
    if config['training']['scratch']:
        lr = config['training']['params']['lr']

        if not config['training']['ht']:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config.get('weight_decay', 1e-5))

        elif (config['training']['params']['optimizer']=='SGD'):
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=config['training']['params']['momentum'], weight_decay=config['training']['params']['weight_decay'])

        elif (config['training']['params']['optimizer']=='Adam'):
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config['training']['params']['weight_decay'])

        elif (config['training']['params']['optimizer']=='AdamW'):
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=config['training']['params']['weight_decay'])
        
        elif (config['training']['params']['optimizer']=='RMSprop'):
            optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=config['training']['params']['momentum'], weight_decay=config['training']['params']['weight_decay'])
        
        # Add scheduler for AlexNet training from scratch
        if config['model']['name'] == 'alexnet':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5, cooldown=1)

    elif not config['training']['layers']: #total freeze except last fcl
            model.freeze_layers_except_last()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['training']['params']['lr'], weight_decay=config.get('weight_decay', 1e-5))
    
    elif not config['training']['ht']: #finetuning without hyperparameter tuning
        lr = config['training']['params']['lr']
        model.freeze_up_to_layer(layer_num=1)
        # Discriminative learning rates: lower for earlier layers, higher for later layers
        if 'resnet' in config['model']['name']:
            optimizer = optim.Adam([
                {'params': model.model.layer2.parameters(), 'lr': lr * 0.1},
                {'params': model.model.layer3.parameters(), 'lr': lr * 0.3},
                {'params': model.model.layer4.parameters(), 'lr': lr * 0.5},
                {'params': model.model.fc.parameters(), 'lr': lr}
            ], lr=lr, weight_decay=config.get('weight_decay', 1e-5))
        elif config['model']['name'] == 'alexnet':
            # AlexNet: progressive unfreezing with discriminative LR
            optimizer = optim.Adam([
                {'params': model.model.features[3:].parameters(), 'lr': lr * 0.3},
                {'params': model.model.classifier[:6].parameters(), 'lr': lr * 0.5},
                {'params': model.model.classifier[6].parameters(), 'lr': lr}
            ], lr=lr, weight_decay=config.get('weight_decay', 1e-5))

    else: #finetuning with hyperparameter tuning
        model.freeze_up_to_layer(layer_num=1)
        lr = config['training']['params']['lr']
        
        # Define parameter groups based on architecture
        if 'resnet' in config['model']['name']:
            param_groups = [
                {'params': model.model.layer2.parameters(), 'lr': lr * 0.1},
                {'params': model.model.layer3.parameters(), 'lr': lr * 0.3},
                {'params': model.model.layer4.parameters(), 'lr': lr * 0.5},
                {'params': model.model.fc.parameters(), 'lr': lr}
            ]
        elif config['model']['name'] == 'alexnet':
            param_groups = [
                {'params': model.model.features[3:].parameters(), 'lr': lr * 0.3},
                {'params': model.model.classifier[:6].parameters(), 'lr': lr * 0.5},
                {'params': model.model.classifier[6].parameters(), 'lr': lr}
            ]

        if (config['training']['params']['optimizer']=='SGD'):
            optimizer = optim.SGD(param_groups, lr=lr, momentum=config['training']['params']['momentum'], weight_decay=config['training']['params']['weight_decay'])

        elif (config['training']['params']['optimizer']=='Adam'):
            optimizer = optim.Adam(param_groups, lr=lr, weight_decay=config['training']['params']['weight_decay'])

        elif (config['training']['params']['optimizer']=='AdamW'):
            optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=config['training']['params']['weight_decay'])
        
        elif (config['training']['params']['optimizer']=='RMSprop'):
            optimizer = optim.RMSprop(param_groups, lr=lr, momentum=config['training']['params']['momentum'], weight_decay=config['training']['params']['weight_decay'])
    
    return optimizer, scheduler

def main(config=None):

    #---------- PREPROCESSING ----------
    
    train_loader, val_loader, model = preprocess(config)
    
    #---------- STRATEGY ----------

    optimizer, scheduler = define_strategy(config, model)

    #---------- TRAINING ---------- 
    #we use DEFAULT THRESHOLD 0.5 to classify images
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    criterion = nn.CrossEntropyLoss()

    best_recall = 0.0
    best_f1 = 0.0  

    patience = 3
    early_stopping_count = 0

    validation_losses_epochs = [] #contains losses after each batch
    train_losses_epochs = [] #same

    train_accuracy_epochs = []
    validation_accuracy_epochs = []
    
    for epoch in range(config['training']['params']['epochs']):
        print(f"\nStarting epoch {epoch+1}/{config['training']['params']['epochs']}...")
        train_corrects = 0
        num_samples = 0  # Initialize for this epoch
        
        model.train()
        running_loss = 0.0
        batch_count = 0
        all_preds = []
        all_labels = []
        all_probs = []
        for images, labels in train_loader:
            bs = images.shape[0]
            num_samples += bs
            batch_count += 1
            if batch_count == 1:
                print(f'First batch loaded successfully! Size: {images.shape}')
            if batch_count % 10 == 0:
                print(f'Processed {batch_count} batches...')
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            _, targets_pred = torch.max(outputs, 1)
            train_corrects += torch.sum(labels == targets_pred).item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * bs
            probs = torch.softmax(outputs, dim=1)[:, 1] 
            all_preds.extend(targets_pred.cpu().numpy())
            all_labels.extend(labels.cpu().detach().numpy())
            all_probs.extend(probs.cpu().detach().numpy())
        
        train_loss = running_loss/num_samples
        train_losses_epochs.append(train_loss)
        train_accuracy_epochs.append(1.0*train_corrects/float(num_samples))

        train_accuracy = accuracy_score(all_labels, all_preds)
        train_f1 = f1_score(all_labels, all_preds, zero_division=0)
        train_recall = recall_score(all_labels, all_preds, zero_division=0)
        train_precision = precision_score(all_labels, all_preds, zero_division=0)
        train_roc_auc = roc_auc_score(all_labels, all_probs)

        
        #---------- VALIDATION ----------
        #we use DEFAULT THRESHOLD 0.5 to classify images

        model.eval()
        val_loss, accuracy, f1, recall, precision, roc_auc, _, num_samples, val_corrects = evaluate(model, val_loader, criterion, device, optimal_threshold=False)
        val_acc = 1.0*val_corrects/float(num_samples)
        validation_accuracy_epochs.append(val_acc)
        validation_losses_epochs.append(val_loss)

        print(f"Epoch {epoch+1}/{config['training']['params']['epochs']}, Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Recall: {train_recall:.4f}, Precision: {train_precision:.4f}, F1: {train_f1:.4f}, ROC-AUC: {train_roc_auc:.4f}")
        print(f"Epoch {epoch+1}/{config['training']['params']['epochs']}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        # Step scheduler if it exists (AlexNet training from scratch)
        if scheduler is not None:
            scheduler.step(val_loss)
        
        if recall > best_recall:
            early_stopping_count = 0
            best_recall = recall
            best_f1 = f1
            output_dir = config.get('output_dir', 'results/baseline')
            os.makedirs(output_dir, exist_ok=True)
            model_save_path = os.path.join(output_dir, 'classifier.pth')
            torch.save(model.state_dict(), model_save_path)
        else:
            early_stopping_count+=1
            
        if early_stopping_count>=patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    
    print(f'Training completed. \nBest recall: {best_recall:.4f}\nBest f1-score: {best_f1:.4f}')
    
    output_dir = config.get('output_dir', 'results/baseline')
    model_save_path = os.path.join(output_dir, 'classifier.pth')

    optimal_threshold = False #initialize for non-best_config runs in hyperparameter tuning 
    
    #---------- PLOTS ----------
    if (config['training']['ht'] and config['best_config_run']) or (not config['training']['ht']): #plots only if its not hyperparameter tuning or if it's the final run of best configuration of hyperparameter tuning
        plot_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        
        plot_train_val_stats(plot_dir, train_losses_epochs, validation_losses_epochs, train_accuracy_epochs, validation_accuracy_epochs)

        #PLOTTING ROC-CURVE and RECALL-PRECISION on VALIDATION SET to compute OPTIMAL THRESHOLD (max F1)
        val_loss_final, val_accuracy, val_f1, val_recall, val_precision, val_roc_auc, val_cm, optimal_threshold = evaluate_with_threshold_tuning(model, val_loader, criterion, device, plot_dir)
        print(f'\nValidation with Optimal Threshold: Loss: {val_loss_final:.4f}, Accuracy: {val_accuracy:.4f}, Recall: {val_recall:.4f}, Precision: {val_precision:.4f}, F1: {val_f1:.4f}, ROC-AUC: {val_roc_auc:.4f}')
        print(f'Validation Confusion Matrix:\n{val_cm}\n')

    #---------- TESTING USING OPTIMAL THRESHOLD ----------
    model.load_state_dict(torch.load(model_save_path))
    test_loss, test_accuracy, test_f1, test_recall, test_precision, test_roc_auc, test_cm = test_model(model, config, device, optimal_threshold)

    #PLOT CONFUSION MATRIX
    if (config['training']['ht'] and config['best_config_run']) or (not config['training']['ht']): #plots only if its not hyperparameter tuning or if it's the final run of best configuration of hyperparameter tuning
        plot_cm(plot_dir, test_cm)
        print(f'Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Recall: {test_recall:.4f}, Precision: {test_precision:.4f}, F1: {test_f1:.4f}, ROC-AUC: {test_roc_auc:.4f}')
        print(f'Optimal Threshold: {optimal_threshold:.3f}')
        print(f'Confusion Matrix:\n{test_cm}')
    
    return {
        'accuracy': test_accuracy,
        'recall': test_recall,
        'precision': test_precision,
        'f1': test_f1,
        'roc_auc': test_roc_auc,
        'val_loss': test_loss,
        'optimal_threshold': optimal_threshold
    }

if __name__ == '__main__':
    main()