import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import yaml
import os
import argparse
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.classifier.classifier import Classifier
from evaluation.domain_shift_evaluation import DomainShiftEvaluator, create_dataloaders
from evaluation.classifier_metrics import evaluate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_classifier_with_domain_shift_eval(
    config_path,
    source_dir='data/processed/domain_adaptation/source_synthetic/train',
    target_dir='data/processed/domain_adaptation/target_real/test',
    output_dir='results/domain_shift'
):
   
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    logger.info(f"Loading source domain from {source_dir}")
    source_dataset = datasets.ImageFolder(source_dir, transform=transform)
    source_loader = DataLoader(
        source_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    logger.info(f"Loading target domain from {target_dir}")
    target_dataset = datasets.ImageFolder(target_dir, transform=transform)
    target_loader = DataLoader(
        target_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    logger.info(f"Source domain: {len(source_dataset)} samples")
    logger.info(f"Target domain: {len(target_dataset)} samples")
    
    model = Classifier(
        num_classes=config['model']['num_classes'],
        model_name=config['model']['architecture'],
        pretrained=config['model']['pretrained']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-5)
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training'].get('scheduler_step', 10),
        gamma=config['training'].get('scheduler_gamma', 0.1)
    )
    
    logger.info("Starting training on source domain...")
    num_epochs = config['training']['num_epochs']
    
    best_val_acc = 0
    patience = config['training'].get('early_stopping_patience', 3)
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        
        for images, labels in source_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in source_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        avg_val_loss = val_loss / len(source_loader)
        val_acc = val_correct / val_total
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}] - "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {avg_val_loss:.4f}, "
                       f"Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), Path(output_dir) / 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(torch.load(Path(output_dir) / 'best_model.pth'))
    
    logger.info("\nTraining completed. Starting domain shift evaluation...")
    
    evaluator = DomainShiftEvaluator(model, device, output_dir=output_dir)
    source_metrics, target_metrics, domain_gap = evaluator.evaluate_domain_shift(
        source_loader, target_loader, criterion
    )
    
    return model, source_metrics, target_metrics, domain_gap


def main():
    parser = argparse.ArgumentParser(
        description='Train classifier and evaluate domain shift'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--source-dir',
        type=str,
        default='data/processed/domain_adaptation/source_synthetic/train',
        help='Path to source domain data'
    )
    parser.add_argument(
        '--target-dir',
        type=str,
        default='data/processed/domain_adaptation/target_real/test',
        help='Path to target domain data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/domain_shift',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    train_classifier_with_domain_shift_eval(
        args.config,
        args.source_dir,
        args.target_dir,
        args.output_dir
    )


if __name__ == '__main__':
    main()
