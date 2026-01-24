import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from training.freeze_classifier import run_baseline
from training.finetune_classifier import run_fine_tuning
from training.pipeline_classifier import generate_final_report
from train_classifier import main as train_main, DatasetCSV, test_model
from models.classifier.classifier import Classifier
from evaluation.classifier_metrics import evaluate_with_threshold_tuning
from evaluation.plots import plot_cm

def evaluate_existing_model(model_path, config):
    """
    Load existing trained model and run evaluation with plots (no training)
    """
    print(f"Loading existing model from: {model_path}")
    
    # Setup transforms
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load validation dataset
    data_path = config.get('data_path', 'data/processed/baseline')
    val_csv = os.path.join(data_path, 'val', 'val.csv')
    val_dataset = DatasetCSV(os.path.join(data_path, 'val'), val_csv, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['params']['batch_size'], shuffle=False, num_workers=0)
    
    # Load model
    model = Classifier(num_classes=2, model_name=config['model']['name'], pretrained=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    
    criterion = nn.CrossEntropyLoss()
    output_dir = config.get('output_dir', 'results/baseline')
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # Evaluate with threshold tuning on validation set
    print("Running evaluation with threshold tuning on validation set...")
    val_loss_final, val_accuracy, val_f1, val_recall, val_precision, val_roc_auc, val_cm, optimal_threshold = evaluate_with_threshold_tuning(
        model, val_loader, criterion, device, plot_dir
    )
    print(f'\nValidation with Optimal Threshold: Loss: {val_loss_final:.4f}, Accuracy: {val_accuracy:.4f}, Recall: {val_recall:.4f}, Precision: {val_precision:.4f}, F1: {val_f1:.4f}, ROC-AUC: {val_roc_auc:.4f}')
    print(f'Validation Confusion Matrix:\n{val_cm}\n')
    
    # Test with optimal threshold
    print("Testing with optimal threshold...")
    test_loss, test_accuracy, test_f1, test_recall, test_precision, test_roc_auc, test_cm = test_model(
        model, config, device, optimal_threshold
    )
    
    # Plot confusion matrix
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

def run_custom_pipeline():
    """
    Custom pipeline to:
    1. Run freeze classifier (baseline) to get updated metrics with plots
    2. Use predefined fine-tuning metrics
    3. Run fine-tuning with specific hyperparameters
    4. Generate final reports
    """
    
    print("="*80)
    print("STEP 1: Evaluating Existing Baseline (Freeze) Model")
    print("="*80)
    
    # Load config and evaluate existing model
    baseline_config_path = "experiments/classifier_baseline_freeze.yaml"
    with open(baseline_config_path, 'r') as f:
        baseline_config = yaml.safe_load(f)
    
    baseline_model_path = "results/classifier_on_baseline/freeze/classifier.pth"
    baseline_metrics = evaluate_existing_model(baseline_model_path, baseline_config)
    
    print(f"\n✓ Baseline evaluation completed.")
    print(f"Baseline Results: {baseline_metrics}")
    
    print("\n" + "="*80)
    print("STEP 2: Using Provided Fine-Tuning Metrics")
    print("="*80)
    
    # Provided fine-tuning metrics from user
    finetune_results = {
        'accuracy': 0.8592156862745098, 
        'recall': 0.83, 
        'precision': 0.44703770197486536, 
        'f1': 0.5810968494749125, 
        'roc_auc': 0.9327051851851852, 
        'val_loss': 0.2298613992625592, 
        'optimal_threshold': 0.24532002
    }
    
    print(f"Fine-tuning Results: {finetune_results}")
    
    print("\n" + "="*80)
    print("STEP 3: Running Fine-Tuning with Custom Configuration")
    print("="*80)
    
    # Create custom config for fine-tuning with specific hyperparameters
    custom_config = {
        'experiment_name': 'baseline_ft_ht',
        'data_path': 'data/processed/baseline',
        'model': {
            'name': 'resnet50',
            'pretrained': True
        },
        'training': {
            'ht': True,  # Using hyperparameter tuning format
            'layers': 2,
            'params': {
                'epochs': 10,
                'batch_size': 64,
                'lr': 0.0001,
                'momentum': 0.9,
                'optimizer': 'Adam',
                'weight_decay': 1e-05
            }
        },
        'best_config_run': True,  # Enable plotting
        'output_dir': 'results/classifier_on_baseline/ft_ht'
    }
    
    print(f"Training with configuration:")
    print(f"  - Learning Rate: {custom_config['training']['params']['lr']}")
    print(f"  - Batch Size: {custom_config['training']['params']['batch_size']}")
    print(f"  - Weight Decay: {custom_config['training']['params']['weight_decay']}")
    print(f"  - Momentum: {custom_config['training']['params']['momentum']}")
    print(f"  - Optimizer: {custom_config['training']['params']['optimizer']}")
    
    # Run training with custom config
    tuning_results = train_main(custom_config)
    
    print(f"\n✓ Fine-tuning with custom hyperparameters completed.")
    print(f"Tuning Results: {tuning_results}")
    
    # Prepare best_config for report
    best_config = {
        'learning_rate': custom_config['training']['params']['lr'],
        'batch_size': custom_config['training']['params']['batch_size'],
        'weight_decay': custom_config['training']['params']['weight_decay'],
        'momentum': custom_config['training']['params']['momentum'],
        'optimizer': custom_config['training']['params']['optimizer'],
        'accuracy': tuning_results['accuracy'],
        'precision': tuning_results['precision'],
        'recall': tuning_results['recall'],
        'f1': tuning_results['f1'],
        'roc_auc': tuning_results['roc_auc'],
        'val_loss': tuning_results['val_loss']
    }
    
    print("\n" + "="*80)
    print("STEP 4: Generating Final Reports")
    print("="*80)
    
    # Generate final report
    data_type = 'baseline'
    generate_final_report(baseline_metrics, finetune_results, tuning_results, best_config, data_type)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print("\nFinal Summary:")
    print(f"Baseline Metrics: {baseline_metrics}")
    print(f"Fine-tuning Metrics: {finetune_results}")
    print(f"Tuning with Custom Config Metrics: {tuning_results}")
    print(f"\nReports saved in: results/classifier_on_baseline/final_report/")

if __name__ == '__main__':
    run_custom_pipeline()
