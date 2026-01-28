"""
Minimal script to re-evaluate DANN model using DomainShiftEvaluator with corrected threshold logic.
This will regenerate all plots and metrics in the specified output directory.

Usage:
    python training/re_evaluate_domain_shift.py \
        --model-path results/domain_shift/ALEXNET/best_dann_model.pth \
        --source-dir data/processed/domain_adaptation/source_synthetic/train \
        --target-dir data/processed/domain_adaptation/target_real/test \
        --output-dir results/domain_shift/ALEXNET/re_eval
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.domain_adaptation.dann import DomainAdversarialNN
from evaluation.domain_shift_evaluation import DomainShiftEvaluator


def main():
    parser = argparse.ArgumentParser(description='Re-evaluate DANN model with DomainShiftEvaluator')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--source-dir', type=str, required=True, help='Source domain directory')
    parser.add_argument('--target-dir', type=str, required=True, help='Target domain directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for new plots/metrics')
    parser.add_argument('--feature-dim', type=int, default=512, help='Feature dimension (default: 512)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = DomainAdversarialNN(feature_dim=args.feature_dim, num_classes=2).to(device)
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model.eval()

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Data loaders
    source_dataset = datasets.ImageFolder(args.source_dir, transform=transform)
    target_dataset = datasets.ImageFolder(args.target_dir, transform=transform)
    source_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Criterion (for loss, not used in threshold selection)
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()

    # Evaluate
    evaluator = DomainShiftEvaluator(model, device, output_dir=args.output_dir)
    evaluator.evaluate_domain_shift(source_loader, target_loader, criterion)
    print(f"\n✓ Re-evaluation complete. Results saved to: {args.output_dir}\n")

if __name__ == '__main__':
    main()
