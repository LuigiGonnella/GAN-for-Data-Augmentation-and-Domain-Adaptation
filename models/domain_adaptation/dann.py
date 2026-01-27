import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function


class FeatureExtractor(nn.Module):
    
    def __init__(self, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Simple feature extraction network
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ClassificationHead(nn.Module):
    """Classification head for predicting class labels"""
    
    def __init__(self, feature_dim=512, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class DomainDiscriminator(nn.Module):
    """Domain discriminator for adversarial domain adaptation"""
    
    def __init__(self, feature_dim=512):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(128, 1)  # No sigmoid - use BCEWithLogitsLoss for stability
        )
    
    def forward(self, x):
        return self.discriminator(x)


class DomainAdversarialNN(nn.Module):
    
    def __init__(self, feature_dim=512, num_classes=2):
        super().__init__()
        self.feature_extractor = FeatureExtractor(feature_dim=feature_dim)
        self.classifier = ClassificationHead(feature_dim=feature_dim, num_classes=num_classes)
        self.domain_discriminator = DomainDiscriminator(feature_dim=feature_dim)
    
    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        class_logits = self.classifier(features)
        
        if return_features:
            return class_logits, features
        return class_logits
    
    def get_features(self, x):
        return self.feature_extractor(x)
    
    def predict_domain(self, features):
        return self.domain_discriminator(features)
    
    def compute_loss(self, class_logits, domain_logits, class_targets, domain_targets, 
                     lambda_d=0.1, num_classes=2):
        """
            
        Returns:
            total_loss: Sum of classification and domain losses
            class_loss: Classification loss
            domain_loss: Domain adversarial loss
        """
        
        class_loss = nn.CrossEntropyLoss()(class_logits, class_targets)
        
        domain_loss = nn.BCELoss()(domain_logits.squeeze(), domain_targets.float())
        
        total_loss = class_loss + lambda_d * domain_loss
        
        return total_loss, class_loss, domain_loss


class GradientReversalFunction(Function):
    """Gradient Reversal Layer for adversarial training.
    
    Forward pass: identity (output = input)
    Backward pass: reverses gradients (grad_input = -lambda * grad_output)
    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Wrapper module for gradient reversal."""
    
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DANNTrainer:
    """Trainer for Domain Adversarial Neural Networks with gradient reversal."""
    
    def __init__(self, model, device, output_dir='results/dann'):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.grl = GradientReversalLayer()
    
    @staticmethod
    def compute_lambda_adaptation(epoch, max_epochs, gamma=10.0):
        """Compute adaptive lambda following DANN paper schedule.
        
        lambda_p = 2 / (1 + exp(-gamma * p)) - 1
        where p = epoch / max_epochs ranges from 0 to 1
        """
        p = epoch / max_epochs
        lambda_p = 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0
        return lambda_p
    
    def train_epoch(self, source_loader, target_loader, optimizer, 
                    criterion_class, criterion_domain, lambda_adapt):
        """Train one epoch with gradient reversal.
        
        Args:
            source_loader: DataLoader for source domain (labeled)
            target_loader: DataLoader for target domain (unlabeled)
            optimizer: Optimizer for all model parameters
            criterion_class: Loss function for classification
            criterion_domain: Loss function for domain discrimination (use BCEWithLogitsLoss)
            lambda_adapt: Adaptive weight for domain loss (applied via GRL)
            
        Returns:
            avg_total_loss: Average total loss
            avg_class_loss: Average classification loss
            avg_domain_loss: Average domain discrimination loss
        """
        self.model.train()
        
        total_loss = 0
        total_class_loss = 0
        total_domain_loss = 0
        total_domain_correct = 0
        total_domain_samples = 0
        num_batches = 0
        
        # Update GRL lambda - this handles the λ weighting
        self.grl.lambda_ = lambda_adapt
        
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        num_iters = min(len(source_loader), len(target_loader))
        
        for _ in range(num_iters):
            # Get source and target batches
            try:
                source_img, source_label = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_img, source_label = next(source_iter)
            
            try:
                target_img, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_img, _ = next(target_iter)
            
            source_img = source_img.to(self.device)
            source_label = source_label.to(self.device)
            target_img = target_img.to(self.device)
            
            # Extract features
            source_features = self.model.feature_extractor(source_img)
            target_features = self.model.feature_extractor(target_img)
            
            # Classification loss (only on source domain)
            source_class_output = self.model.classifier(source_features)
            class_loss = criterion_class(source_class_output, source_label)
            
            # Domain discrimination with gradient reversal
            source_features_reversed = self.grl(source_features)
            target_features_reversed = self.grl(target_features)
            
            source_domain_output = self.model.domain_discriminator(source_features_reversed)
            target_domain_output = self.model.domain_discriminator(target_features_reversed)
            
            # Domain labels: 0 = source, 1 = target
            source_domain_labels = torch.zeros(source_img.size(0), 1).to(self.device)
            target_domain_labels = torch.ones(target_img.size(0), 1).to(self.device)
            
            domain_loss_source = criterion_domain(source_domain_output, source_domain_labels)
            domain_loss_target = criterion_domain(target_domain_output, target_domain_labels)
            domain_loss = domain_loss_source + domain_loss_target
            
            # Calculate domain accuracy for monitoring (optional debugging metric)
            # Goal: ~50% means perfect domain invariance (discriminator can't tell domains apart)
            with torch.no_grad():
                source_domain_preds = (torch.sigmoid(source_domain_output) < 0.5).float()
                target_domain_preds = (torch.sigmoid(target_domain_output) >= 0.5).float()
                domain_correct = source_domain_preds.sum() + target_domain_preds.sum()
                total_domain_correct += domain_correct.item()
                total_domain_samples += source_img.size(0) + target_img.size(0)
            
            # Total loss: class_loss + domain_loss
            # NOTE: λ weighting already applied by GRL, so we don't multiply domain_loss by lambda_adapt here
            # (Otherwise we'd get λ² scaling which is incorrect)
            total = class_loss + domain_loss
            
            # Backward and optimize
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            
            # Track losses
            total_loss += total.item()
            total_class_loss += class_loss.item()
            total_domain_loss += domain_loss.item()
            num_batches += 1
        
        avg_domain_acc = total_domain_correct / total_domain_samples if total_domain_samples > 0 else 0.0
        
        return (total_loss / num_batches, 
                total_class_loss / num_batches, 
                total_domain_loss / num_batches,
                avg_domain_acc)  # Add domain accuracy for monitoring
    
    @torch.no_grad()
    def evaluate(self, loader, criterion):
        """Evaluate model on a given loader with optimal threshold tuning.
        
        Args:
            loader: DataLoader for evaluation
            criterion: Loss function
            
        Returns:
            avg_loss: Average loss
            accuracy: Classification accuracy with optimal threshold
            recall: Recall/sensitivity with optimal threshold
        """
        from sklearn.metrics import precision_recall_curve, accuracy_score, recall_score
        
        self.model.eval()
        
        total_loss = 0
        all_probs = []
        all_targets = []
        num_samples = 0
        
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(images)
            loss = criterion(logits, labels)
            
            # Get probability for class 1 (malignant)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            total_loss += loss.item() * images.shape[0]
            num_samples += images.shape[0]
        
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        avg_loss = total_loss / num_samples
        
        # Find optimal threshold that maximizes F1 
        precision_vals, recall_vals, thresholds_pr = precision_recall_curve(all_targets, all_probs)
        
        # Avoid division by zero with corrected logic
        f1_scores = np.zeros(len(precision_vals))
        valid_idx = (precision_vals + recall_vals) > 0
        f1_scores[valid_idx] = 2 * (precision_vals[valid_idx] * recall_vals[valid_idx]) / (precision_vals[valid_idx] + recall_vals[valid_idx])
        
        optimal_idx = np.argmax(f1_scores)
        if optimal_idx < len(thresholds_pr):
            optimal_threshold = thresholds_pr[optimal_idx]
        else:
            optimal_threshold = 0.5
        
        # Ensure threshold is in valid range [0, 1]
        optimal_threshold = np.clip(optimal_threshold, 0.0, 1.0)
        
        # Generate predictions with optimal threshold
        optimal_preds = (all_probs >= optimal_threshold).astype(int)
        accuracy = accuracy_score(all_targets, optimal_preds)
        recall = recall_score(all_targets, optimal_preds, zero_division=0)
        
        return avg_loss, accuracy, recall
