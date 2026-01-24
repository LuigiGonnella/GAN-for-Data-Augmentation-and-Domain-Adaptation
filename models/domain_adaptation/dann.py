import torch
import torch.nn as nn
import torch.nn.functional as F


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
            
            nn.Linear(128, 1),
            nn.Sigmoid()
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


class DANNTrainer:
    
    def __init__(self, model, device, output_dir='results/dann'):
        self.model = model
        self.device = device
        self.output_dir = output_dir
    
    def train_step(self, source_loader, target_loader, optimizer, criterion, 
                   lambda_d=0.1, num_classes=2):
        
        self.model.train()
        
        total_loss = 0
        total_class_loss = 0
        total_domain_loss = 0
        num_batches = 0
        
        for (source_img, source_label), (target_img, target_label) in zip(source_loader, target_loader):
            
            source_img = source_img.to(self.device)
            source_label = source_label.to(self.device)
            target_img = target_img.to(self.device)
            
            source_features = self.model.get_features(source_img)
            source_class_logits = self.model.classifier(source_features)
            source_domain_logits = self.model.domain_discriminator(source_features)
            
            target_features = self.model.get_features(target_img)
            target_domain_logits = self.model.domain_discriminator(target_features)
            
            source_domain_labels = torch.zeros(source_img.size(0), device=self.device)
            target_domain_labels = torch.ones(target_img.size(0), device=self.device)
            
            domain_logits = torch.cat([source_domain_logits, target_domain_logits], dim=0)
            domain_labels = torch.cat([source_domain_labels, target_domain_labels], dim=0)
            
            class_loss = nn.CrossEntropyLoss()(source_class_logits, source_label)
            domain_loss = nn.BCELoss()(domain_logits.squeeze(), domain_labels)
            total = class_loss + lambda_d * domain_loss
            
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            
            total_loss += total.item()
            total_class_loss += class_loss.item()
            total_domain_loss += domain_loss.item()
            num_batches += 1
        
        return (total_loss / num_batches, 
                total_class_loss / num_batches, 
                total_domain_loss / num_batches)
    
    @torch.no_grad()
    def evaluate(self, loader, criterion):
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
