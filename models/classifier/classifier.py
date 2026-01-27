import torch
import torch.nn as nn
import torchvision.models as models

class Classifier(nn.Module):
    def __init__(self, num_classes=2, model_name='resnet50', pretrained=True):
        super(Classifier, self).__init__()
        self.model_name = model_name
        
        if model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        elif model_name == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif model_name == 'alexnet':
            self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError(f"Model {model_name} not supported.")
        
        # Adjust final layer based on architecture
        if 'resnet' in model_name:
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        elif model_name == 'alexnet':
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
    def freeze_layers_except_last(self):
        """Freeze all layers except the final classification layer."""
        for name, param in self.model.named_parameters():
            # ResNet: freeze everything except 'fc'
            # AlexNet: freeze everything except 'classifier.6'
            if 'resnet' in self.model_name:
                if "fc" not in name:
                    param.requires_grad = False
            elif self.model_name == 'alexnet':
                if "classifier.6" not in name:
                    param.requires_grad = False
    
    def unfreeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = True
    
    def freeze_up_to_layer(self, layer_num):
        """
        Freeze the first layer_num convolutional layer groups.
        
        For ResNet: layer_num=1 freezes conv1+bn1+layer1, layer_num=2 freezes up to layer2, etc.
        For AlexNet: layer_num freezes first N layers of features module (0-12 layers available).
        """
        if 'resnet' in self.model_name:
            freeze_list = ['conv1', 'bn1']  # Always freeze initial conv and bn
            for i in range(1, layer_num + 1):
                freeze_list.append(f'layer{i}')
            
            for name, param in self.model.named_parameters():
                if any(layer in name for layer in freeze_list):
                    param.requires_grad = False
                    
        elif self.model_name == 'alexnet':
            # AlexNet features: 0-12 (conv layers, relu, pooling)
            # Freeze first layer_num layers in features module
            for name, param in self.model.named_parameters():
                if 'features' in name:
                    # Extract layer index from name like 'features.0.weight'
                    try:
                        layer_idx = int(name.split('.')[1])
                        if layer_idx < layer_num:
                            param.requires_grad = False
                    except (IndexError, ValueError):
                        pass

    def freeze_specific_layers(self, layer_names):
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False