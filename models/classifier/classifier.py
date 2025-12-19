import torch
import torch.nn as nn
import torchvision.models as models

class Classifier(nn.Module):
    def __init__(self, num_classes=2, model_name='resnet50', pretrained=True):
        super(Classifier, self).__init__()
        if model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        elif model_name == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError(f"Model {model_name} not supported.")
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
    def freeze_layers_except_last(self):
        for name, param in self.model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
    
    def unfreeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = True
    
    def freeze_up_to_layer(self, layer_name):
        layers_to_freeze = []
        for name, param in self.model.named_parameters():
            if layer_name in name:
                break
            layers_to_freeze.append(param)
        for param in layers_to_freeze:
            param.requires_grad = False

    def freeze_specific_layers(self, layer_names):
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False