import torch
import torch.nn as nn


class ConditionalDCGANGenerator(nn.Module):
   
    def __init__(self, input_dim=100, num_classes=2, n1=512, channels=3, width=128, height=128):
        super(ConditionalDCGANGenerator, self).__init__()
        
        self.num_classes = num_classes
        self.n1 = n1
        self.input_dim = input_dim
        
        self.class_embedding = nn.Embedding(num_classes, 50)
        
        self.fc = nn.Linear(input_dim + 50, n1 * 4 * 4)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(n1),
            
            nn.ConvTranspose2d(n1, n1 // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n1 // 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(n1 // 2, n1 // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n1 // 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(n1 // 4, n1 // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n1 // 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(n1 // 8, n1 // 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n1 // 16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(n1 // 16, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        class_embedding = self.class_embedding(labels)  
        
        z_concat = torch.cat([z, class_embedding], dim=1)  
        
        x = self.fc(z_concat) 
        x = x.view(x.size(0), self.n1, 4, 4) 
        x = self.conv_blocks(x)  
        return x


class ConditionalDCGANDiscriminator(nn.Module):

    def __init__(self, num_classes=2, channels=3, width=128, height=128, dropout=0.5):
        super(ConditionalDCGANDiscriminator, self).__init__()
        
        self.num_classes = num_classes
        
        self.class_embedding = nn.Embedding(num_classes, 512)
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4 + 512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )
    
    def forward(self, x, labels):

        features = self.conv_layers(x) 
        features = features.view(features.size(0), -1)  
        
        class_embedding = self.class_embedding(labels)  
        
        combined = torch.cat([features, class_embedding], dim=1)
        output = self.fc(combined) 
        return output
