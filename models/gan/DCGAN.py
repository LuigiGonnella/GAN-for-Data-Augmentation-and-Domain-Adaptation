import torch
import torch.nn as nn

class DCGANGenerator(nn.Module):
    def __init__(self, input_dim=100, n1=512, channels=3, width=128, height=128):
        super(DCGANGenerator, self).__init__()

        self.n1 = n1
        self.fc = nn.Linear(input_dim, n1 * 4 * 4)

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
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.n1, 4, 4)  
        x = self.conv_blocks(x)
        return x

class DCGANDiscriminator(nn.Module):
    def __init__(self, channels=3, width=128, height=128, dropout=0.5):
        super(DCGANDiscriminator, self).__init__()

        self.model = nn.Sequential(
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

            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x
