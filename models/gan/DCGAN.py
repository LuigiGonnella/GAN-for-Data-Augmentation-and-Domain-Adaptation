import torch
import torch.nn as nn

class DCGANGenerator(nn.Module):
    def __init__(self, input_dim=100, n1=512, channels=3):
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

class PatchGANDiscriminatorSN(nn.Module):
    """
    PatchGAN Discriminator with Spectral Normalization
    
    Architecture based on:
    - PatchGAN: Isola et al. "Image-to-Image Translation with Conditional GANs" (CVPR 2017)
    - Spectral Norm: Miyato et al. "Spectral Normalization for GANs" (ICLR 2018)
    
    Key features:
    - Outputs NxN predictions for local patches
    - Spectral normalization on all convolutional layers
    - No sigmoid activation (for use with Hinge Loss)
    - Removed batch normalization (incompatible with SN in discriminator)
    """
    
    def __init__(self, channels=3, ndf=64, n_layers=3, dropout=0.3):
        """
        Args:
            channels: Number of input channels (3 for RGB)
            ndf: Number of discriminator filters in first conv layer
            n_layers: Number of downsampling conv layers (controls patch size)
            dropout: Dropout probability (applied after LeakyReLU)
        
        Receptive field for 128x128 images:
            n_layers=3 → 14x14 output → 70x70 pixel receptive field (recommended)
            n_layers=4 → 6x6 output → 142x142 pixel receptive field
            n_layers=5 → 2x2 output → full image receptive field
        """
        super(PatchGANDiscriminatorSN, self).__init__()
        
        # First conv layer (no normalization, per original PatchGAN)
        model = [
            nn.utils.spectral_norm(
                nn.Conv2d(channels, ndf, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout)
        ]
        
        # Gradually increase number of filters
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # Cap at 8x base filters
            
            model += [
                nn.utils.spectral_norm(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                             kernel_size=4, stride=2, padding=1)
                ),
                # Note: NO BatchNorm when using Spectral Normalization
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(dropout)
            ]
        
        # Penultimate layer (stride=1 to maintain spatial dimensions)
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=4, stride=1, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout)
        ]
        
        # Final output layer - produces NxN logits (no sigmoid)
        model += [
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
            )
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images [B, C, H, W]
        
        Returns:
            Patch logits [B, 1, N, N] - no sigmoid, raw logits for Hinge Loss
        """
        return self.model(x)

class PatchGANDiscriminator(nn.Module):
    """
    Standard PatchGAN Discriminator with Batch Normalization
    
    Architecture based on:
    - PatchGAN: Isola et al. "Image-to-Image Translation with Conditional GANs" (CVPR 2017)
    
    Key features:
    - Outputs NxN predictions for local patches
    - Batch normalization on convolutional layers (except first)
    - No sigmoid activation (for use with various loss functions)
    - Standard architecture without spectral normalization
    """
    
    def __init__(self, channels=3, ndf=64, n_layers=3, dropout=0.0):
        """
        Args:
            channels: Number of input channels (3 for RGB)
            ndf: Number of discriminator filters in first conv layer
            n_layers: Number of downsampling conv layers (controls patch size)
            dropout: Dropout probability (applied after LeakyReLU)
        
        Receptive field for 128x128 images:
            n_layers=3 → 14x14 output → 70x70 pixel receptive field (recommended)
            n_layers=4 → 6x6 output → 142x142 pixel receptive field
            n_layers=5 → 2x2 output → full image receptive field
        """
        super(PatchGANDiscriminator, self).__init__()
        
        # First conv layer (no normalization, per original PatchGAN)
        model = [
            nn.Conv2d(channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        if dropout > 0:
            model.append(nn.Dropout2d(dropout))
        
        # Gradually increase number of filters
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # Cap at 8x base filters
            
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            
            if dropout > 0:
                model.append(nn.Dropout2d(dropout))
        
        # Penultimate layer (stride=1 to maintain spatial dimensions)
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                     kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        if dropout > 0:
            model.append(nn.Dropout2d(dropout))
        
        # Final output layer - produces NxN logits (no sigmoid)
        model += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input images [B, C, H, W]
        
        Returns:
            Patch logits [B, 1, N, N] - no sigmoid, raw logits
        """
        return self.model(x)
