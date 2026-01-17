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


class ConditionalPatchGANDiscriminatorSN(nn.Module):
    """
    Conditional PatchGAN Discriminator with Spectral Normalization
    
    PatchGAN outputs spatial logits (NxN) instead of a single value.
    Each output corresponds to a patch's receptive field in the input.
    Class conditioning via projection discriminator approach.
    """

    def __init__(self, num_classes=2, ndf=64, channels=3, n_layers=3, width=128, height=128, dropout=0.3):
        super(ConditionalPatchGANDiscriminatorSN, self).__init__()
        
        self.num_classes = num_classes
        
        # First conv layer (no normalization)
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
        
        self.conv_layers = nn.Sequential(*model)
        
        # Class embedding for projection discriminator
        self.class_embedding = nn.Embedding(num_classes, ndf * nf_mult)
        
        # Final projection layer for PatchGAN output
        # Produces spatial logits (no sigmoid)
        self.output_conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def forward(self, x, labels):
        """
        Forward pass with conditional projection
        
        Args:
            x: Input images [B, C, H, W]
            labels: Class labels [B]
        
        Returns:
            Patch logits [B, 1, N, N] - raw logits for Hinge Loss
        """
        # Extract spatial features
        features = self.conv_layers(x)  # [B, 512, 8, 8]
        
        # Get spatial output
        output = self.output_conv(features)  # [B, 1, 7, 7]
        
        # Conditional projection discriminator
        # Add class information via inner product with embedding
        class_embedding = self.class_embedding(labels)  # [B, 512]
        class_embedding = class_embedding.view(class_embedding.size(0), -1, 1, 1)  # [B, 512, 1, 1]
        
        # Project features with class embedding and add to output
        projection = torch.sum(features * class_embedding, dim=1, keepdim=True)  # [B, 1, 8, 8]
        
        # Interpolate projection to match output spatial size
        if projection.size(-1) != output.size(-1):
            projection = nn.functional.adaptive_avg_pool2d(projection, output.size()[-2:])
        
        output = output + projection
        
        return output


class ConditionalPatchGANDiscriminator(nn.Module):
    """
    Conditional PatchGAN Discriminator (without Spectral Normalization)
    
    PatchGAN outputs spatial logits (NxN) instead of a single value.
    Each output corresponds to a patch's receptive field in the input.
    Uses BatchNorm for stability (alternative to spectral normalization).
    """

    def __init__(self, num_classes=2, ndf=64, channels=3, n_layers=3, width=128, height=128, dropout=0.3):
        super(ConditionalPatchGANDiscriminator, self).__init__()
        
        self.num_classes = num_classes
        
        # First conv layer (no normalization)
        model = [
            nn.Conv2d(channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout)
        ]
        
        # Gradually increase number of filters
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # Cap at 8x base filters
            
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(dropout)
            ]
        
        # Penultimate layer (stride=1 to maintain spatial dimensions)
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                     kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout)
        ]
        
        self.conv_layers = nn.Sequential(*model)
        
        # Class embedding for projection discriminator
        self.class_embedding = nn.Embedding(num_classes, ndf * nf_mult)
        
        # Final projection layer for PatchGAN output
        # Produces spatial logits (no sigmoid)
        self.output_conv = nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
    
    def forward(self, x, labels):
        """
        Forward pass with conditional projection
        
        Args:
            x: Input images [B, C, H, W]
            labels: Class labels [B]
        
        Returns:
            Patch logits [B, 1, N, N] - raw logits for Hinge Loss
        """
        # Extract spatial features
        features = self.conv_layers(x)  # [B, 512, 8, 8]
        
        # Get spatial output
        output = self.output_conv(features)  # [B, 1, 7, 7]
        
        # Conditional projection discriminator
        # Add class information via inner product with embedding
        class_embedding = self.class_embedding(labels)  # [B, 512]
        class_embedding = class_embedding.view(class_embedding.size(0), -1, 1, 1)  # [B, 512, 1, 1]
        
        # Project features with class embedding and add to output
        projection = torch.sum(features * class_embedding, dim=1, keepdim=True)  # [B, 1, 8, 8]
        
        # Interpolate projection to match output spatial size
        if projection.size(-1) != output.size(-1):
            projection = nn.functional.adaptive_avg_pool2d(projection, output.size()[-2:])
        
        output = output + projection
        
        return output
