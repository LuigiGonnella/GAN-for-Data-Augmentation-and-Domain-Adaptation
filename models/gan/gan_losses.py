import torch
import torch.nn as nn


class BCEWithLogitsLoss:
    """
    Binary Cross Entropy with Logits Loss
    """
    def __init__(self):
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def discriminator_loss(self, real_output, fake_output):
        """
        Discriminator loss: classify real as 1, fake as 0

        """
        real_loss = self.loss_fn(real_output, torch.ones_like(real_output))
        fake_loss = self.loss_fn(fake_output, torch.zeros_like(fake_output))
        return (real_loss + fake_loss) / 2
    
    def generator_loss(self, fake_output):
        """
        Generator loss: fool discriminator into thinking fakes are real
        
        """
        return self.loss_fn(fake_output, torch.ones_like(fake_output))


class WassersteinLoss:
    """
    Wasserstein Loss (WGAN)

    """
    def __init__(self, lambda_gp=10):
        self.lambda_gp = lambda_gp
    
    def discriminator_loss(self, real_output, fake_output):
        """
        Wasserstein discriminator loss (critic loss)
        Maximizes: E[D(real)] - E[D(fake)]
        In optimization: minimizes -E[D(real)] + E[D(fake)]

        """
        return torch.mean(fake_output) - torch.mean(real_output)
    
    def generator_loss(self, fake_output):
        """
        Wasserstein generator loss
        Minimizes: -E[D(fake)] (equivalent to maximizing E[D(fake)])

        """
        return -torch.mean(fake_output)
    
    def gradient_penalty(self, discriminator, real_data, fake_data, labels=None):
        batch_size = real_data.size(0)
        device = real_data.device
        
        alpha = torch.rand(batch_size, 1, 1, 1, device=device, requires_grad=True)
        
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        
        if labels is not None:
            d_interpolates = discriminator(interpolates, labels)
        else:
            d_interpolates = discriminator(interpolates)
        
        fake = torch.ones(batch_size, 1, device=device)
        
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        
        gradients_flat = gradients.view(batch_size, -1)
        gradient_norm = torch.sqrt(torch.sum(gradients_flat ** 2, dim=1) + 1e-12)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        
        return self.lambda_gp * gradient_penalty


class MCELoss:
    """
    Mean Squared Error Loss for GANs
    """
    def __init__(self):
        self.loss_fn = nn.MSELoss()
    
    def discriminator_loss(self, real_output, fake_output):
        """
        MSE discriminator loss
    
        """
        real_loss = self.loss_fn(real_output, torch.ones_like(real_output))
        fake_loss = self.loss_fn(fake_output, torch.zeros_like(fake_output))
        return (real_loss + fake_loss) / 2
    
    def generator_loss(self, fake_output):
        """
        MSE generator loss
        
        """
        return self.loss_fn(fake_output, torch.ones_like(fake_output))

class HingeLoss:
    """
    Hinge Loss for GANs
    
    Used in SN-GAN (Miyato et al., 2018) and SA-GAN (Zhang et al., 2019).
    Provides stable training with strong gradients and better mode coverage.
    
    References:
        - Miyato et al. "Spectral Normalization for GANs" (ICLR 2018)
        - Zhang et al. "Self-Attention GANs" (ICML 2019)
    """
    
    def __init__(self):
        pass
    
    def discriminator_loss(self, real_output, fake_output):
        """
        Hinge loss for discriminator:
        L_D = E[max(0, 1 - D(x_real))] + E[max(0, 1 + D(x_fake))]
        
        The discriminator tries to:
        - Push real samples above +1
        - Push fake samples below -1
        
        Args:
            real_output: Discriminator output for real images [B, 1, N, N] 
            fake_output: Discriminator output for fake images [B, 1, N, N] 
        
        Returns:
            Scalar loss value
        """
        # For PatchGAN, average over all spatial positions
        real_loss = torch.mean(torch.relu(1.0 - real_output))
        fake_loss = torch.mean(torch.relu(1.0 + fake_output))
        return real_loss + fake_loss
    
    def generator_loss(self, fake_output):
        """
        Hinge loss for generator:
        L_G = -E[D(G(z))]
        
        Generator tries to maximize discriminator output for fake images.
        
        Args:
            fake_output: Discriminator output for fake images [B, 1, N, N] or [B, 1]
        
        Returns:
            Scalar loss value
        """
        return -torch.mean(fake_output)


def get_loss_fn(loss_type, **kwargs):
    """
    Factory function to instantiate loss function by name
    
    Args:
        loss_type: 'bce', 'wasserstein', or 'mse'
        **kwargs: additional arguments for loss initialization
                  - lambda_gp: gradient penalty weight for Wasserstein loss (default: 10)
    
    Returns:
        Loss function instance
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'bce':
        return BCEWithLogitsLoss()
    elif loss_type == 'wasserstein':
        lambda_gp = kwargs.get('lambda_gp', 10)
        return WassersteinLoss(lambda_gp=lambda_gp)
    elif loss_type == 'mse':
        return MCELoss()
    elif loss_type == 'hinge':
        return HingeLoss()
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. "
            f"Choose from: 'bce', 'wasserstein', 'mse', 'hinge'"
        )


