# PatchGAN with Hinge Loss and Spectral Normalization

This document provides a comprehensive guide on implementing and training a PatchGAN discriminator with Hinge Loss and Spectral Normalization, based on state-of-the-art GAN architectures for high-quality image generation.

## Table of Contents
1. [Theoretical Background](#theoretical-background)
2. [Implementation Guide](#implementation-guide)
3. [Training Procedure](#training-procedure)
4. [Why This Works Better](#why-this-works-better)
5. [Integration with Existing Code](#integration-with-existing-code)
6. [References](#references)

---

## Theoretical Background

### PatchGAN Architecture

PatchGAN is a discriminator architecture that classifies whether **overlapping patches** of an image are real or fake, rather than classifying the entire image as a single scalar value. Each output value in the NxN output tensor corresponds to a patch's receptive field in the input image.

**Key Concept**: Instead of outputting a single value D(x) ∈ ℝ, PatchGAN outputs D(x) ∈ ℝ^(N×N), where each element D_ij represents the realness score of a local patch.

**Advantages for Medical Imaging**:
- Better captures local texture details critical for skin lesion images
- More parameter-efficient than full-image discriminators
- Provides stronger gradient signals through multiple patch classifications

### Hinge Loss

Hinge loss is a margin-based loss function popularized by spectral normalization GANs (SN-GAN) [Miyato et al., 2018]. Unlike BCE loss which uses sigmoid activation, hinge loss operates directly on the discriminator's raw logits.

**Mathematical Formulation**:

For the discriminator:
```
L_D = 𝔼[max(0, 1 - D(x_real))] + 𝔼[max(0, 1 + D(x_fake))]
```

For the generator:
```
L_G = -𝔼[D(G(z))]
```

**Intuition**: 
- The discriminator tries to push real samples above +1 and fake samples below -1
- A margin of 2 units provides more stable gradients than sigmoid-based losses
- No saturation issues that plague BCE loss

**Why Hinge Loss?**
1. **No gradient vanishing**: Unlike BCE, gradients remain strong even when discriminator is confident
2. **Stable training dynamics**: The margin prevents discriminator from becoming too confident
3. **Better mode coverage**: Generator receives meaningful gradients across wider range of outputs
4. **Empirically superior**: Consistently produces higher quality images [Zhang et al., 2019]

### Spectral Normalization

Spectral normalization (SN) constrains the Lipschitz constant of the discriminator by normalizing the weight matrices by their spectral norm (largest singular value) [Miyato et al., 2018].

**Mathematical Definition**:

For weight matrix W, the spectral normalized weight is:
```
W_SN = W / σ(W)
```
where σ(W) is the largest singular value of W (spectral norm).

**Implementation**: PyTorch provides `nn.utils.spectral_norm()` which wraps any layer with spectral normalization. The spectral norm is computed efficiently using power iteration method during training.

**Why Spectral Normalization?**
1. **Lipschitz continuity**: Bounds ||∇_x D(x)||_2 ≤ K, ensuring discriminator doesn't change too rapidly
2. **Training stability**: Prevents discriminator gradients from exploding
3. **No hyperparameter tuning**: Unlike gradient penalty (WGAN-GP), requires no λ coefficient
4. **Computational efficiency**: Minimal overhead compared to gradient penalty
5. **Better than BatchNorm**: Removes need for batch statistics in discriminator

**Theoretical Guarantee**: If each layer has Lipschitz constant ≤ 1, the entire network has Lipschitz constant ≤ 1 (by composition property).

---

## Implementation Guide

### Step 1: Implement Hinge Loss

Add the following class to [`models/gan/gan_losses.py`](../models/gan/gan_losses.py):

```python
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
            real_output: Discriminator output for real images [B, 1, N, N] or [B, 1]
            fake_output: Discriminator output for fake images [B, 1, N, N] or [B, 1]
        
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
```

Update the `get_loss_fn()` function in the same file:

```python
def get_loss_fn(loss_type, **kwargs):
    """
    Factory function to instantiate loss function by name
    
    Args:
        loss_type: 'bce', 'wasserstein', 'mse', or 'hinge'
        **kwargs: additional arguments for loss initialization
    
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
```

### Step 2: Implement PatchGAN with Spectral Normalization

Replace the `PatchDCGANDiscriminator` class in [`models/gan/DCGAN.py`](../models/gan/DCGAN.py):

```python
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
        
        Receptive field for 128×128 images:
            n_layers=3 → 14×14 output → 70×70 pixel receptive field (recommended)
            n_layers=4 → 6×6 output → 142×142 pixel receptive field
            n_layers=5 → 2×2 output → full image receptive field
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
```

### Step 3: Update Training Configuration

Create or modify your experiment config file (e.g., `experiments/dcgan_patchgan_hinge_sn.yaml`):

```yaml
model:
  type: "dcgan"
  generator:
    latent_dim: 100
    n1: 512
    channels: 3
    width: 128
    height: 128
  discriminator:
    type: "patchgan_sn"
    channels: 3
    ndf: 64
    n_layers: 3      # For 128×128 → 70×70 pixel receptive field
    dropout: 0.3

training:
  batch_size: 32
  epochs: 300
  g_lr: 0.0001     # Lower learning rate with SN
  d_lr: 0.0004     # Discriminator can train faster with SN
  n_critic: 1      # Balanced training (SN provides stability)

loss:
  type: "hinge"

data:
  train_dir: "data/processed/baseline/train/malignant"
  image_size: 128

output:
  sample_dir: "data/synthetic/dcgan_patchgan_hinge_sn"
  metrics_dir: "results/gan_metrics/dcgan_patchgan_hinge_sn"
  save_interval: 50

evaluation:
  fid_is_interval: 50
  fid_is_num_samples: 256
```

---

## Training Procedure

### Modified Training Loop

The training procedure for PatchGAN with Hinge Loss and Spectral Normalization in [`training/train_dcgan.py`](../training/train_dcgan.py) is similar to standard DCGAN, with key differences:

```python
def train_step(self, real_images, latent_dim):
    """
    Single training step for PatchGAN with Hinge Loss and Spectral Normalization
    """
    batch_size = real_images.size(0)
    
    # ---------------------
    # Train Discriminator
    # ---------------------
    self.d_optimizer.zero_grad()
    
    # Real images - PatchGAN outputs [B, 1, N, N]
    real_output = self.discriminator(real_images)
    
    # Fake images
    z = torch.randn(batch_size, latent_dim, device=self.device)
    fake_images = self.generator(z)
    fake_output = self.discriminator(fake_images.detach())
    
    # Hinge loss for discriminator
    # L_D = E[max(0, 1 - D(x_real))] + E[max(0, 1 + D(x_fake))]
    d_loss = self.loss_fn.discriminator_loss(real_output, fake_output)
    
    d_loss.backward()
    # Gradient clipping still recommended for stability
    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
    self.d_optimizer.step()
    
    # ---------------------
    # Train Generator
    # ---------------------
    self.g_optimizer.zero_grad()
    
    # Generate new fake images
    z = torch.randn(batch_size, latent_dim, device=self.device)
    fake_images = self.generator(z)
    fake_output = self.discriminator(fake_images)
    
    # Hinge loss for generator
    # L_G = -E[D(G(z))]
    g_loss = self.loss_fn.generator_loss(fake_output)
    
    g_loss.backward()
    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
    self.g_optimizer.step()
    
    return d_loss.item(), g_loss.item()
```

### Key Training Differences

1. **No Gradient Penalty**: Unlike WGAN-GP, spectral normalization doesn't require gradient penalty computation
2. **Balanced Training**: With SN, discriminator is naturally constrained, so `n_critic=1` is sufficient
3. **Learning Rates**: Can use higher discriminator learning rate (e.g., `d_lr=0.0004`, `g_lr=0.0001`)
4. **No BatchNorm in Discriminator**: Spectral norm replaces batch normalization
5. **Raw Logits**: Discriminator outputs are NOT passed through sigmoid (hinge loss uses raw logits)

---

## Why This Works Better

### 1. Hinge Loss Advantages

**Problem with BCE Loss**: The standard BCE loss with sigmoid can saturate when the discriminator becomes too confident:
```
L_BCE = -[log(D(x_real)) + log(1 - D(x_fake))]
```
When D(x_real) ≈ 1 or D(x_fake) ≈ 0, gradients vanish (∇log(x) → 0 as x → 1).

**Hinge Loss Solution**: 
```
L_Hinge = max(0, 1 - D(x_real)) + max(0, 1 + D(x_fake))
```
- Provides **constant gradient** magnitude when discriminator is wrong (within margin)
- **No saturation**: Gradients don't vanish even when discriminator is confident
- **Margin-based**: The ±1 margin provides natural regularization

**Empirical Results** [Miyato et al., 2018; Zhang et al., 2019]:
- **Inception Score**: 5-15% improvement over BCE
- **FID Score**: 10-20% reduction (better image quality)
- **Training Stability**: Fewer mode collapses and oscillations

### 2. Spectral Normalization Advantages

**Problem with Unconstrained Discriminator**: 
- Discriminator gradients can explode → unstable training
- Requires careful hyperparameter tuning (learning rates, gradient clipping)
- BatchNorm adds noise that can harm discriminator performance

**Spectral Normalization Solution**:
By constraining ||W||_2 ≤ 1 for all weight matrices:

1. **Lipschitz Continuity**: Guarantees ||∇_x D(x)||_2 ≤ K
   - Discriminator can't change too rapidly
   - Smoother loss landscape for generator

2. **Stable Gradients**: Prevents gradient explosion without gradient clipping
   - More reliable gradient signals to generator
   - Consistent training dynamics

3. **No Hyperparameter Sensitivity**: 
   - Unlike WGAN-GP (requires tuning λ_gp)
   - Unlike layer normalization (requires tuning scale/shift)
   - Works out-of-the-box with default settings

4. **Computational Efficiency**:
   - O(n) per iteration using power iteration
   - No extra backward pass (unlike gradient penalty)
   - Minimal memory overhead

**Theoretical Justification** [Miyato et al., 2018]:
The discriminator's Lipschitz constant bounds the gradients:
```
||∇_x D(x)||_2 ≤ ∏_{l=1}^L ||W_l||_2
```
By setting ||W_l||_2 = 1, we get ||∇_x D(x)||_2 ≤ 1, preventing gradient explosion.

### 3. PatchGAN Advantages

**Problem with Full-Image Discriminator**:
- Single scalar output provides weak gradient signal
- Must learn global and local features simultaneously
- Inefficient for high-resolution images

**PatchGAN Solution**:
- **Multiple gradient sources**: Each patch provides independent gradient
- **Local texture focus**: Better for medical imaging where local details matter
- **Parameter efficiency**: Fully convolutional → fewer parameters than FC layers

**For Medical Imaging Specifically**:
- Skin lesion diagnosis relies on local texture, color variation, and boundaries
- PatchGAN naturally emphasizes these local features
- Better augmentation quality → better classifier performance

### 4. Combined Synergy

The combination of **Hinge Loss + Spectral Norm + PatchGAN** is particularly powerful:

1. **Stable Training**: SN + Hinge Loss eliminate most training instabilities
2. **Better Gradients**: Multiple patches × non-saturating loss = strong learning signals
3. **High Quality**: Local discrimination + stable training = realistic details
4. **Minimal Tuning**: Works with standard hyperparameters

**Published Results**:
- **SN-GAN** [Miyato et al., 2018]: Achieved SOTA on ImageNet generation (IS: 52.52)
- **SA-GAN** [Zhang et al., 2019]: Further improved with self-attention (IS: 52.52 → 55.14)
- **BigGAN** [Brock et al., 2019]: Scaled to 512×512 images using these techniques

---

## Integration with Existing Code

### Step 1: Add Hinge Loss Implementation

```bash
# Edit models/gan/gan_losses.py
# Add the HingeLoss class as shown above
```

### Step 2: Add Spectral Norm Discriminator

```bash
# Edit models/gan/DCGAN.py
# Add the PatchGANDiscriminatorSN class as shown above
```

### Step 3: Update Discriminator Initialization

In [`training/train_dcgan.py`](../training/train_dcgan.py), modify the `_build_discriminator` method:

```python
def _build_discriminator(self):
    disc_config = self.config['model']['discriminator']
    disc_type = disc_config.get('type', 'patch')
    
    if disc_type == 'patchgan_sn':
        from models.gan.DCGAN import PatchGANDiscriminatorSN
        discriminator = PatchGANDiscriminatorSN(
            channels=disc_config['channels'],
            ndf=disc_config.get('ndf', 64),
            n_layers=disc_config.get('n_layers', 3),
            dropout=disc_config.get('dropout', 0.3)
        ).to(self.device)
    else:
        # Fallback to existing PatchDCGANDiscriminator
        discriminator = PatchDCGANDiscriminator(
            channels=disc_config['channels'],
            width=disc_config['width'],
            height=disc_config['height'],
            dropout=disc_config['dropout']
        ).to(self.device)
    
    return discriminator
```

### Step 4: Run Training

```bash
# Train with the new configuration
python training/train_dcgan.py --config experiments/dcgan_patchgan_hinge_sn.yaml
```

### Step 5: Monitor Metrics

Expected improvements over standard DCGAN with BCE:
- **FID Score**: 10-20% lower (better)
- **Inception Score**: 5-15% higher (better diversity)
- **Training Stability**: Fewer oscillations in loss curves
- **Visual Quality**: Better texture details in skin lesions

---

## References

### Core Papers

1. **Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018)**  
   *"Spectral Normalization for Generative Adversarial Networks"*  
   International Conference on Learning Representations (ICLR 2018)  
   [https://arxiv.org/abs/1802.05957](https://arxiv.org/abs/1802.05957)
   - Introduced spectral normalization for GANs
   - Demonstrated superior training stability and image quality
   - Achieved SOTA on conditional image generation

2. **Zhang, H., Goodfellow, I., Metaxas, D., & Odena, A. (2019)**  
   *"Self-Attention Generative Adversarial Networks"*  
   International Conference on Machine Learning (ICML 2019)  
   [https://arxiv.org/abs/1805.08318](https://arxiv.org/abs/1805.08318)
   - Used hinge loss with spectral normalization
   - Added self-attention mechanisms for long-range dependencies
   - Achieved IS: 52.52 on ImageNet

3. **Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017)**  
   *"Image-to-Image Translation with Conditional Adversarial Networks"*  
   IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2017)  
   [https://arxiv.org/abs/1611.07004](https://arxiv.org/abs/1611.07004)
   - Introduced PatchGAN discriminator (pix2pix)
   - Demonstrated effectiveness for texture and local structure
   - Widely used in image translation tasks

4. **Brock, A., Donahue, J., & Simonyan, K. (2019)**  
   *"Large Scale GAN Training for High Fidelity Natural Image Synthesis"*  
   International Conference on Learning Representations (ICLR 2019)  
   [https://arxiv.org/abs/1809.11096](https://arxiv.org/abs/1809.11096)
   - BigGAN: Scaled GANs to 512×512 using SN + Hinge Loss
   - Achieved SOTA on ImageNet (IS: 166.3, FID: 7.4)
   - Demonstrated importance of these techniques for large-scale training

### Supporting Literature

5. **Radford, A., Metz, L., & Chintala, S. (2016)**  
   *"Unsupervised Representation Learning with Deep Convolutional GANs"*  
   International Conference on Learning Representations (ICLR 2016)  
   [https://arxiv.org/abs/1511.06434](https://arxiv.org/abs/1511.06434)
   - DCGAN architecture guidelines
   - Foundation for modern GAN architectures

6. **Lim, J. H., & Ye, J. C. (2017)**  
   *"Geometric GAN"*  
   [https://arxiv.org/abs/1705.02894](https://arxiv.org/abs/1705.02894)
   - Analyzed margin-based losses for GANs
   - Theoretical justification for hinge loss

7. **Yoshida, Y., & Miyato, T. (2017)**  
   *"Spectral Norm Regularization for Improving the Generalizability of Deep Learning"*  
   [https://arxiv.org/abs/1705.10941](https://arxiv.org/abs/1705.10941)
   - Theoretical foundations of spectral normalization
   - Lipschitz continuity and generalization

### Medical Imaging Applications

8. **Frid-Adar, M., Diamant, I., Klang, E., Amitai, M., Goldberger, J., & Greenspan, H. (2018)**  
   *"GAN-based Synthetic Medical Image Augmentation"*  
   [https://arxiv.org/abs/1805.00247](https://arxiv.org/abs/1805.00247)
   - Application of GANs for medical image augmentation
   - Demonstrated improved classifier performance

9. **Bissoto, A., Perez, F., Valle, E., & Avila, S. (2019)**  
   *"Skin Lesion Synthesis with GANs"*  
   ISIC Skin Image Analysis Workshop, CVPR 2019  
   - GAN applications for dermatology datasets
   - Showed importance of local texture for skin lesion synthesis

---

## Quick Reference

### Key Hyperparameters

| Parameter | Recommended Value | Notes |
|-----------|------------------|-------|
| Loss type | `hinge` | Better than BCE for image quality |
| n_layers | `3` (for 128×128) | 70×70 pixel receptive field |
| ndf | `64` | Base discriminator filters |
| dropout | `0.3` | Lower than standard due to SN |
| g_lr | `0.0001` | Standard Adam learning rate |
| d_lr | `0.0004` | Can be higher with SN stability |
| n_critic | `1` | Balanced training with SN |
| β1, β2 | `0.5, 0.999` | Standard Adam betas for GANs |

### Training Tips

1. **Remove BatchNorm from Discriminator**: SN and BN conflict in discriminator
2. **Keep BatchNorm in Generator**: Generator benefits from BN
3. **No Sigmoid in Discriminator**: Hinge loss uses raw logits
4. **Monitor Discriminator Outputs**: Should be around ±1 (hinge margin)
5. **Check Spectral Norms**: Can log `torch.linalg.matrix_norm(W, ord=2)` for debugging

### Expected Results

Compared to DCGAN with BCE loss on 128×128 skin lesion images:
- **FID Score**: ~30-40 → ~20-30 (30% improvement)
- **Inception Score**: ~2.0-2.5 → ~2.5-3.0 (20% improvement)  
- **Training Time**: Similar or slightly faster due to stability
- **Visual Quality**: Noticeably better texture and boundary details

---

## Conclusion

The combination of **PatchGAN**, **Hinge Loss**, and **Spectral Normalization** represents the state-of-the-art for stable, high-quality GAN training. These techniques:

1. **Eliminate common GAN failures**: Mode collapse, training instability, gradient issues
2. **Improve image quality**: Better FID/IS scores, more realistic local textures
3. **Require minimal tuning**: Work with standard hyperparameters
4. **Scale effectively**: Used in production systems (BigGAN, StyleGAN variations)

For medical image augmentation, particularly skin lesion synthesis, these techniques are essential for generating training data that improves downstream classifier performance while maintaining clinically relevant local features.
