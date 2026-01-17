import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import yaml
from pathlib import Path
import argparse
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gan.DCGAN import DCGANGenerator, PatchGANDiscriminatorSN, PatchGANDiscriminator
from models.gan.gan_losses import get_loss_fn
from evaluation.gan_metrics import GANMetrics


class GANTrainer:
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        print(f"Loss type: {self.config['loss']['type']}")
        
        self._create_output_dirs()
        
        self.generator = self._build_generator()

        if config['loss']['type'] == 'hinge':
            self.discriminator = self._build_discriminator_with_SN()
        else:
            self.discriminator = self._build_discriminator_without_SN()
        
        self.loss_fn = self._init_loss()
        
        self.g_optimizer, self.d_optimizer = self._init_optimizers()
        
        self.metrics = GANMetrics(
            output_dir=self.config['output']['metrics_dir'],
            device=self.device
        )
        self.train_loader = self._init_dataloader()
        
        self.fid_is_interval = self.config.get('evaluation', {}).get('fid_is_interval', 50)
        self.fid_is_num_samples = self.config.get('evaluation', {}).get('fid_is_num_samples', 256)
    
    
    def _create_output_dirs(self):
        dirs = [
            self.config['output']['sample_dir'],
            self.config['output']['metrics_dir'],
            self.config['output'].get('checkpoint_dir', 'results/checkpoints')
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def _build_generator(self):
        gen_config = self.config['model']['generator']
        generator = DCGANGenerator(
            input_dim=gen_config['latent_dim'],
            n1=gen_config['n1'],
            channels=gen_config['channels'],
        ).to(self.device)
        return generator
    
    def _build_discriminator_without_SN(self):
        disc_config = self.config['model']['discriminator']
        discriminator = PatchGANDiscriminator(
            channels=disc_config['channels'],
            dropout=disc_config['dropout'],
            ndf= disc_config['ndf'],
            n_layers= disc_config['n_layers']
        ).to(self.device)
        return discriminator
    
    def _build_discriminator_with_SN(self):
        disc_config = self.config['model']['discriminator']
        discriminator = PatchGANDiscriminatorSN(
            channels=disc_config['channels'],
            dropout=disc_config['dropout'],
            ndf= disc_config['ndf'],
            n_layers= disc_config['n_layers']
        ).to(self.device)
        return discriminator
    
    def _init_loss(self):
        loss_type = self.config['loss']['type']
        loss_kwargs = {}
        
        if loss_type == 'wasserstein':
            loss_kwargs['lambda_gp'] = self.config['loss'].get('lambda_gp', 10)
        
        loss_fn = get_loss_fn(loss_type, **loss_kwargs)
        return loss_fn
    
    def _init_optimizers(self):
        g_lr = self.config['training']['g_lr']
        d_lr = self.config['training']['d_lr']
        
        g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=g_lr,
            betas=(0.5, 0.999)
        )
        
        d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=d_lr,
            betas=(0.5, 0.999)
        )
        
        return g_optimizer, d_optimizer
    
    def _init_dataloader(self):
        data_dir = self.config['data']['train_dir']
        image_size = self.config['data']['image_size']
        batch_size = self.config['training']['batch_size']
        
        # Training augmentation for GANs - helps with limited medical data
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
        ])
        
        dataset = datasets.ImageFolder(
            root=str(Path(data_dir).parent),
            transform=transform
        )
        
        malignant_indices = [i for i, (_, label) in enumerate(dataset.samples)
                            if 'malignant' in dataset.samples[i][0]]
        
        from torch.utils.data import Subset
        malignant_dataset = Subset(dataset, malignant_indices)
        
        dataloader = DataLoader(
            malignant_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Loaded {len(malignant_dataset)} malignant images")
        return dataloader
    
    def _collect_real_images(self, num_samples):
        real_images = []
        
        for real_batch, _ in self.train_loader:
            real_images.append(real_batch.to(self.device))
            if sum(img.shape[0] for img in real_images) >= num_samples:
                break
        
        real_images = torch.cat(real_images, dim=0)[:num_samples]
        return real_images
    
    def _generate_fake_images(self, num_samples, latent_dim):
        fake_images = []
        num_generated = 0
        
        self.generator.eval()
        with torch.no_grad():
            while num_generated < num_samples:
                batch_size = min(64, num_samples - num_generated)
                z = torch.randn(batch_size, latent_dim, device=self.device)
                batch_fake = self.generator(z)
                fake_images.append(batch_fake)
                num_generated += batch_size
        
        fake_images = torch.cat(fake_images, dim=0)[:num_samples]
        self.generator.train()
        
        return fake_images
    
    def _compute_fid_is(self, latent_dim):
        try:
            print("Computing FID/IS scores...")
            
            # Collect images
            real_images = self._collect_real_images(self.fid_is_num_samples)
            fake_images = self._generate_fake_images(self.fid_is_num_samples, latent_dim)
            
            # Compute FID
            fid_score = self.metrics.compute_fid(real_images, fake_images)
            
            # Compute IS
            is_mean, is_std = self.metrics.compute_inception_score(fake_images)
            
            # Update metrics
            self.metrics.update_fid_is(fid_score, is_mean, is_std)
            
            print(f"  FID Score: {fid_score:.4f} | IS Score: {is_mean:.4f}±{is_std:.4f}")
            
            return fid_score, is_mean, is_std
        
        except Exception as e:
            print(f"Error computing FID/IS: {str(e)}")
            return None, None, None
    
    def train(self, n_iter=False):
        epochs = self.config['training']['epochs']
        n_critic = self.config['training'].get('n_critic', 1)  # For Wasserstein
        save_interval = self.config['output']['save_interval']
        latent_dim = self.config['model']['generator']['latent_dim']
        loss_type = self.config['loss']['type']
        
        print(f"\nStarting training for {epochs} epochs with {loss_type} loss")
        print(f"n_critic: {n_critic}\n")
        
        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_d_real = 0.0
            epoch_d_fake = 0.0
            num_batches = 0
            
            for batch_idx, (real_images, _) in enumerate(self.train_loader):
                try:
                    real_images = real_images.to(self.device, non_blocking=True)
                    batch_size = real_images.size(0)

                    # ---------------------
                    # Train Discriminator
                    # ---------------------
                    
                    for _ in range(n_critic):
                        self.d_optimizer.zero_grad()
                        
                        # Real images - PatchGAN outputs [B, 1, N, N]
                        d_real_output = self.discriminator(real_images)
                        
                        # Fake images
                        z = torch.randn(batch_size, latent_dim, device=self.device)
                        fake_images = self.generator(z)
                        d_fake_output = self.discriminator(fake_images.detach())
                        
                        if loss_type == 'wasserstein':
                            d_loss = self.loss_fn.discriminator_loss(d_real_output, d_fake_output)
                            gp = self.loss_fn.gradient_penalty(
                                self.discriminator, real_images, fake_images.detach()
                            )
                            d_loss = d_loss + gp
                        else:
                            d_loss = self.loss_fn.discriminator_loss(d_real_output, d_fake_output)
                        
                        d_loss.backward()

                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                        self.d_optimizer.step()
                    
                    # ---------------------
                    # Train Generator
                    # ---------------------
                    self.g_optimizer.zero_grad()
                    
                    # Generate new fake images
                    z = torch.randn(batch_size, latent_dim, device=self.device)
                    fake_images = self.generator(z)
                    d_fake_output = self.discriminator(fake_images)
                    
                    g_loss = self.loss_fn.generator_loss(d_fake_output)
                    
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                    self.g_optimizer.step()
                    
                    epoch_g_loss += g_loss.item()
                    epoch_d_loss += d_loss.item()
                    epoch_d_real += torch.mean(d_real_output).item()
                    epoch_d_fake += torch.mean(d_fake_output).item()
                    num_batches += 1
                    
                    # Clear cache every 10 batches
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    torch.cuda.empty_cache()
                    raise
            
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            avg_d_real = epoch_d_real / num_batches
            avg_d_fake = epoch_d_fake / num_batches
            
            self.metrics.update(
                epoch, avg_g_loss, avg_d_loss,
                d_real=avg_d_real, d_fake=avg_d_fake
            )
            
            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"G_Loss: {avg_g_loss:.4f} | D_Loss: {avg_d_loss:.4f} | "
                  f"D(Real): {avg_d_real:.4f} | D(Fake): {avg_d_fake:.4f}")
            
            if (epoch + 1) % save_interval == 0:
                self._save_samples(epoch, latent_dim, n_iter=n_iter)
                self._save_checkpoint(epoch, n_iter=n_iter)
            
            if (epoch + 1) % self.fid_is_interval == 0:
                self._compute_fid_is(latent_dim)
        
            if (epoch + 1) % 100 == 0:
                mode_collapse_info = self.metrics.detect_mode_collapse(fake_images)
                vanishing_info = self.metrics.detect_vanishing_gradients()
                
                print(f"  Mode Collapse: {mode_collapse_info['is_collapsed']} "
                      f"(diversity: {mode_collapse_info['diversity_score']:.4f})")
                print(f"  Vanishing Gradients: {vanishing_info['has_vanishing_gradients']} "
                      f"(g_grad: {vanishing_info['g_gradient_magnitude']:.6f})")
        

        self._compute_fid_is(latent_dim)
        print("\nTraining complete!")

        self._save_samples(epochs - 1, latent_dim, final=True, n_iter=n_iter)
        self._save_checkpoint(epochs - 1, final=True, n_iter=n_iter)
        self.metrics.save_metrics(n_iter=n_iter)
        self.metrics.plot_losses(n_iter=n_iter)
        
        if self.metrics.metrics_history['fid_score']:
            fid_scores = self.metrics.metrics_history['fid_score']
            is_scores = [(m, s) for m, s in zip(
                self.metrics.metrics_history['is_score_mean'],
                self.metrics.metrics_history['is_score_std']
            )]
            self.metrics.plot_quality_metrics(fid_scores=fid_scores, is_scores=is_scores, n_iter=n_iter)
        
        print(f"Results saved to: {self.config['output']['sample_dir']}") 

        # Return minimum FID score for hyperparameter selection
        if self.metrics.metrics_history['fid_score']:
            best_fid = min(self.metrics.metrics_history['fid_score'])
            print(f"Best FID score achieved: {best_fid:.4f}")
            return best_fid
        else:
            return float('inf')
    
    def _save_samples(self, epoch, latent_dim, final=False, n_iter=False):
        """Generate and save sample images"""
        sample_dir = Path(self.config['output']['sample_dir'])
      
        
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(16, latent_dim, device=self.device)
            samples = self.generator(z)
            samples = (samples + 1) / 2  
        
        from torchvision.utils import save_image
        
        if final:
            filename = 'final_samples'
        else:
            filename = f'samples_epoch_{epoch+1}'
        
          
        if n_iter != False:
            filename += f'_iter_{n_iter}.png'
        else:
            filename += '.png'
        
        save_image(samples, sample_dir / filename, nrow=4, padding=2)
        print(f"Saved samples: {filename}")
        
        self.generator.train()
    
    def _save_checkpoint(self, epoch, final=False, n_iter=False):
        """Save model checkpoints"""
        checkpoint_dir = Path(self.config['output'].get('checkpoint_dir', 'results/checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'config': self.config
        }
        
        if final:
            filename = 'final_checkpoint'
            # Also save generator separately for easy loading
            if n_iter != False:
                generator_filename = f'final_generator_iter_{n_iter}.pth'
            else:
                generator_filename = 'final_generator.pth'
            torch.save(self.generator.state_dict(), checkpoint_dir / generator_filename)
            print(f"Saved final generator: {generator_filename}")
        else:
            filename = f'checkpoint_epoch_{epoch+1}'
        
        if n_iter != False:
            filename += f'_iter_{n_iter}.pth'
        else:
            filename += '.pth'
        
        torch.save(checkpoint, checkpoint_dir / filename)
        print(f"Saved checkpoint: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Train DCGAN on malignant images')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file') #we specify the type of loss to use passing the correspondent experiment as an argument
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f'Exception occurred: {e}')

    trainer = GANTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
