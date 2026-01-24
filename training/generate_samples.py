"""
Generate large batches of synthetic images from trained GAN
"""

import torch
import torch.nn as nn
from torchvision.utils import save_image
from pathlib import Path
import argparse
import yaml
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gan.DCGAN import DCGANGenerator
from models.gan.cDCGAN import ConditionalDCGANGenerator
from config.utils import load_config

class ImageGenerator:
    def __init__(self, generator_path, config, device):
        """
        Initialize generator for bulk image creation
        
        Args:
            generator_path: Path to saved generator checkpoint (.pth file)
            config_path: Path to training config (to get latent_dim, etc.)
            device: 'cuda' or 'cpu'
        """
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load config
        self.config = config
        
        # Build generator based on type
        gen_config = self.config['model']['generator']
        model_type = self.config['model'].get('type', 'dcgan')  # Default to DCGAN if not specified
        
        if model_type == 'cdcgan':
            self.generator = ConditionalDCGANGenerator(
                input_dim=gen_config['latent_dim'],
                num_classes=gen_config.get('num_classes', 2),
                n1=gen_config['n1'],
                channels=gen_config['channels']
            ).to(self.device)
            self.is_conditional = True
        else:  # DCGAN
            self.generator = DCGANGenerator(
                input_dim=gen_config['latent_dim'],
                n1=gen_config['n1'],
                channels=gen_config['channels']
            ).to(self.device)
            self.is_conditional = False
        
        # Load trained weights
        checkpoint = torch.load(generator_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'generator' in checkpoint:
            self.generator.load_state_dict(checkpoint['generator'])
        else:
            self.generator.load_state_dict(checkpoint)
        
        self.generator.eval()
        self.latent_dim = gen_config['latent_dim']
        
        print(f"Generator loaded from: {generator_path}")
        print(f"Latent dimension: {self.latent_dim}")
    
    def generate_batch(self, batch_size=64, label=1):
        """
        Generate a batch of images
        
        Args:
            batch_size: Number of images to generate
            label: Class label for conditional GAN (default 1 for malignant)
            
        Returns:
            Tensor of images [B, C, H, W] in range [0, 1]
        """
        with torch.no_grad():
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            
            if self.is_conditional:
                labels = torch.full((batch_size,), label, dtype=torch.long, device=self.device)
                images = self.generator(z, labels)
            else:
                images = self.generator(z)
            
            # Convert from [-1, 1] to [0, 1]
            images = (images + 1) / 2
            images = torch.clamp(images, 0, 1)
        return images
    
    def generate_and_save(self, num_samples, output_dir, batch_size=64, prefix='synthetic_malignant', label=1):
        """
        Generate and save individual images to disk
        
        Args:
            num_samples: Total number of images to generate
            output_dir: Directory to save images
            batch_size: Batch size for generation (larger = faster but more memory)
            prefix: Prefix for image filenames
            label: Class label for conditional GAN (default 1 for malignant)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating {num_samples} images...")
        print(f"Output directory: {output_dir}")
        print(f"Batch size: {batch_size}")
        if self.is_conditional:
            print(f"Label: {label} ({'malignant' if label == 1 else 'benign'})")
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        image_idx = 0
        
        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            # Calculate batch size for last batch
            current_batch_size = min(batch_size, num_samples - image_idx)
            
            # Generate batch
            images = self.generate_batch(current_batch_size, label=label)
            
            # Save individual images
            for img in images:
                filename = f"{prefix}_{image_idx:06d}.jpg"
                save_image(img, output_dir / filename)
                image_idx += 1
            
            # Clear cache periodically
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        print(f"\n✓ Successfully generated {num_samples} images")
        print(f"  Saved to: {output_dir}")
        return output_dir
    
    def generate_preview_grid(self, num_images=64, output_path=None):
        """
        Generate a grid of sample images for quality inspection
        
        Args:
            num_images: Number of images in preview (will be arranged in grid)
            output_path: Path to save preview grid
        """
        images = self.generate_batch(num_images)
        
        if output_path is None:
            output_path = "generated_preview.png"
        
        # Calculate grid dimensions (square grid)
        import math
        nrow = int(math.sqrt(num_images))
        
        save_image(images, output_path, nrow=nrow, padding=2)
        print(f"Preview grid saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic images from trained GAN'
    )

    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to config YAML file') 

    parser.add_argument(
        '--preview',
        action='store_true',
        help='Generate preview grid before bulk generation'
    )
    parser.add_argument(
        '--preview_only',
        action='store_true',
        help='Only generate preview grid, skip bulk generation'
    )
    
    args = parser.parse_args()
    
    # Check if config exists
    if not Path(args.config).exists():
        print(f"Error: Config not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize generator
    generator = ImageGenerator(
        config['generation']['checkpoint'],
        config,
        device=device
    )
    
    # Generate preview if requested
    if args.preview or args.preview_only:
        output_dir = Path(config['generation']['sample_dir'])
        preview_dir = Path(config['generation']['preview_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / "preview_grid.png"
        generator.generate_preview_grid(
            num_images=64,
            output_path=preview_path
        )
        
        if args.preview_only:
            print("\nPreview generated. Exiting (use --preview without --preview_only to continue)")
            return
        
        print("\nPreview generated. Continuing with bulk generation...")
        input("Press Enter to continue or Ctrl+C to cancel...")
    
    num_samples = config['generation']['num_samples']
    output_dir = config['generation']['sample_dir']
    
    # Generate bulk images
    generator.generate_and_save(
        num_samples=num_samples,
        output_dir=output_dir,
        batch_size=config['generation']['batch_size'],
        prefix='synthetic_malignant'
    )
    
    print("\n" + "="*60)
    print("Generation complete!")
    print("="*60)
    print(f"Total images generated: {num_samples}")
    print(f"Location: {output_dir}")
    print("\nNext steps:")
    print("1. Inspect generated images for quality")
    print("2. Combine with real malignant images for training")
    print("3. Train classifier on augmented dataset")


if __name__ == '__main__':
    main()
