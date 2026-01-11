# GAN-Based Data Augmentation Guide

## Problem: Imbalanced Dataset
- **Benign images**: 10,000
- **Malignant images**: 1,000
- **Imbalance ratio**: 10:1

## Solution: Generate Synthetic Malignant Images

### Recommended Strategy

Generate **5,000-7,000 synthetic malignant images** for optimal results.

#### Why Not Generate 9,000?

1. **Quality over Quantity**: Too many synthetic images can introduce artifacts
2. **Validation Integrity**: Keep validation/test sets with real images only
3. **Incremental Approach**: Start conservative, increase if needed
4. **Real Data Priority**: Maintain 1:5-7 real:synthetic ratio

### Final Dataset Composition

After augmentation with 6,000 synthetic images:
- **Benign (real)**: 10,000 images
- **Malignant (real)**: 1,000 images
- **Malignant (synthetic)**: 6,000 images
- **Total Malignant**: 7,000 images
- **New ratio**: ~1.4:1 (much better than 10:1)

---

## Step-by-Step Process

### Step 1: Train Your GAN

```bash
# Train GAN on malignant images using hinge loss and spectral norm
python training/train_dcgan.py --config experiments/dcgan_hinge.yaml
```

This will:
- Train for 300 epochs
- Save checkpoints every 50 epochs in `results/checkpoints/dcgan_hinge/`
- Save final generator as `final_generator.pth`
- Generate sample images for visual inspection

**Training Tips**:
- Monitor FID scores (lower is better, aim for < 30)
- Check sample images for quality and diversity
- Watch for mode collapse (all images look similar)
- Training takes ~4-8 hours on GPU depending on hardware

### Step 2: Inspect Generator Quality

After training, examine the generated samples:

```bash
# Check final samples
ls data/synthetic/dcgan_hinge/final_samples.png

# Review FID/IS scores
cat results/gan_metrics/dcgan_hinge/metrics.json
```

**Quality Checklist**:
- [ ] Images look realistic (texture, colors, shapes)
- [ ] Good variety (not all images identical)
- [ ] FID score < 40 (< 30 is excellent)
- [ ] No obvious artifacts (checkerboard patterns, weird colors)
- [ ] Medical features visible (lesion boundaries, color variation)

### Step 3: Generate Preview (Optional but Recommended)

Before generating thousands of images, create a preview grid:

```bash
python training/generate_samples.py \
    --checkpoint results/checkpoints/dcgan_hinge/final_generator.pth \
    --config experiments/dcgan_hinge.yaml \
    --output_dir data/synthetic/gan_v1/malignant \
    --preview_only
```

This creates `preview_grid.png` with 64 samples. Review for quality before proceeding.

### Step 4: Generate Bulk Synthetic Images

Generate 6,000 synthetic malignant images:

```bash
python training/generate_samples.py \
    --checkpoint results/checkpoints/dcgan_hinge/final_generator.pth \
    --config experiments/dcgan_hinge.yaml \
    --num_samples 6000 \
    --output_dir data/processed/augmented/train/malignant \
    --batch_size 64 \
    --preview
```

**Parameters**:
- `--num_samples 6000`: Generate 6,000 images
- `--batch_size 64`: Process 64 images at a time (adjust based on GPU memory)
- `--preview`: Shows preview before starting bulk generation
- `--device cuda`: Use GPU (default)

**Expected Time**: ~10-20 minutes for 6,000 images on GPU

### Step 5: Organize Your Dataset

Your augmented dataset structure:

```
data/processed/augmented/
├── train/
│   ├── benign/              # 10,000 real images
│   │   ├── image_0001.png
│   │   └── ...
│   └── malignant/           # 7,000 total (1,000 real + 6,000 synthetic)
│       ├── real_malignant_0001.png  # Copy 1,000 real images here
│       ├── synthetic_malignant_000000.png
│       ├── synthetic_malignant_000001.png
│       └── ...
├── val/                     # Real images only!
│   ├── benign/
│   └── malignant/
└── test/                    # Real images only!
    ├── benign/
    └── malignant/
```

**Copy real malignant images to augmented directory**:

```bash
# Copy real images alongside synthetic ones
cp data/processed/baseline/train/malignant/*.png \
   data/processed/augmented/train/malignant/

# Rename to distinguish from synthetic
cd data/processed/augmented/train/malignant/
for f in *.png; do mv "$f" "real_$f"; done
```

**Important**: NEVER use synthetic images in validation or test sets!

### Step 6: Train Classifier on Augmented Data

Update your classifier training config to use augmented dataset:

```yaml
# experiments/classifier_gan_augmented.yaml
data:
  train_dir: "data/processed/augmented/train"  # Uses synthetic images
  val_dir: "data/processed/baseline/val"       # Real images only!
  test_dir: "data/processed/baseline/test"     # Real images only!
  image_size: 128
```

Train the classifier:

```bash
python training/train_classifier.py --config experiments/classifier_gan_augmented.yaml
```

### Step 7: Compare Results

Compare classifier performance with and without augmentation:

| Metric | Baseline (No Aug) | With GAN Aug | Improvement |
|--------|-------------------|--------------|-------------|
| Accuracy | ? | ? | ? |
| Sensitivity (Recall) | ? | ? | ? |
| F1-Score | ? | ? | ? |
| AUC-ROC | ? | ? | ? |

Expected improvements:
- **Sensitivity on malignant class**: +5-15%
- **Overall F1-Score**: +3-10%
- **Reduced overfitting**: Better val/train ratio

---

## Alternative Strategies

### Conservative Approach (Start Small)

If unsure about GAN quality:

1. Generate 2,000 synthetic images first
2. Train classifier and evaluate
3. If results improve, generate more (up to 6,000)
4. Compare performance at each step

```bash
# Generate 2,000 images
python training/generate_samples.py \
    --checkpoint results/checkpoints/dcgan_hinge/final_generator.pth \
    --config experiments/dcgan_hinge.yaml \
    --num_samples 2000 \
    --output_dir data/processed/augmented/train/malignant_2k
```

### Aggressive Approach (Maximum Balance)

For maximum balance (1:1 ratio):

```bash
# Generate 9,000 images
python training/generate_samples.py \
    --num_samples 9000 \
    --output_dir data/processed/augmented/train/malignant_full
```

**Caution**: More synthetic images = higher risk of classifier learning GAN artifacts

---

## Quality Assurance

### Visual Inspection

Randomly sample and inspect generated images:

```python
import random
from PIL import Image
from pathlib import Path

# Sample 50 random images
synthetic_dir = Path("data/processed/augmented/train/malignant")
synthetic_images = list(synthetic_dir.glob("synthetic_*.png"))
sample = random.sample(synthetic_images, 50)

# Review each image
for img_path in sample:
    img = Image.open(img_path)
    img.show()
    response = input("Quality OK? (y/n/q): ")
    if response == 'q':
        break
```

### Automated Quality Checks

Check for common issues:

```python
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

def check_image_statistics(image_dir):
    """Check if synthetic images have realistic statistics"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=64, num_workers=4)
    
    all_images = []
    for images, _ in loader:
        all_images.append(images)
    
    all_images = torch.cat(all_images, dim=0)
    
    # Check statistics
    mean = all_images.mean(dim=[0, 2, 3])
    std = all_images.std(dim=[0, 2, 3])
    
    print(f"Mean RGB: {mean}")
    print(f"Std RGB: {std}")
    print(f"Min pixel: {all_images.min()}")
    print(f"Max pixel: {all_images.max()}")
    
    # Check for artifacts
    variance = all_images.var(dim=[0, 2, 3])
    print(f"Variance: {variance}")
    
    return mean, std
```

### Diversity Check

Ensure generated images are diverse:

```python
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

def check_diversity(images_tensor):
    """Check diversity using PCA and pairwise distances"""
    # Flatten images
    images_flat = images_tensor.reshape(images_tensor.size(0), -1).numpy()
    
    # PCA
    pca = PCA(n_components=50)
    pca_features = pca.fit_transform(images_flat)
    
    # Pairwise distances
    distances = pairwise_distances(pca_features)
    avg_distance = distances.mean()
    
    print(f"Average pairwise distance: {avg_distance}")
    print(f"PCA variance explained: {pca.explained_variance_ratio_.sum()}")
    
    if avg_distance < 10:
        print("⚠️ Warning: Low diversity detected (possible mode collapse)")
    else:
        print("✓ Good diversity")
```

---

## Troubleshooting

### Problem: Generated Images Look Unrealistic

**Solutions**:
- Train GAN for more epochs (300+ recommended)
- Try different loss function (hinge loss recommended)
- Use spectral normalization (reduces artifacts)
- Increase discriminator capacity (more filters)
- Check if training data is sufficient and varied

### Problem: Mode Collapse (All Images Similar)

**Solutions**:
- Use minibatch discrimination
- Try different GAN architecture (StyleGAN, Progressive GAN)
- Adjust learning rates (lower generator LR)
- Increase diversity in training data
- Use unrolled GAN or other mode collapse prevention techniques

### Problem: Classifier Doesn't Improve

**Possible Causes**:
- Synthetic images have artifacts classifier learns to exploit
- Too many synthetic images overwhelming real data
- GAN didn't learn real distribution well

**Solutions**:
- Reduce number of synthetic images
- Retrain GAN with better architecture/hyperparameters
- Use mix-up or label smoothing during classifier training
- Try different augmentation techniques (rotation, color jittering)

---

## Advanced: Multiple GAN Variants

Generate from multiple trained GANs for diversity:

```bash
# Train multiple GANs with different losses
python training/train_dcgan.py --config experiments/dcgan_hinge.yaml
python training/train_dcgan.py --config experiments/dcgan_wasserstein.yaml
python training/train_dcgan.py --config experiments/dcgan_bce.yaml

# Generate 2,000 from each
python training/generate_samples.py \
    --checkpoint results/checkpoints/dcgan_hinge/final_generator.pth \
    --num_samples 2000 \
    --output_dir data/processed/augmented/train/malignant

python training/generate_samples.py \
    --checkpoint results/checkpoints/dcgan_wasserstein/final_generator.pth \
    --num_samples 2000 \
    --output_dir data/processed/augmented/train/malignant

python training/generate_samples.py \
    --checkpoint results/checkpoints/dcgan_bce/final_generator.pth \
    --num_samples 2000 \
    --output_dir data/processed/augmented/train/malignant
```

This provides 6,000 total images from diverse generators.

---

## Quick Reference Commands

```bash
# 1. Train GAN
python training/train_dcgan.py --config experiments/dcgan_hinge.yaml

# 2. Generate preview
python training/generate_samples.py \
    --checkpoint results/checkpoints/dcgan_hinge/final_generator.pth \
    --config experiments/dcgan_hinge.yaml \
    --output_dir data/processed/augmented/train/malignant \
    --preview_only

# 3. Generate 6,000 images
python training/generate_samples.py \
    --checkpoint results/checkpoints/dcgan_hinge/final_generator.pth \
    --config experiments/dcgan_hinge.yaml \
    --num_samples 6000 \
    --output_dir data/processed/augmented/train/malignant \
    --batch_size 64

# 4. Train classifier on augmented data
python training/train_classifier.py --config experiments/classifier_gan_augmented.yaml
```

---

## Best Practices Summary

✅ **DO**:
- Generate 5,000-7,000 synthetic images (conservative approach)
- Use only real images for validation and testing
- Visually inspect generated images before bulk generation
- Monitor GAN training with FID/IS scores
- Compare classifier performance with/without augmentation
- Keep real:synthetic ratio at least 1:5-7

❌ **DON'T**:
- Use synthetic images in validation/test sets
- Generate more synthetic than 10x real images
- Skip quality inspection of generated images
- Trust low-quality GAN outputs
- Ignore mode collapse warnings
- Forget to save GAN checkpoints regularly

---

## Expected Results

With high-quality synthetic data:
- **Malignant class sensitivity**: Improve by 5-15%
- **F1-Score**: Increase by 3-10%
- **Reduced overfitting**: Better generalization to test set
- **Training stability**: More balanced gradients

Success metric: **Classifier performs better on real test set** (not just synthetic data)
