# Classifier-Based Evaluation in GAN Hyperparameter Tuning

## Overview

The GAN hyperparameter tuning (both DCGAN and cDCGAN) has been enhanced to include classifier-based evaluation alongside FID scores. This provides a more comprehensive assessment of generated samples by measuring how well they are classified by a pre-trained medical image classifier.

## Architecture

### Modular Structure

The classifier evaluation functionality has been separated into a dedicated module for better code organization and reusability:

**Core Module:**
- `evaluation/gan_classifier_evaluation.py` - Contains all classifier evaluation logic
  - `SyntheticDataset` class
  - `generate_samples_for_evaluation()` function
  - `evaluate_with_classifier()` function
  - `compute_combined_score()` utility function
  - Constants (`CLASSIFIER_PATH`, `CLASSIFIER_CONFIG`)

**Usage in:**
- `training/hyperparameter_tuning_DCGAN.py`
- `training/hyperparameter_tuning_cDCGAN.py`

This modular design allows:
- Easy reuse across different GAN architectures
- Centralized maintenance of evaluation logic
- Simple extension for future GAN variants
- Clear separation of concerns

## Key Components

### 1. Core Evaluation Module (`evaluation/gan_classifier_evaluation.py`)

**Classes:**
- `SyntheticDataset`: PyTorch Dataset wrapper for synthetic images

**Functions:**
- `generate_samples_for_evaluation()`: Generates synthetic samples from trained generator
  - Supports both DCGAN and cDCGAN (via `gan_type` parameter)
  - Configurable sample count and batch size
  - Returns list of PIL Images
  
- `evaluate_with_classifier()`: Evaluates samples with pre-trained classifier
  - Loads classifier from checkpoint
  - Processes synthetic images
  - Returns precision, recall, accuracy, F1-score
  
- `compute_combined_score()`: Calculates weighted score
  - Combines FID and recall metrics
  - Configurable weights
  - Normalizes FID across iterations

**Constants:**
- `CLASSIFIER_PATH`: Path to pre-trained classifier
- `DEFAULT_CLASSIFIER_CONFIG`: Default classifier configuration (ResNet50)
- `DEFAULT_NUM_SAMPLES`: Default number of samples (500)
- `DEFAULT_BATCH_SIZE`: Default batch size (64)

**Configuration:**
The `evaluate_with_classifier()` function accepts a `classifier_config` parameter that can be:
- `None`: Uses the default built-in config (ResNet50)
- `dict`: Directly provides a configuration dictionary
- `str`: Path to a YAML config file (e.g., `'experiments/classifier_baseline_ft_ht.yaml'`)

This allows flexibility in evaluating with different classifier architectures or configurations.

## Key Changes

### 1. Classifier Evaluation Integration

**Evaluation Process:**
1. After each GAN training iteration, generate 500 synthetic malignant samples
2. Load pre-trained classifier from `results/classifier_on_baseline/ft_ht/classifier.pth`
3. Evaluate classifier performance on synthetic samples
4. Return precision, recall, accuracy, and F1-score

### 2. Combined Scoring Metric

**Selection Strategy:**
The best model is now selected using a combined score that balances:
- **60% weight on Recall** (most important for medical imaging)
- **40% weight on normalized FID** (image quality)

**Formula:**
```python
combined_score = compute_combined_score(
    fid_score, recall, fid_scores_list,
    recall_weight=0.6, fid_weight=0.4
)
# Result: -0.6 * recall + 0.4 * normalized_fid
# (Lower score is better; recall is negated because we want to maximize it)
```

**Rationale:**
- **Recall prioritization**: For malignancy detection, missing a cancer (false negative) is more critical than false alarms
- **FID as quality check**: Ensures images are realistic enough to be useful
- This approach aligns with medical imaging priorities while maintaining generation quality

### 3. Updated Functions in Hyperparameter Tuning Files

**Modified:**
- `tune_lr()`: Now evaluates with classifier and returns metrics
- `run_lr_tuning()`: Combines FID and classifier metrics for LR selection
- `tune_with_hyperparams()`: Evaluates each hyperparameter combination with classifier
- `run_hyperparameter_tuning()`: Uses combined scoring for final model selection
- `run_best_config()`: Accepts `use_classifier` parameter

### 4. New Command Line Options

## CLI Parameters

**Available Flags:**
```bash
--config PATH                 # Required: Path to GAN config YAML file
--only_lr                     # Only run learning rate tuning and exit
--not_lr                      # Skip LR tuning, use values from config
--no_classifier               # Disable classifier evaluation (FID only)
--classifier_config PATH      # Path to classifier config YAML (optional)
```

**Usage examples:**
```bash
# DCGAN - With classifier evaluation (default)
python training/hyperparameter_tuning_DCGAN.py --config experiments/dcgan_bce.yaml

# DCGAN - With custom classifier config
python training/hyperparameter_tuning_DCGAN.py --config experiments/dcgan_bce.yaml \
    --classifier_config experiments/classifier_baseline_ft.yaml

# DCGAN - Without classifier evaluation
python training/hyperparameter_tuning_DCGAN.py --config experiments/dcgan_bce.yaml --no_classifier

# cDCGAN - Only LR tuning with custom classifier
python training/hyperparameter_tuning_cDCGAN.py --config experiments/cdcgan_hinge.yaml \
    --only_lr --classifier_config experiments/classifier_baseline_ft_ht.yaml

# cDCGAN - Full tuning without classifier
python training/hyperparameter_tuning_cDCGAN.py --config experiments/cdcgan_mse.yaml --no_classifier
```

## Output Changes

### Enhanced CSV Results

**lr_tuning_results.csv** now includes:
- `fid_score`
- `g_lr`, `d_lr`
- `classifier_precision`
- `classifier_recall`
- `classifier_accuracy`
- `classifier_f1`

**hyperparameter_tuning_results.csv** now includes:
- All hyperparameters (latent_dim, n_layers, etc.)
- `fid_score`
- `classifier_precision`
- `classifier_recall`
- `classifier_accuracy`
- `classifier_f1`

### Console Output

The tuning process now prints:
```
[1/5] Testing combination:
  Generator LR: 2e-04
  Discriminator LR: 2e-04
  Generating 500 samples for classifier evaluation...
  Classifier metrics - Precision: 0.7532, Recall: 0.8421
  → FID Score: 245.3421

Using combined scoring: -0.6*recall + 0.4*normalized_fid

Best learning rates (FID: 245.34):
  Generator LR: 2e-04
  Discriminator LR: 2e-04

Classifier Performance on Generated Samples:
  Recall: 0.8421
  Precision: 0.7532
  F1-Score: 0.7952
  Accuracy: 0.8120
```

## Technical Details

### Classifier Configuration

**Configuration Options:**

The classifier configuration can be customized via the `--classifier_config` CLI parameter:

```bash
# Use default built-in config (ResNet50)
python training/hyperparameter_tuning_DCGAN.py --config experiments/dcgan_bce.yaml

# Use custom classifier config from YAML file
python training/hyperparameter_tuning_DCGAN.py --config experiments/dcgan_bce.yaml \
    --classifier_config experiments/classifier_baseline_ft_ht.yaml

# For cDCGAN
python training/hyperparameter_tuning_cDCGAN.py --config experiments/cdcgan_hinge.yaml \
    --classifier_config experiments/classifier_baseline_ft.yaml
```

**Default Settings:**
- Model: ResNet50
- Classifier path: `results/classifier_on_baseline/ft_ht/classifier.pth`
- Batch size: 32
- Image size: 224x224
- All generated samples labeled as malignant (class 1)

**Custom Config Format (YAML):**
```yaml
model:
  name: resnet50  # or other supported models
training:
  params:
    batch_size: 32
# Other classifier-specific settings...
```

**Preprocessing:**
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### GAN Type Support

The evaluation module supports both:
- **DCGAN**: Standard unconditional GAN
- **cDCGAN**: Conditional GAN with class labels
  - Automatically generates malignant class (label=1)
  - Requires `num_classes` parameter

### Error Handling

- Automatically falls back to FID-only if classifier not found
- Continues evaluation even if classifier fails for individual iterations
- Prints warnings but doesn't halt the entire tuning process
- Graceful degradation ensures robustness

## Benefits for Medical Imaging

1. **Clinically-Relevant Metrics**: Recall and precision directly measure how well synthetic samples mimic real malignant cases
2. **Quality + Utility**: FID measures visual quality, classifier metrics measure downstream task utility
3. **Aligns with Clinical Priorities**: 60% weight on recall reflects the importance of not missing malignancies
4. **Comprehensive Evaluation**: Multiple metrics provide better confidence in model selection
5. **Modular Design**: Easy to adapt for different medical imaging tasks or classifiers

## Limitations & Considerations

1. **Classifier Dependency**: Requires pre-trained classifier (currently assumes it exists)
2. **Computational Cost**: Adds ~1-2 minutes per iteration for sample generation and evaluation
3. **Single-Class Evaluation**: Only evaluates malignant samples (assumes GAN generates only malignant)
4. **Fixed Weights**: 60/40 split is hardcoded in calls; can be customized via function parameters
5. **Memory Usage**: Generating 500 samples requires GPU memory; adjust `num_samples` if needed

## Future Improvements

1. **Configurable Weights**: Allow users to specify recall/FID balance via config file
2. **Multi-Class Support**: Evaluate both benign and malignant generation
3. **Domain Adaptation Metrics**: Add metrics for domain shift evaluation
4. **Threshold Tuning**: Use optimal threshold from classifier instead of 0.5
5. **Parallel Evaluation**: Generate and evaluate samples in parallel to reduce time
6. **Alternative Classifiers**: Support for multiple pre-trained classifiers

## Related Files

- `evaluation/gan_classifier_evaluation.py` - **Core evaluation module**
- `training/hyperparameter_tuning_DCGAN.py` - DCGAN tuning implementation
- `training/hyperparameter_tuning_cDCGAN.py` - cDCGAN tuning implementation
- `evaluation/metrics.py` - Classifier evaluation functions
- `models/classifier/classifier.py` - Classifier model
- `models/gan/DCGAN.py` - DCGAN generator architecture
- `models/gan/cDCGAN.py` - cDCGAN generator architecture
