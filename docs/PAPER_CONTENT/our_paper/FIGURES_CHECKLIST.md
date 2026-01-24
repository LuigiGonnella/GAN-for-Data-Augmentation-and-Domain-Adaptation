# Paper Figures Checklist

This document lists all figures referenced in the paper and their expected locations.

## Figures Added to Paper

### 1. GAN Training and Evaluation

**Figure 1: Training Dynamics (fig:gan_training_curves)**
- **Path**: `../../results/gan_metrics/cdcgan_hinge/plots/training_curves.png`
- **Content**: Generator/discriminator losses and confidence scores over 300 epochs
- **Status**: ⚠️ Need to verify this file exists or create it

**Figure 2: FID Score Progression (fig:fid_progression)**
- **Path**: `../../results/gan_metrics/cdcgan_hinge/plots/fid_progression.png`
- **Content**: FID scores tracked across training epochs
- **Status**: ⚠️ Need to verify this file exists or create it

**Figure 3: Synthetic Samples Grid (fig:synthetic_samples)**
- **Path**: `../../data/synthetic/cdcgan_hinge/samples_grid.png`
- **Content**: Grid of 64 synthetic malignant lesions
- **Status**: ⚠️ Need to create this grid from generated samples

**Figure 4: Real vs Synthetic Comparison (fig:real_vs_synthetic)**
- **Path**: `../../data/synthetic/cdcgan_hinge/real_vs_synthetic_comparison.png`
- **Content**: Side-by-side comparison of real and synthetic samples
- **Status**: ⚠️ Need to create this comparison image

### 2. Classifier Evaluation

**Figure 5: Optimization Report (fig:optimization_report)**
- **Path**: `../../results/classifier_on_baseline/final_report/optimization_report.png`
- **Content**: Performance progression across 3 training stages
- **Status**: ✅ EXISTS - Already generated

**Figure 6: Confusion Matrix (fig:confusion_baseline)**
- **Path**: `../../results/classifier_on_baseline/ft_ht/plots/confusion_matrix.png`
- **Content**: Confusion matrix for final baseline classifier
- **Status**: ✅ EXISTS - Already generated

**Figure 7: ROC Curve (fig:roc_curve)**
- **Path**: `../../results/classifier_on_baseline/ft_ht/plots/roc_curve.png`
- **Content**: ROC curve with optimal threshold marked
- **Status**: ⚠️ Need to verify this file exists

**Figure 8: Training Curves (fig:classifier_training)**
- **Path**: `../../results/classifier_on_baseline/ft/plots/training_curves.png`
- **Content**: Loss and accuracy curves during fine-tuning
- **Status**: ⚠️ Need to verify this file exists

**Figure 9: Hyperparameter Search Results (fig:hyperparam_search)**
- **Path**: `../../results/classifier_on_baseline/ft_ht/plots/hyperparameter_search_results.png`
- **Content**: Distribution of validation recall across configurations
- **Status**: ⚠️ Need to create this visualization

### 3. Architecture Diagram

**Figure 10: Architecture Overview (fig:architecture)**
- **Path**: `architecture_diagram.png`
- **Content**: Diagram showing GAN generator and discriminator architecture
- **Status**: ⚠️ Need to create architecture diagram (can be simple schematic)

## Required Actions

### Priority 1: Verify Existing Plots
Check which plots already exist in the results folder from previous training runs.

### Priority 2: Create Missing Figures

1. **Synthetic samples grid**: Use existing generated images to create 8×8 grid
2. **Real vs synthetic comparison**: Show 4 real + 4 synthetic in two rows
3. **Hyperparameter search visualization**: Box plot of validation metrics
4. **Architecture diagram**: Simple schematic (can use LaTeX tikz or external tool)

### Priority 3: Generate Missing Training Plots

If training curves don't exist:
- Re-run plotting scripts on saved checkpoints/logs
- Or use saved training logs to regenerate plots

## Notes

- All relative paths in LaTeX are from `docs/PAPER_CONTENT/our_paper/`
- Images should be high resolution (300 DPI for publication)
- Consider creating placeholder figures if originals unavailable
- Architecture diagram can be created with draw.io, PowerPoint, or tikz

## Compilation Instructions

Once all figures are ready:

```bash
cd docs/PAPER_CONTENT/our_paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Expected Paper Length

With all figures and expanded content:
- Current structure targets 6-8 pages (excluding references)
- Figures will add ~2-3 pages
- Total estimated: 8-10 pages with references
- If too long, can reduce discussion/future work sections
