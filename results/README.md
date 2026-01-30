
## Results Folder Structure

This folder contains all output, checkpoints, and evaluation results generated during experiments and training runs. Below is a detailed description of its organization and the purpose of each subfolder:

### Folder Overview

- **checkpoints/**
	- Stores model checkpoints for various training runs. Each subfolder is named according to the experiment and model type (e.g., `classifier_on_augmented_cDCGAN_RESNET50`, `classifier_on_baseline_RESNET18`). These contain saved weights and states for resuming or analyzing training.

- **domain_shift/**
	- Contains results and metrics related to domain shift evaluation experiments. This includes outputs from scripts and evaluations that measure how well models adapt to new domains.

- **gan_metrics/**
	- Stores evaluation metrics for GAN models, such as FID, IS, or other custom metrics. Useful for comparing GAN performance across different experiments.

- **scratch_classifier_on_augmented_ALEXNET/**, **scratch_classifier_on_augmented_RESNET18/**, **scratch_classifier_on_baseline_ALEXNET/**, **scartch_classifier_on_baseline_RESNET18/**
	- These folders contain results and checkpoints for classifiers trained from scratch on either augmented or baseline datasets, using different architectures (ALEXNET, RESNET18). Each folder includes logs, metrics, and model weights specific to the experiment.

### Usage

Use these folders to:
- Retrieve trained model weights for further analysis or inference.
- Access evaluation metrics and logs for reporting or comparison.
- Track experiment outputs and ensure reproducibility.

### Notes

- Folder names are structured to indicate the experiment type, dataset, and model architecture for clarity.
- Some folders may be empty if the corresponding experiment has not been run yet.

For more details on experiment configurations, refer to the `experiments/` folder and associated YAML files.
