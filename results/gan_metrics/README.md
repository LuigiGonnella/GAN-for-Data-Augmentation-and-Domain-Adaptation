
## GAN Metrics Folder Structure

This folder contains evaluation results and metrics for various GAN experiments conducted during data augmentation and domain adaptation studies. Each subfolder corresponds to a specific GAN configuration, experiment, or hyperparameter tuning run.

### Organization

- Each subfolder is named according to the GAN type, loss function, and/or hyperparameter setting (e.g., `cdcgan_bce_ht_lr`, `cdcgan_hinge_final`).
- Inside each subfolder, you will typically find:
	- Metric files (e.g., FID, IS, precision, recall)
	- Logs and plots summarizing GAN performance
	- Configuration details for the experiment

### Notes on Hyperparameter Tuning

- In some subfolders, hyperparameter tuning was intentionally interrupted due to consistently poor performance or non-convergent results. These folders may contain partial logs or incomplete metric files, and are retained for transparency and reproducibility.
- Interrupted runs are clearly indicated in the folder name or by notes in the logs/config files.

### Usage

- Use this folder to compare GAN models and configurations based on quantitative metrics.
- Refer to logs and plots for insights into training stability and output quality.
- Review interrupted runs to understand which configurations were not successful and why.

### Additional Information

- For details on experiment setup and configuration, see the corresponding YAML files in the `experiments/` folder.
- For model checkpoints, refer to the `results/checkpoints/` directory.
