lr tuned on lowest FID + visual check
rest tuned on recall + fid


baseline_classifier_domain_eval WORKFLOW:

Step 1: Prepare Domain Adaptation Data
python scripts/prepare_domain_adaptation_data.py --synthetic-malignant data/synthetic/FINAL_cdcgan_hinge/generation --real-benign-train data/processed/baseline/train/benign --test-images-dir data/processed/baseline/test


Step 2: Run Baseline Domain Shift Evaluation
python training/baseline_classifier_domain_eval.py --config experiments/domain_shift_eval.yaml