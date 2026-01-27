lr tuned on lowest FID + visual check
rest tuned on recall + fid


baseline_classifier_domain_eval WORKFLOW:

Step 1: Prepare Domain Adaptation Data
python scripts/prepare_domain_adaptation_data.py --baseline-dir data/processed/baseline --synthetic-malignant data/synthetic/cdcgan_hinge_final --output-dir data/processed/domain_adaptation


Step 2: Run Baseline Domain Shift Evaluation
python training/baseline_classifier_domain_eval.py --config experiments/domain_shift_eval.yaml


pipeline_classifier.py WORKFLOW:
python training/pipeline_classifier.py --data_type baseline --model alexnet