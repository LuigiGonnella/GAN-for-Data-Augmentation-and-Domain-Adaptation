SYNTHETIC IMAGES CHOICE:
we decided to generate 3000 synthetic images because:

    We more than double the minority class (1400 → 4400)
    We improve class balance significantly (1:7.5 → 1:2.4)
    We maintain a reasonable proportion of real data (32% real, 68% synthetic in malignant class)
    Not overwhelming the classifier with synthetic-only data

WORKFLOW: es dcgan_hinge
1) Generate synthetic samples:
    python training/generate_samples.py --config experiments/dcgan_hinge.yaml --preview

2) Populate augmented folder:
    python scripts/populate_augmented.py --version dcgan_hinge

3) Run classifier pipeline:
    python training/pipeline_classifier.py --data_type augmented