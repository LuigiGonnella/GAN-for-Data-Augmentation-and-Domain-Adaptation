# STEP 3: Final Report
[WARN] Could not generate improved report image: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
Traceback (most recent call last):
  File "D:\PersonalStudy\projects\GAN-for-Data-Augmentation-and-Domain-Adaptation\training\pipeline_classifier.py", line 271, in <module>
    run_full_pipeline(mode)
  File "D:\PersonalStudy\projects\GAN-for-Data-Augmentation-and-Domain-Adaptation\training\pipeline_classifier.py", line 45, in run_full_pipeline
    generate_final_report(baseline_metrics, finetune_results, tuning_results, best_config)
  File "D:\PersonalStudy\projects\GAN-for-Data-Augmentation-and-Domain-Adaptation\training\pipeline_classifier.py", line 179, in generate_final_report
    f.write(f"  - Accuracy: {tuning_results['accuracy'].item():.4f}\n")
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Utente\AppData\Local\Programs\Python\Python312\Lib\site-packages\pandas\core\base.py", line 422, in item
    raise ValueError("can only convert an array of size 1 to a Python scalar")
ValueError: can only convert an array of size 1 to a Python scalar