# Imputation study outputs

Full-precision result and statistical tables are stored as CSV files. Figures are saved as PNG and PDF. Config and metadata record the run settings and environment.

Each completed evaluation has a CSV checkpoint in checkpoints/. Coverage records successful, failed, and pending evaluations; missing results are not invented.

The notebook has no elapsed-time cutoff. Platform session limits still apply. The ZIP is refreshed at experiment-stage boundaries and after final findings. During a stage, the individual disk checkpoints are the latest record of progress.

To resume in another session, retain these outputs, attach them as Kaggle input, and set RESUME_FROM to the exported ZIP or its study_results/<run-id> directory. Use identical datasets, settings, and package versions. Within the same output directory, disk checkpoints are reused automatically.
