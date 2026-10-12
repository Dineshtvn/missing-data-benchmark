# Missing-Data Strategies under Varying Feature Importance and Missingness Mechanisms in Tabular Classification

Does the best imputation method depend on **how important the missing feature is**, and on **why it is missing**? This repository contains the full experimental code and the exported results for a controlled study of 8 imputation strategies, 8 missingness mechanisms, 3 feature-importance tiers and 10 tabular datasets.

> **Headline result:** differences between imputers are small (mean macro-F1 drops below 0.01 at 20% missingness), no method is a universal winner, and none of the 224 pairwise comparisons survives Holm correction. The study reports this honestly instead of declaring a champion.

---

## Contents

- [Research questions](#research-questions)
- [Study design](#study-design)
- [Key findings](#key-findings)
- [Repository structure](#repository-structure)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Output files](#output-files)
- [Limitations](#limitations)
- [License](#license)

## Research questions

RQ1. How does macro-F1 loss vary across representative feature-importance tiers?
RQ2. How do strategy comparisons vary across missingness mechanisms and rates, and how does reconstruction error relate to predictive loss?
RQ3. How do comparisons change under train/test missingness mismatch and the restricted classifier, importance-definition, and natural-missingness checks?

## Study design

| Component | Details |
|---|---|
| **Controlled datasets (7)** | Breast Cancer Wisconsin, Banknote Authentication, Digits, Spambase, MAGIC Gamma Telescope, Ionosphere, Phoneme |
| **Natural-missingness datasets (3)** | Adult Census Income, Horse Colic, Credit Approval |
| **Imputers (8)** | mean, median, mode, median + missing-indicator, KNN (k=5, distance-weighted), iterative (BayesianRidge), iterative (random forest), native (XGBoost handles NaN directly) |
| **Mechanisms (8 + 2 block)** | `mcar`, `mar_redundant`, `mar_unrelated`, `mnar_mild`, `mnar_moderate`, `mnar_strong`, `mnar_censor`, `label`, plus `block_top` / `block_random` (two features missing together) |
| **Missing rates** | 10%, 20%, 40% (MCAR, strong MNAR, label); 20% for the other mechanisms |
| **Importance tiers** | Top / middle / bottom, ranked on training data by permutation importance (Random Forest reference) with a mutual-information robustness basis; feature redundancy measured by cross-validated R² |
| **Primary classifier** | XGBoost (GPU). Robustness classifiers: logistic regression, random forest, HistGradientBoosting, MLP |
| **Train/test mismatch (7 scenarios)** | matched, train-only, test-only, rate shift, mechanism shift, matched MNAR, matched label |
| **Validation** | 3-fold × 2 repeats, 2 mask seeds, stratified subsampling to at most 1,500 rows per dataset |
| **Primary metric** | Macro-F1 drop versus the complete-data baseline (negative drops are retained) |
| **Total evaluations** | 31,884 planned, 31,884 completed (100%) |

**Leakage controls:** categorical vocabularies, scalers, imputers, importance rankings and missingness generators are all fitted on training folds only. Mask generators are unit-tested (`mask_unit_tests.csv`, `mask_checks.csv`) for realised versus requested rates.

**Excluded datasets (see `dataset_audit.csv`):** Wine (fewer than 300 rows) and Pima (no natural missingness).

## Key findings

At 20% target-feature missingness, averaged equally over the 7 controlled datasets:

| Mechanism | Lowest mean macro-F1 drop | Mean drop |
|---|---|---|
| MCAR | KNN | 0.0032 |
| MAR (redundant driver) | RF-iterative | 0.0035 |
| MAR (unrelated driver) | KNN | 0.0033 |
| MNAR mild | RF-iterative | 0.0033 |
| MNAR moderate | KNN | 0.0037 |
| MNAR strong | native | 0.0018 |
| MNAR censor | native | 0.0019 |
| Label-dependent | native | -0.0089 |

1. **No universal winner.** Winners vary by mechanism, and the lowest-drop method is never clearly separated from the rest.
2. **Differences are statistically weak.** No Friedman test is significant after Holm correction across mechanisms, and 0 of 224 Holm-adjusted pairwise Wilcoxon comparisons are significant.
3. **The best imputer changes with feature importance.** The winning method differs between importance tiers on all 7 controlled datasets (`winner_switches.csv`).
4. **Reconstruction error is not a selection criterion.** Spearman correlation between imputation RMSE and downstream drop is about -0.011 (descriptive only).
5. **Runtime is rarely a deciding factor**, with one exception: RF-iterative costs about 0.84 s per evaluation versus 0.14 to 0.16 s for every other method.
6. **Train/test mismatch results are dataset-sensitive** (5 datasets); see `mismatch_summary.csv` and `effect_contrasts.csv` for uncertainty.

> Aggregates describe this experimental setup. They do not establish a universal ranking. Validate the choice on your own missingness process.

## Repository structure

```
.
├── implementation.ipynb          # Full experiment (Kaggle notebook)
├── results/
    |__ figures
│   ├── key_findings.md
│   ├── config.json               # Exact run settings
│   ├── metadata.json             # Versions, GPU, runtime
│   ├── results.csv               # Full-precision per-evaluation results
│   ├── *.csv                     # Statistical and summary tables
│   ├── figures/                  # PNG + PDF figures
│   └── importance/           # Per-dataset, per-fold feature importance
└── README.md
```

The 30,900+ per-evaluation checkpoint files are not stored in the repository. They are available in the release archive: `<link to GitHub Release / Zenodo>`.

## Reproducing the experiments

**Environment used for the reported run**

- Python 3.13, NumPy 2.1.3, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.6.1, XGBoost 3.4.1, PyTorch 2.11 (CUDA 12.8), statsmodels 0.15.0
- Kaggle, 2× Tesla T4 GPUs, about 3.4 hours (12,103 s)

**Steps**

1. Upload `implementation.ipynb` to Kaggle and enable a GPU accelerator.
2. Attach the datasets (Banknote, Spambase, MAGIC, Adult, Horse Colic); Breast Cancer and Digits load from scikit-learn, and Ionosphere, Phoneme and Credit Approval load from OpenML (`ALLOW_OPENML = True`).
3. Run all cells. Outputs go to `/kaggle/working/study_results/<run-id>/`, and a ZIP is refreshed at stage boundaries.

**Key switches (first cell)**

```python
FAST_MODE = False     # True = small smoke test; False = full study
USE_GPU = True
SEED = 2026
MASK_SEEDS = [17, 47]
MAX_ROWS = 1500
RESUME_FROM = None    # path to a previous study_results/<run-id> or exported ZIP
```

**Resuming:** completed evaluations are checkpointed to disk. In a new session, attach the previous outputs as input and set `RESUME_FROM`. Dataset contents, settings and package versions must match.

## Output files

| File | Description |
|---|---|
| `results.csv`, `plan.csv`, `coverage.csv` | Every evaluation, the planned grid, and completed/failed/pending status |
| `performance_summary.csv`, `baselines.csv`, `multimodel_summary.csv` | Aggregated performance and complete-data baselines |
| `average_ranks.csv`, `friedman.csv`, `wilcoxon_holm.csv` | Rank-based tests with Holm correction |
| `bootstrap_intervals.csv`, `effect_contrasts.csv` | Bootstrap CIs and practical-effect contrasts (threshold 0.01 macro-F1) |
| `ols_*.csv`, `dataset_slopes.csv` | Exploratory OLS models for importance and redundancy moderation |
| `mismatch*.csv` | Train/test mismatch experiment |
| `natural*.csv`, `group_b_validation.csv` | Natural-missingness validation |
| `feature_importance.csv`, `importance_robustness_summary.csv` | Importance rankings and robustness checks |
| `figures/` | Critical-difference diagrams, drop by mechanism, rate and strength, block missingness, mismatch, RMSE vs drop |

## Limitations

- Importance is model-relative, and hyperparameters are fixed rather than tuned.
- Datasets are finite and subsampled to 1,500 rows, and only 7 controlled datasets are available for cluster-level inference, so tests are low-powered and the OLS models are exploratory.
- Missingness is simulated, and conclusions depend on the assumed generating processes.
- Natural-missingness datasets have no complete-data truth, so true imputation RMSE and complete-data drops are unavailable there.
- Null and unfinished results are reported as such, and negative drops are kept.


## License

- **Code** (`implementation.ipynb`): MIT License.
- **Results, tables and figures** (`results/`): CC BY 4.0.
- **Datasets:** not redistributed in this repository. Each dataset remains under its original license; see the UCI, OpenML, scikit-learn and Kaggle dataset pages linked in the notebook. Users are responsible for complying with those terms.

## Author

tallam venkata naga Dinesh, Anne venkata praveen krishna ,Koneru Lakshmaiah Education Foundation (KL University), Vaddeswaram, Andhra Pradesh, India
