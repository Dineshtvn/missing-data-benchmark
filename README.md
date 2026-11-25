# Importance-Tiered Feature Masking and Train/Test Missingness Mismatch

Code and results for the manuscript:

> **Importance-Tiered Feature Masking and Train/Test Missingness Mismatch: A Controlled Evaluation of Imputation and Native Missing-Value Handling for XGBoost Tabular Classification**
> Tallam Venkata Naga Dinesh and Anne Venkata Praveen Krishna
> Department of Computer Science and Engineering, Koneru Lakshmaiah Education Foundation, Vaddeswaram, Andhra Pradesh, India
> *Manuscript under review (submission TCSST-26-766).*

The study compares eight missing-data strategies with an XGBoost classifier. Missingness is placed in features chosen by importance rank and simulated under several mechanisms, including cases where training and test data have different missingness. A second run tests whether the conclusions depend on the main design choices.

---

## Contents

- [What the study does](#what-the-study-does)
- [Main findings](#main-findings)
- [Repository layout](#repository-layout)
- [Datasets](#datasets)
- [Running the notebooks](#running-the-notebooks)
- [Result archives](#result-archives)
- [Reproducibility notes](#reproducibility-notes)
- [Computing environment](#computing-environment)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)

---

## What the study does

**Strategies compared (8):**
- mean, median, and mode imputation;
- median imputation with missingness indicators;
- KNN imputation (k-nearest neighbors);
- iterative imputation with Bayesian ridge, labelled Iterative (BR);
- iterative imputation with random forests, labelled Iterative (RF);
- XGBoost's native missing-value handling.

**Main learner:** XGBoost with fixed settings: 120 trees, depth 4, learning rate 0.1, subsample 0.9, colsample 0.9, L2 penalty 1.0.

**Where missingness is placed:**
- Within each outer training fold, features are ranked by random-forest permutation importance, measured on a 25% inner holdout.
- The ranking is split into three tiers. The middle-ranked feature of each tier is its *representative*, and that feature is masked.
- Blocks of two or three top-ranked features are also masked together.

**Missingness mechanisms:**
- MCAR (missing completely at random);
- MAR (missing at random), with the most- and least-correlated feature as driver;
- self-masking MNAR (missing not at random) at three strengths (mild, moderate, strong);
- censoring;
- a label-dependent mask. This mask uses the true labels, so it could never occur in deployment. It is used for diagnosis only and always reported separately.

**Train/test mismatch scenarios:**
- matched MCAR;
- training-only missingness;
- test-only missingness;
- rate shift (10% → 40%);
- mechanism shift (MCAR → strong MNAR);
- matched strong MNAR;
- matched label-dependent missingness.

**Evaluation:**
- Outer evaluation is 3-fold stratified cross-validation, repeated twice (6 folds).
- Every strategy sees the same masks and the same complete-data baseline.
- Loss is the drop in macro-F1, in percentage points (pp), relative to the complete-data baseline.
- Analysis is estimation-first: dataset-level effects with pointwise bootstrap intervals. Friedman and Wilcoxon signed-rank tests with Holm correction are reported for transparency.

### Evaluation stages

| Stage | Datasets | Folds / masks | What varies | Evaluations |
|---|---|---|---|---|
| Core | 7 | 6 / 2 | 44 conditions; XGBoost; 8 strategies | 29,568 |
| Classifier sensitivity | 2 | 3 / 1 | LR, RF, HGB, MLP | 1,044 |
| Importance sensitivity | 2 | 3 / 1 | Mutual-information targets | 288 |
| Mismatch | 5 | 3 / 1 | 7 train/test scenarios | 840 |
| Natural missingness | 3 | 6 / – | Real missing values | 144 |
| **Main study total** | | | | **31,884** |
| S2 Tier design | 7 | 6 / 1 | Tier members; equal-size control blocks; 2 rankings | 15,840 |
| S3 Imputer settings | 7 | 6 / 1 | 5 vs 25 iterations; all-predictor variants; MissForest; MICE (m = 5) | 2,520 |
| S4 Learners and seeds | 7 | 6 / 1 | LR, RF, HGB, XGBoost (5 seeds), tuned XGBoost | 8,736 |
| S5 Row cap | 4 | 6 / 1 | Capped sample, second capped sample, full data | 1,008 |
| **Robustness total** | | | | **28,104** |

The robustness run also has four stages that fit no evaluation models:
- **S1:** stability of four importance estimates.
- **S6:** cost per fit, measured with a fixed number of threads.
- **S7:** reruns of stored records.
- **S8:** regeneration of the original masks.

---

## Main findings

All findings are **descriptive**. With seven datasets, the Holm-corrected signed-rank tests cannot reject, so none of the comparisons is statistically significant after correction.

- **Importance tiers:** at 20% masking, mean macro-F1 loss was 0.37 pp for top-tier representatives and 0.15 pp for bottom-tier representatives. Two of the seven datasets showed the reverse.
- **Blocks of features:** blocks of top-ranked features cost more than random, redundancy-matched, and bottom-ranked blocks of the same size. Under MCAR the losses were 1.90, 1.31, 0.83, and 0.73 pp.
- **Mechanisms:** the strategy with the lowest loss depended on the mechanism. Native handling's advantage over Iterative (RF) fell from 0.14 to 0.03 pp once the label-dependent mask was excluded.
- **Test-only missingness:** this scenario produced the largest losses. Native handling lost 2.41 pp, mostly in Banknote and MAGIC. KNN lost 0.15 pp and Iterative (RF) lost 0.25 pp.
- **Robustness:** the strategy comparisons changed little with longer iteration budgets, canonical MissForest, pooled MICE, three other learners, or tuning.
- **Seeds and row samples:** changing only the XGBoost seed moved losses more than changing the strategy (median standard deviations 0.48 vs 0.24 pp). Strategy orderings also changed between two independent 1,500-row samples.

**Practical message:** validate missing-data pipelines under the train/test conditions you expect in deployment, and over several seeds and samples. Do not rely on a single benchmark ranking.

---

## Repository layout

```
.
├── README.md
├── LICENSE                              # MIT (code)
├── notebooks/
│   ├── implementation.ipynb             # main study (31,884 evaluations)
│   └── supplementary_experiments.ipynb  # robustness run S1–S8 (28,104 evaluations)
├── figures/                             # figures used in the manuscript (PNG/PDF)
└── results/                             # small summary tables (CSV); full archives are linked below
```

> Adjust the file names above if yours differ. The full result archives are too large for a normal Git repository (see [Result archives](#result-archives)).

---

## Datasets

All datasets are public. They are **not redistributed** in this repository; the notebooks load them from scikit-learn, Kaggle, or OpenML.

| Dataset | Use | Source |
|---|---|---|
| Breast Cancer Wisconsin (Diagnostic) | Controlled | scikit-learn |
| Optical Recognition of Handwritten Digits | Controlled (capped at 1,500 rows) | scikit-learn |
| Banknote Authentication | Controlled | Kaggle: `ritesaluja/bank-note-authentication-uci-data` |
| Spambase | Controlled (capped at 1,500 rows) | Kaggle: `colormap/spambase` (main study); `kaggleprollc/spam-email-data-uci` (robustness run) |
| MAGIC Gamma Telescope | Controlled (capped at 1,500 rows) | Kaggle: `abhinavmangalore/magic-gamma-cherenkov-telescope-dataset` |
| Ionosphere | Controlled | OpenML ID 59 |
| Phoneme | Controlled (capped at 1,500 rows) | OpenML ID 1489 |
| Adult | Natural missingness (capped at 1,500 rows) | Kaggle: `uciml/adult-census-income` |
| Horse Colic | Natural missingness | Kaggle: `uciml/horse-colic` |
| Credit Approval | Natural missingness | OpenML ID 29 |

**Notes:**
- Capped datasets were reduced to 1,500 rows by stratified subsampling (seed 2026) to limit computation.
- SHA-256 fingerprints of the processed data are stored in `config.json` (main study) and `manifest.json` (robustness run).
- The two Spambase mirrors give different fingerprints but the same 4,210 rows after deduplication, identical mask rates, and identical results for every deterministic strategy.
- Download dates and Kaggle version numbers were not recorded.

---

## Running the notebooks

Both notebooks were written for **Kaggle Notebooks** with a GPU accelerator (2 × Tesla T4). They also run on a local machine with fewer resources, but much more slowly.

### 1. Main study: `implementation.ipynb`

1. Create a Kaggle notebook and import `implementation.ipynb`.
2. Under **Add Input**, add the Kaggle datasets listed above. The OpenML and scikit-learn datasets are downloaded or loaded automatically. The notebook looks for files under `/kaggle/input`; to point it at a file explicitly, use `FILE_OVERRIDES`, e.g. `{'spambase': '/kaggle/input/.../spambase.data'}`.
3. Set the options in the configuration cell:
   - `FAST_MODE = False` runs the full study; `True` runs a quick smoke test.
   - `USE_GPU = True` uses the GPU for XGBoost.
   - `RESUME_FROM = None` starts fresh. To resume, set it to a previous `study_results/<run-id>` folder or its exported ZIP.
4. Turn on the GPU accelerator and choose **Run All**.
5. Results are written to `/kaggle/working/study_results/<run-id>/`. Each completed evaluation is checkpointed, so an interrupted run can be resumed. The recorded full run took 12,103 s (about 3.4 h).

### 2. Robustness run: `supplementary_experiments.ipynb`

1. Create a new Kaggle notebook and import `supplementary_experiments.ipynb`.
2. Add the same dataset inputs as for the main study.
3. Set the options in the configuration cell:
   - `FAST = True` first, for a quick smoke test (2 datasets, 2 folds, fewer settings). Then `FAST = False` for the full run.
   - `N_THREADS = 4` sets a fixed CPU thread budget for every library.
   - `RUN = {'S1': True, ..., 'S8': True}` switches individual stages on or off.
   - `ORIGINAL_RESULTS = None` searches for the main-study results automatically. This input is **optional**: S7 compares against the original records only when they are available.
   - `DETERMINISTIC_PREDICT = True` makes random forests predict single-threaded, so the run is bit-reproducible.
   - `SAVE_MODEL_FOLDS = [0]` and `SAVE_MODEL_EXPS = ['S4', 'S5']` choose which fitted models are saved; all other models can be rebuilt from the saved splits, masks, and seeds.
4. Turn on the GPU accelerator and choose **Run All**.
5. Outputs go to `/kaggle/working/supplementary/` (or `supplementary_FAST/` in fast mode).

**Fixed settings of the robustness run:**
- base seed 2026, mask seed 17, 20% masking;
- 3 folds × 2 repeats;
- up to 5 members per tier;
- 5 XGBoost seeds;
- a 12-configuration random search with inner 3-fold cross-validation for tuning;
- MICE with m = 5 imputations;
- the second capped sample uses seed 2027.

---

## Result archives

| Archive | Size (uncompressed) | Contents |
|---|---|---|
| `study_results/<run-id>` (main study) | ≈ 192 MB | Evaluation records, baselines, statistics, figures, checkpoints |
| `supplementary/` (robustness run) | ≈ 477 MB | Records, splits, masks, predictions, models, summaries |

GitHub rejects files over 100 MB, so attach these archives to a **GitHub Release** or deposit them on **Zenodo**, and link them here:

- Main study results: `<add link>`
- Robustness results: `<add link>`

### Main study files (`study_results/<run-id>/`)

| File | Description |
|---|---|
| `results.csv` | Core and sensitivity-stage evaluation records |
| `mismatch.csv` | Train/test mismatch records |
| `natural.csv` | Natural-missingness records |
| `baselines.csv` | 66 complete-data baselines |
| `plan.csv`, `mismatch_plan.csv`, `natural_plan.csv` | Planned evaluation grids |
| `coverage.csv`, `errors.csv` | Completion status (`errors.csv` is empty) |
| `config.json`, `metadata.json`, `run_summary.json` | Settings, environment, and run time |
| `dataset_audit.csv`, `dataset_summary.csv` | Dataset screening and processed-data summaries |
| `mask_checks.csv`, `mismatch_mask_checks.csv`, `mask_unit_tests.csv` | Realized masking-rate checks |
| `feature_importance.csv`, `importance/` | Importance rankings and tier representatives |
| `bootstrap_intervals.csv`, `effect_contrasts.csv`, `friedman.csv`, `wilcoxon_holm.csv` | Statistical summaries |
| `ols_*.csv`, `dataset_slopes.csv` | Importance-slope regressions |
| `figures/` | Figures (PNG and PDF) |
| `checkpoints/` | Per-evaluation checkpoints |

### Robustness files (`supplementary/`)

| Path | Description |
|---|---|
| `S2_records.csv` … `S5_records.csv` | Evaluation records for each stage |
| `summary/S1_importance_stability.csv` | Kendall τ across folds for four importance estimates |
| `summary/S5_neighbourhood_density.csv` | Nearest-neighbour distances, capped vs full data |
| `summary/S6_compute.csv`, `summary/S6_hardware.json` | Timing and peak memory with a fixed thread budget |
| `summary/S7_rerun_check.csv`, `summary/S7_thread_order_test.csv`, `summary/S7_imputer_seed_sensitivity.csv` | Rerun and nondeterminism checks |
| `summary/S8_mask_match.csv`, `summary/S8_original_mask_index.csv` | Regenerated original masks vs recorded rates |
| `summary/response_numbers.md` | Every number quoted in the revised manuscript |
| `artifacts/splits/` | Train/test split indices |
| `artifacts/original_run/` | Regenerated masks for every original core and mismatch condition |
| `artifacts/predictions/` | Masks, predicted probabilities, and reconstructed values for each robustness record |
| `artifacts/models/` | Fitted imputers and classifiers (first fold of S4 and S5) |
| `manifest.json` | Settings, hardware, dataset fingerprints, and SHA-256 hashes of all files |

---

## Reproducibility notes

- **Seeds:** all experiment-specific seeds are derived from base seed 2026 by SHA-256 hashing of condition identifiers.
- **Masks:** the regenerated masks reproduce the recorded rates of all 3,696 core and 105 mismatch train/test mask pairs.
- **Deterministic strategies:** 2,250 robustness records repeat a core condition with mean, median plus indicators, Iterative (BR), or native handling, and all of them reproduce the original macro-F1 exactly. So do 40 randomly chosen robustness records rebuilt from the archive.
- **Random-forest nondeterminism:** scikit-learn's random forest adds up tree predictions across threads, and the order of addition can change the last bits of the imputed values.
  - In the main study, one duplicated Spambase condition gave 0.933333 in `results.csv` and 0.931307 in `mismatch.csv`.
  - Eight multi-threaded reruns of that condition gave 0.933333 seven times and 0.935363 once. Eight single-threaded reruns all gave 0.933333.
  - The robustness run therefore predicts single-threaded. Because of this, only 261 of 594 repeated Iterative (RF) records match the main study exactly (mean absolute difference 0.25 pp).
- **KNN:** 401 of 468 repeated KNN records match exactly. The differences are confined to Digits and Spambase, are at most 0.63 pp, and their source was not isolated.

---

## Computing environment

| | Main study | Robustness run |
|---|---|---|
| Platform | Kaggle Notebook | Kaggle Notebook |
| CPU | Intel Xeon @ 2.20 GHz, 4 logical CPUs | Intel Xeon @ 2.00 GHz, 4 logical CPUs, 33.7 GB RAM |
| GPU | 2 × NVIDIA Tesla T4 | 2 × NVIDIA Tesla T4 |
| Python | 3.13.15 | 3.13.16 |
| Libraries | NumPy 2.1.3, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.6.1, XGBoost 3.4.1, PyTorch 2.11.0+cu128, statsmodels 0.15.0 | Same NumPy, pandas, SciPy, scikit-learn, and XGBoost versions |
| Threads | Not limited | Limited to 4 |

To install a matching environment locally:

```bash
pip install numpy==2.1.3 pandas==2.3.3 scipy==1.16.3 scikit-learn==1.6.1 xgboost==3.4.1 statsmodels==0.15.0 matplotlib
# Main study only (MLP check):
pip install torch==2.11.0
```

---

## Limitations

- The datasets are small to medium, numeric, and come from a fixed, ad hoc list. Five datasets were capped at 1,500 rows.
- The importance tiers rest on small inner holdouts and were weakly reproduced across folds in four datasets.
- The robustness run used one mask seed, 20% masking, at most four mechanisms, and six of the eight strategies.
- No comparison survives Holm correction. Rejection would need at least 11–12 datasets.
- Label-dependent masking is a diagnostic simulation only. No recommendation relies on it.

---

## Citation

If you use this code or these results, please cite the manuscript. The details will be updated after publication.

```bibtex
@unpublished{dinesh_krishna_missingness,
  author = {Dinesh, Tallam Venkata Naga and Krishna, Anne Venkata Praveen},
  title  = {Importance-Tiered Feature Masking and Train/Test Missingness Mismatch:
            A Controlled Evaluation of Imputation and Native Missing-Value Handling
            for XGBoost Tabular Classification},
  note   = {Manuscript under review},
  year   = {2026}
}
```

---

## License

- **Code** (notebooks): MIT License.
- **Results, tables, and figures:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Datasets:** these remain under the terms of their original sources and are not redistributed here.

## Contact

Tallam Venkata Naga Dinesh — 2501050173@kluniversity.in
Koneru Lakshmaiah Education Foundation, Vaddeswaram, Andhra Pradesh, India
