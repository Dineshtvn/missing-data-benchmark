# Key findings

Run `26ed27b95505`; primary classifier: `xgb`; mode: FULL.

Coverage: 31884/31884 planned evaluations completed (100.0%).

Empirical lowest macro-F1 drop at 20% target-feature missingness (averaged equally over datasets):

- label: native; mean drop -0.0089, N=7 datasets. Winner uncertainty remains.

- mar_redundant: rf_iterative; mean drop 0.0035, N=7 datasets. Winner uncertainty remains.

- mar_unrelated: knn; mean drop 0.0033, N=7 datasets. Winner uncertainty remains.

- mcar: knn; mean drop 0.0032, N=7 datasets. Winner uncertainty remains.

- mnar_censor: native; mean drop 0.0019, N=7 datasets. Winner uncertainty remains.

- mnar_mild: rf_iterative; mean drop 0.0033, N=7 datasets. Winner uncertainty remains.

- mnar_moderate: knn; mean drop 0.0037, N=7 datasets. Winner uncertainty remains.

- mnar_strong: native; mean drop 0.0018, N=7 datasets. Winner uncertainty remains.

Observed overall best: native, mean drop 0.0018. Validate the method on the expected observation process; this aggregate does not establish a universal winner.

Median measured imputation+training+prediction runtime per evaluation: native 0.14s, mean 0.14s, median 0.14s, median_indicator 0.14s, mode 0.15s, iterative 0.15s, knn 0.16s, rf_iterative 0.84s.

Mismatch-specific empirical choices:

- matched: native, mean drop 0.0031. See mismatch_shift rows in effect_contrasts.csv for uncertainty.

- matched_label: native, mean drop -0.0081. See mismatch_shift rows in effect_contrasts.csv for uncertainty.

- matched_mnar: mean, mean drop 0.0002. See mismatch_shift rows in effect_contrasts.csv for uncertainty.

- mechanism_shift: knn, mean drop 0.0027. See mismatch_shift rows in effect_contrasts.csv for uncertainty.

- rate_shift: rf_iterative, mean drop 0.0046. See mismatch_shift rows in effect_contrasts.csv for uncertainty.

- test_only: knn, mean drop 0.0015. See mismatch_shift rows in effect_contrasts.csv for uncertainty.

- train_only: native, mean drop 0.0010. See mismatch_shift rows in effect_contrasts.csv for uncertainty.

Reconstruction/downstream Spearman association: -0.011 (descriptive dependent observations). Select by held-out predictive performance; reconstruction error alone is not a selection criterion.

Natural-missingness validation: 3 dataset(s); complete-data drops and true imputation RMSE are unavailable.

Interpretation limits: model-relative importance, fixed hyperparameters, finite datasets, row caps, simulation assumptions, and small-cluster inference. Negative drops are retained. Null and unfinished results are reported without favorable assumptions.