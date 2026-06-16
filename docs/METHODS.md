# Methods

This document summarizes the current main workflow for the project.

## Main Feature Set

The current main feature branch is:

```text
results/final/01D_core18_first3s_sw_segments_ica_clip
```

It contains first-3-second multimodal features expanded into generated segment rows:

- Segment design: 1-second windows with 0.5-second step inside the first 3 seconds.
- Number of generated segment rows: 21,947.
- Number of original trials: 4,390.
- Median segments per trial: 5.
- EEG branch: CORE18 channels, ICA, 100 uV clipping, differential entropy features, 133 features.
- CV branch: AU-only features on matched segment intervals, 136 features.
- Fusion branch: EEG and CV concatenation, 269 features.

## Main Model

The main model is implemented in:

```text
scripts/training/train_affec_v6_robust_v2.py
```

The classifier is an RBF-kernel SVM wrapped in a scikit-learn pipeline:

- Median imputation.
- Standard scaling.
- `SelectKBest` with `f_classif`.
- `SVC(kernel="rbf", class_weight="balanced", probability=True)`.

The script evaluates EEG-only, CV-only, and early-fusion modalities.

## Evaluation Protocol

The main script uses a trial-level protocol:

- Build a trial table by `original_trial_uid`.
- Split original trials into 60% train, 20% validation, and 20% test.
- Tune `k` and `C` inside the train set with stratified folds.
- Fit the final model on the full train set.
- Predict generated segment rows for validation and test sets.
- Aggregate segment probabilities by `original_trial_uid` using mean soft voting.
- Report trial-level macro F1, weighted F1, and accuracy.

Generated segment rows from the same trial are highly correlated. Do not randomly split rows directly.

## Experimental Branches

The following are not the current main direction and are stored under `old_version/`:

- `old_version/scripts/train/train_affec_v6_robust_v3.py`: alternative SVM variant with extra gamma/scaler experiments.
- `old_version/scripts/train/train_eegnet.py`: experimental EEGNet workflow.
- `old_version/scripts/train/experimental/`: exploratory raw-EEG and EEGNet utilities.
