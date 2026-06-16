# Multimodal Computer Vision and EEG in Emotion Recognition

This repository contains research code for multimodal emotion recognition using EEG features and computer-vision facial action unit features. The main direction is a first-3-seconds segmented feature pipeline with an RBF-SVM classifier.

## Main Research Direction

The current primary workflow is:

```text
01D_core18_first3s_sw_segments_ica_clip
  -> train_affec_v6_robust_v2.py
  -> SVM with RBF kernel
  -> EEG-only, CV-only, and early-fusion comparisons
```

The main feature branch uses the first 3 seconds of each trial, expanded into 1-second sliding windows with 0.5-second steps. Each generated segment is represented by:

- EEG: CORE18 channels, ICA, 100 uV clipping, differential entropy features, 133 features.
- CV: facial action unit features on matched segment intervals, 136 features.
- Fusion: EEG and CV feature concatenation, 269 features.

The main classifier is `scripts/training/train_affec_v6_robust_v2.py`, which performs trial-level 60/20/20 train/validation/test splitting, inner train-set tuning, segment-level prediction, and trial-level probability aggregation by `original_trial_uid`.

EEGNet is kept as an experimental branch only. It is useful for exploration, but it is not the current main project result.

## Project Status

This is a research project workspace cleaned for GitHub publication. Source code and documentation are intended to be versioned. Raw data, extracted tensors, intermediate feature matrices, model outputs, and most generated figures are intentionally excluded from Git because they are large and may contain dataset-specific information.

## Repository Structure

```text
.
+-- docs/
|   +-- GITHUB_READINESS.md
|   +-- METHODS.md
|   +-- references/
+-- scripts/
|   +-- analysis/        # Final 01D feature analysis and diagnostics
|   +-- reporting/       # Final report generation utilities
|   +-- training/        # Main SVM RBF pipeline
|   +-- utils/           # Trial/feature comparison helpers
|   +-- visualization/   # EEG and trial timeline visualization
+-- data/
|   +-- raw/             # Local-only raw datasets
|   +-- processed/       # Local-only processed datasets
+-- results/
|   +-- final/           # Local-only final feature/results workspace
+-- old_version/         # Local-only legacy scripts/results, ignored by Git
+-- requirements.txt
+-- README.md
```

## Key Files

Main pipeline:

- `scripts/training/train_affec_v6_robust_v2.py`: primary RBF-SVM training and evaluation script.
- `results/final/01D_core18_first3s_sw_segments_ica_clip/feature_registry_clean.json`: local feature-registry reference for the main 01D branch. This file is not pushed by default because it lives under ignored results, but its structure defines the expected input.
- `scripts/analysis/eda_clean_03_part1_qa_corr_fingerprint.py`: quality, correlation, and subject-fingerprint analysis for 01D.
- `scripts/analysis/eda_clean_03_part2_domain_visualization.py`: EEG/CV domain visualizations.
- `scripts/analysis/eda_clean_03_part3_label_segment_effect.py`: label and segment-effect analysis.
- `scripts/reporting/final_report_first3s_nominal.py`: summary and report export utilities.
- `scripts/visualization/visualize_run_timeline.py`: EEG/CV trial timeline visualization.

Archived locally under `old_version/`:

- SVM v1/v3 variants.
- EEGNet and raw-EEG tensor experiments.
- Earlier EDA/extraction scripts.
- Non-final results and plots.

## Installation

Use Python 3.10 or newer. A virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch is only required for the archived experimental EEGNet branch under `old_version/`.

## Data Layout

The repository does not include raw EEG files, processed feature matrices, or trained outputs. Keep local data and generated artifacts in:

```text
data/raw/
data/processed/
results/
```

The main local feature directory now lives under:

```text
results/final/01D_core18_first3s_sw_segments_ica_clip/
+-- feature_registry_clean.json
+-- metadata_trials_clean.csv
+-- trial_feature_quality_clean.csv
+-- eeg/
|   +-- X_eeg_core18_first3s_seg1s_step0_5s_clean.npy
|   +-- feature_names_eeg_core18_first3s_seg1s_step0_5s_clean.json
+-- cv/
    +-- X_cv_au_only_first3s_seg1s_step0_5s.npy
    +-- feature_names_cv_au_only_first3s_seg1s_step0_5s.json
```

## Main Workflow

Run the main 3-class RBF-SVM protocol:

```bash
python scripts/training/train_affec_v6_robust_v2.py \
  --feature_dir results/final/01D_core18_first3s_sw_segments_ica_clip \
  --task 3class \
  --targets f_emotion_a_3class_v2 f_emotion_v_3class_v2 \
  --modalities eeg cv fusion \
  --out_prefix v6_robust_svm
```

Optional feature filtering using EDA phase-1 reports:

```bash
python scripts/training/train_affec_v6_robust_v2.py \
  --feature_dir results/final/01D_core18_first3s_sw_segments_ica_clip \
  --task 3class \
  --modalities eeg cv fusion \
  --filter_corr \
  --drop_top_id 10 \
  --eda_dir results/final/v6D_eda_reports/phase1_qa_corr
```

Experimental EEGNet code has been moved to `old_version/` and is not part of the final GitHub upload.

## Reproducibility Notes

- Split by `original_trial_uid`, not by generated segment rows.
- Keep all segments from the same original trial in the same split.
- Report trial-level metrics after aggregating segment probabilities.
- Keep raw data, `.npy` matrices, per-trial predictions, and generated plots outside normal Git.
- Commit only small, curated, anonymized summaries when they are needed for a release or paper.

## License

No license has been selected yet. Add a `LICENSE` file before making this repository public so users know how they may reuse the code.
