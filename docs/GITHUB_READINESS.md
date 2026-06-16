# GitHub Readiness Review

This document records the current publication status of the project and which files are suitable for GitHub.

## Recommended Files to Push

Core repository files:

- `README.md`
- `.gitignore`
- `requirements.txt`
- `docs/GITHUB_READINESS.md`
- `docs/METHODS.md`
- `docs/references/README.md`
- `results/README.md`

Source code:

- `scripts/analysis/*.py`
- `scripts/reporting/*.py`
- `scripts/training/*.py`
- `scripts/utils/*.py`
- `scripts/visualization/*.py`

Current main workflow files:

- `scripts/training/train_affec_v6_robust_v2.py`
- `scripts/analysis/eda_clean_03_part1_qa_corr_fingerprint.py`
- `scripts/analysis/eda_clean_03_part2_domain_visualization.py`
- `scripts/analysis/eda_clean_03_part3_label_segment_effect.py`
- `scripts/analysis/eda_sequence_preparation.py`
- `scripts/analysis/eval_v6_fingerprint_importance.py`
- `scripts/reporting/final_report_first3s_nominal.py`
- `scripts/visualization/visualize_run_timeline.py`

Current main local feature branch, not pushed by default:

- `results/final/01D_core18_first3s_sw_segments_ica_clip/`
- `results/final/v6D_eda_reports/`
- `results/final/v6_robust_svm_binary_paper_protocol/`

Legacy code that can be pushed if you want method history:

- `old_version/scripts/eda/`
- `old_version/scripts/train/`
- `old_version/scripts/models/`

For a final-only GitHub upload, keep `old_version/` local and ignored.

Small optional artifacts:

- Small anonymized `*_summary.json` or `summary_*.csv` files if they are necessary to support a paper result.
- A small number of curated figures for the README or paper, preferably under a future `docs/figures/` directory.

## Files That Should Not Be Pushed by Default

- `results/**/*.npy`
- `results/**/*.csv` when they are large, per-trial, or contain dataset-specific identifiers.
- `results/**/*.png` unless manually curated for documentation.
- `results/**/*.pt`, `*.pth`, `*.ckpt`, `*.joblib`, `*.pkl`
- Raw EEG data such as `*.edf`, `*.bdf`, `*.fif`
- `__pycache__/` and `*.pyc`
- `*.bak`
- `docs/references/*.pdf` unless you have redistribution permission.
- `old_version/` unless you intentionally want to publish legacy experiments.

## Current Assessment

Strengths:

- The repository already has a clear research direction: EEG, CV action units, and early-fusion emotion recognition.
- The main direction is now clearly documented as 01D first-3-second sliding-window features with `scripts/training/train_affec_v6_robust_v2.py`.
- The code covers multiple stages: feature extraction, EDA, training, visualization, and final reporting.
- The main SVM pipeline uses train/validation/test separation and trial-level probability aggregation.
- EEGNet and earlier SVM/EDA variants are separated into `old_version/` instead of being presented as the primary result.

Weaknesses before public release:

- No license is defined yet.
- No dataset access instructions or data card are included.
- Generated artifacts are much larger than source code and should remain outside normal Git.
- Scripts are research scripts rather than a Python package with stable CLI entry points.
- There are no automated tests or smoke-test commands.
- Some script text appears to contain encoding artifacts from earlier edits; review docstrings before publication.

## Legacy Folder Assessment

Keep as method archive:

- `old_version/scripts/train/old/train_affec_v3_deonly_svm_3class.py`
- `old_version/scripts/train/old/train_affec_v4_binary_deonly_svm.py`
- `old_version/scripts/train/old/train01_B_tabular_first3s_baselines.py`
- `old_version/scripts/train/old/train01_A2_first3s_nonlinear*.py`
- `old_version/scripts/train/old/train01_A2_first3s_boosting.py`
- `old_version/scripts/eda/oldv2/eda_clean_00_manifest_feature_audit_video.py`
- `old_version/scripts/eda/oldv2/eda_clean_01*_extract_*.py`
- `old_version/scripts/eda/oldv2/eda_clean_02*_*.py`
- `old_version/scripts/eda/oldv2/eda03_feature_audit_core18_auonly.py`

Consider removing or moving later to a separate archive branch:

- `old_version/scripts/eda/old/scratch_check_cv.py`
- Any script that only reproduces a one-time local check and is not referenced in documentation.
- `old_version/scripts/train/train_affec_v6_robust_v2.py.bak` should stay ignored and should not be pushed.

## Recommended Next Additions

Before publishing as a complete GitHub project:

- Add a `LICENSE` file.
- Add `CITATION.cff` if this supports a thesis, paper, or report.
- Add a `docs/DATA.md` file describing dataset source, access, preprocessing assumptions, privacy constraints, and directory layout.
- Add a tiny smoke test or sample fixture that runs without private data.
- Add a curated result table in `docs/RESULTS.md` instead of committing full result folders.
- Normalize script encoding and clean up old mojibake in docstrings before release.
