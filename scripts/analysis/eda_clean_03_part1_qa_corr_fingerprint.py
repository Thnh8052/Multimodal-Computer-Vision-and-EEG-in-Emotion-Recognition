#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: EDA - Quality Assurance, Correlation & Subject Fingerprint
Dành cho tập dữ liệu 01D Sliding Windows.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

sns.set_theme(style="whitegrid")

def load_data(feature_dir: Path):
    registry_path = feature_dir / "feature_registry_clean.json"
    with open(registry_path, "r") as f:
        registry = json.load(f)
    
    eeg_set, cv_set = None, None
    if "feature_sets" in registry:
        for k, v in registry["feature_sets"].items():
            if v.get("modality") == "eeg": eeg_set = v
            elif v.get("modality") == "cv": cv_set = v
            
    eeg_x = np.load(feature_dir / eeg_set.get("path", eeg_set.get("clean_file")))
    with open(feature_dir / eeg_set["feature_names"], "r") as f: eeg_names = json.load(f)
    
    cv_x = np.load(feature_dir / cv_set.get("path", cv_set.get("clean_file")))
    with open(feature_dir / cv_set["feature_names"], "r") as f: cv_names = json.load(f)
    
    meta_path = registry.get("metadata_csv", registry.get("metadata", {}).get("clean_file", "metadata_trials_clean.csv"))
    meta = pd.read_csv(feature_dir / meta_path)
    
    return eeg_x, eeg_names, cv_x, cv_names, meta

def plot_top_variance(X, names, modality, out_dir):
    variances = np.var(X, axis=0)
    stds = np.std(X, axis=0)
    
    df_var = pd.DataFrame({"feature": names, "variance": variances, "std": stds})
    df_var = df_var.sort_values("variance", ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_var.head(30), x="variance", y="feature", hue="feature", palette="viridis", legend=False)
    plt.title(f"Top 30 Highest Variance Features ({modality.upper()})")
    plt.xlabel("Variance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_dir / f"variance_top30_{modality}.png", dpi=150)
    plt.close()
    
    near_constant = df_var[df_var["std"] < 0.01]
    return near_constant

def plot_global_histogram(X, modality, out_dir):
    plt.figure(figsize=(10, 6))
    vals = X.flatten()
    # Randomly sample if too large to speed up plotting
    if len(vals) > 500000:
        vals = np.random.choice(vals, 500000, replace=False)
    
    sns.histplot(vals, bins=100, kde=True, color="teal" if modality=="eeg" else "coral")
    plt.title(f"Global Value Distribution ({modality.upper()})")
    plt.xlabel("Feature Values")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_dir / f"global_histogram_{modality}.png", dpi=150)
    plt.close()

def analyze_correlation(X, names, modality, out_dir):
    df = pd.DataFrame(X, columns=names)
    corr = df.corr()
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, 
                xticklabels=False, yticklabels=False)
    plt.title(f"Feature Correlation Heatmap ({modality.upper()})")
    plt.tight_layout()
    plt.savefig(out_dir / f"corr_heatmap_{modality}.png", dpi=150)
    plt.close()
    
    # Extract high correlation pairs
    corr_unstacked = corr.abs().unstack()
    corr_unstacked = corr_unstacked[corr_unstacked < 1.0] # Remove self correlation
    high_corr = corr_unstacked[corr_unstacked > 0.95].drop_duplicates().reset_index()
    high_corr.columns = ["Feature_1", "Feature_2", "Abs_Correlation"]
    high_corr = high_corr.sort_values("Abs_Correlation", ascending=False)
    
    high_corr.to_csv(out_dir / f"high_corr_pairs_{modality}.csv", index=False)
    return high_corr

def analyze_subject_fingerprint(X, names, meta, modality, out_dir):
    print(f"Testing Subject Fingerprint (Identity Leakage) using {modality.upper()}...")
    subject_labels = meta["subject"].astype(str).values
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accs = []
    importances = np.zeros(X.shape[1])
    
    for tr_idx, te_idx in skf.split(X, subject_labels):
        clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        clf.fit(X[tr_idx], subject_labels[tr_idx])
        preds = clf.predict(X[te_idx])
        accs.append(accuracy_score(subject_labels[te_idx], preds))
        importances += clf.feature_importances_
        
    mean_acc = np.mean(accs)
    importances /= 3.0
    
    print(f"Subject Prediction Accuracy ({modality.upper()}): {mean_acc*100:.2f}% (Random guess: ~{100/24:.2f}%)")
    
    # Save Top fingerprint features
    df_imp = pd.DataFrame({"feature": names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False)
    df_imp.to_csv(out_dir / f"subject_fingerprint_top_features_{modality}.csv", index=False)
    
    # Plot top 20 fingerprint features
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df_imp.head(20), x="importance", y="feature", hue="feature", palette="Reds_r", legend=False)
    plt.title(f"Top 20 Identity-Leaking Features ({modality.upper()})")
    plt.xlabel("Random Forest Feature Importance (Predicting Subject ID)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_dir / f"subject_fingerprint_top20_{modality}.png", dpi=150)
    plt.close()
    
    with open(out_dir / f"subject_fingerprint_report_{modality}.txt", "w") as f:
        f.write(f"Subject Fingerprint Accuracy: {mean_acc*100:.2f}%\n")
        f.write(f"Random Guess Baseline: {100/24:.2f}%\n")
        if mean_acc > 0.8:
            f.write("WARNING: Strong identity leakage detected.\n")
        else:
            f.write("GOOD: Identity leakage is reduced or minimal.\n")

def main():
    feature_dir = Path(r"results\demo\01D_core18_first3s_sw_segments_ica_clip")
    out_dir = Path(r"results\demo\v6D_eda_reports\phase1_qa_corr")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    eeg_x, eeg_names, cv_x, cv_names, meta = load_data(feature_dir)
    print(f"EEG: {eeg_x.shape}, CV: {cv_x.shape}")
    
    # --- 1. QA & Global Distributions ---
    print("Plotting distributions and checking variance...")
    plot_global_histogram(eeg_x, "eeg", out_dir)
    plot_global_histogram(cv_x, "cv", out_dir)
    
    nc_eeg = plot_top_variance(eeg_x, eeg_names, "eeg", out_dir)
    nc_cv = plot_top_variance(cv_x, cv_names, "cv", out_dir)
    
    nc_all = pd.concat([nc_eeg.assign(modality="eeg"), nc_cv.assign(modality="cv")])
    nc_all.to_csv(out_dir / "near_constant_features.csv", index=False)
    print(f"Found {len(nc_all)} near-constant features.")
    
    # --- 2. Correlation Analysis ---
    print("Analyzing correlations...")
    hc_eeg = analyze_correlation(eeg_x, eeg_names, "eeg", out_dir)
    hc_cv = analyze_correlation(cv_x, cv_names, "cv", out_dir)
    print(f"Found {len(hc_eeg)} highly correlated pairs (>0.95) in EEG.")
    print(f"Found {len(hc_cv)} highly correlated pairs (>0.95) in CV.")
    
    # --- 3. Subject Fingerprint ---
    print("Analyzing Subject Fingerprints...")
    analyze_subject_fingerprint(eeg_x, eeg_names, meta, "eeg", out_dir)
    analyze_subject_fingerprint(cv_x, cv_names, meta, "cv", out_dir)
    
    print(f"\nPhase 1 Complete. Check results in: {out_dir}")

if __name__ == "__main__":
    main()
