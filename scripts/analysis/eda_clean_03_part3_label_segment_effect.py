#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3: EDA - Discriminative Power & Temporal Dynamics
Phân tích theo Nhãn cảm xúc (Arousal/Valence) và Vị trí Segment (Hiệu ứng Sliding Window).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA

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

def extract_top_features(X, y, names, modality, target, out_dir, top_k=30):
    # Handle NaNs in y if any (e.g., cross-corpus labels)
    valid_idx = ~np.isnan(y)
    X_v = X[valid_idx]
    y_v = y[valid_idx]
    
    f_scores, p_vals = f_classif(X_v, y_v)
    
    df = pd.DataFrame({
        "feature": names,
        "f_score": f_scores,
        "p_value": p_vals
    }).sort_values("f_score", ascending=False)
    
    top_df = df.head(top_k)
    top_df.to_csv(out_dir / f"top_features_{target}_{modality}.csv", index=False)
    
    # Plot top features
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_df.head(15), x="f_score", y="feature", hue="feature", palette="magma", legend=False)
    plt.title(f"Top 15 Features for {target.upper()} ({modality.upper()})")
    plt.tight_layout()
    plt.savefig(out_dir / f"top15_barplot_{target}_{modality}.png", dpi=150)
    plt.close()
    
    # Plot boxplot for Top 1 feature
    top1_feat = top_df.iloc[0]["feature"]
    top1_idx = names.index(top1_feat)
    
    plt.figure(figsize=(8, 6))
    plot_df = pd.DataFrame({top1_feat: X_v[:, top1_idx], "label": y_v})
    sns.boxplot(data=plot_df, x="label", y=top1_feat, hue="label", palette="Set2", legend=False)
    plt.title(f"Best Feature ({top1_feat}) Distribution by {target.upper()}")
    plt.savefig(out_dir / f"top1_boxplot_{target}_{modality}.png", dpi=150)
    plt.close()

def plot_pca(X, y, modality, target, out_dir):
    valid_idx = ~np.isnan(y)
    X_v = X[valid_idx]
    y_v = y[valid_idx]
    
    # Standardize first
    X_std = (X_v - np.mean(X_v, axis=0)) / (np.std(X_v, axis=0) + 1e-8)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_std)
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_v, palette="deep", alpha=0.6, s=15)
    plt.title(f"PCA 2D Projection ({modality.upper()}) - Colored by {target}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.tight_layout()
    plt.savefig(out_dir / f"pca_2d_{modality}_by_{target}.png", dpi=150)
    plt.close()

def analyze_segment_effect(X, names, meta, modality, out_dir):
    if "segment_idx" not in meta.columns:
        print(f"Skipping segment effect for {modality} - no 'segment_idx' found.")
        return
        
    df = pd.DataFrame(X, columns=names)
    df["segment_idx"] = meta["segment_idx"].values
    
    if modality == "eeg":
        # Group features by band if possible
        de_cols = [c for c in names if c.startswith("DE_")]
        if not de_cols: return
        
        # Melt and extract band
        melted = df[de_cols + ["segment_idx"]].melt(id_vars="segment_idx", var_name="feature", value_name="value")
        melted["band"] = melted["feature"].apply(lambda x: x.split("_")[1] if len(x.split("_"))>1 else "unk")
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=melted, x="segment_idx", y="value", hue="band", errorbar="ci", marker="o")
        plt.title("EEG DE Mean by Segment Index (Sliding Window Effect)")
        plt.xticks([0,1,2,3,4])
        plt.savefig(out_dir / "segment_effect_eeg_band_mean.png", dpi=150)
        plt.close()
        
    elif modality == "cv":
        pres_cols = [c for c in names if "presence" in c]
        if not pres_cols: return
        
        melted = df[pres_cols + ["segment_idx"]].melt(id_vars="segment_idx", var_name="feature", value_name="value")
        # Extract AU
        melted["au"] = melted["feature"].apply(lambda x: x.split("_")[0])
        
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=melted, x="segment_idx", y="value", hue="au", errorbar=None, marker="s")
        plt.title("CV AU Presence Rate by Segment Index")
        plt.xticks([0,1,2,3,4])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.savefig(out_dir / "segment_effect_au_presence.png", dpi=150)
        plt.close()

def main():
    feature_dir = Path(r"results\demo\01D_core18_first3s_sw_segments_ica_clip")
    out_dir = Path(r"results\demo\v6D_eda_reports\phase3_label_segment")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data for Phase 3...")
    eeg_x, eeg_names, cv_x, cv_names, meta = load_data(feature_dir)
    
    # Chuyển đổi nhãn Arousal và Valence sang dạng số
    # Binary thresholding for EDA simplicity: 1 = Low, 3 = High (Drop 2/Medium)
    for target in ["f_emotion_a_3class_v2", "f_emotion_v_3class_v2"]:
        t_name = "arousal" if "_a_" in target else "valence"
        y_raw = pd.to_numeric(meta[target], errors="coerce")
        # For ANOVA and PCA, we can use 3class directly (0,1,2 mapped to 1,2,3 usually)
        y = y_raw.values
        
        print(f"\n--- Extracting Top Features for {t_name.upper()} ---")
        extract_top_features(eeg_x, y, eeg_names, "eeg", t_name, out_dir)
        extract_top_features(cv_x, y, cv_names, "cv", t_name, out_dir)
        
        print(f"--- Plotting PCA for {t_name.upper()} ---")
        plot_pca(eeg_x, y, "eeg", t_name, out_dir)
        plot_pca(cv_x, y, "cv", t_name, out_dir)
        
    print("\n--- Analyzing Segment (Sliding Window) Effect ---")
    analyze_segment_effect(eeg_x, eeg_names, meta, "eeg", out_dir)
    analyze_segment_effect(cv_x, cv_names, meta, "cv", out_dir)
    
    print(f"\nPhase 3 Complete. Check results in: {out_dir}")

if __name__ == "__main__":
    main()
