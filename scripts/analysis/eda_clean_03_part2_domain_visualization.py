#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2: EDA - Domain-Specific Visualization (EEG & CV)
Phân tích theo các cấu trúc sinh lý (Kênh EEG, Băng tần não, Action Units trên mặt).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    return eeg_x, eeg_names, cv_x, cv_names

def parse_eeg_features(X, names):
    print("  -> Parsing EEG features...")
    df = pd.DataFrame(X, columns=names)
    
    # 1. DE Features: eeg_de_{channel}_{band}
    de_cols = [c for c in names if c.startswith("eeg_de_")]
    de_df = df[de_cols].melt(var_name="feature", value_name="value")
    def parse_de(f):
        f = f.replace("eeg_de_", "")
        parts = f.split("_")
        return parts[0], "_".join(parts[1:])
    if not de_df.empty:
        parsed = de_df["feature"].apply(parse_de)
        de_df["channel"] = [x[0] for x in parsed]
        de_df["band"] = [x[1] for x in parsed]
    
    # 2. DASM Features: eeg_dasm_de_{ch1}_{ch2}_{band}
    dasm_cols = [c for c in names if c.startswith("eeg_dasm_de_")]
    dasm_df = df[dasm_cols].melt(var_name="feature", value_name="value")
    def parse_dasm(f):
        f = f.replace("eeg_dasm_de_", "")
        parts = f.split("_")
        return parts[0] + "_" + parts[1], "_".join(parts[2:])
    if not dasm_df.empty:
        parsed = dasm_df["feature"].apply(parse_dasm)
        dasm_df["pair"] = [x[0] for x in parsed]
        dasm_df["band"] = [x[1] for x in parsed]
    
    # 3. Region Features: eeg_region_{region}_de_{band}
    reg_cols = [c for c in names if c.startswith("eeg_region_") and "_de_" in c]
    reg_df = df[reg_cols].melt(var_name="feature", value_name="value")
    def parse_reg(f):
        f = f.replace("eeg_region_", "")
        parts = f.split("_de_")
        return parts[0], parts[1] if len(parts)>1 else "unknown"
    if not reg_df.empty:
        parsed = reg_df["feature"].apply(parse_reg)
        reg_df["region"] = [x[0] for x in parsed]
        reg_df["band"] = [x[1] for x in parsed]
    
    return de_df, dasm_df, reg_df

def parse_cv_features(X, names):
    print("  -> Parsing CV (AU) features...")
    df = pd.DataFrame(X, columns=names)
    
    au_cols = [c for c in names if c.startswith("cv_AU")]
    melted = df[au_cols].melt(var_name="feature", value_name="value")
    
    def parse_au(feat):
        f = feat.replace("cv_", "")
        parts = f.split("_")
        return parts[0], parts[1], "_".join(parts[2:])
        
    if not melted.empty:
        parsed = melted["feature"].apply(parse_au)
        melted["au"] = [x[0] for x in parsed]
        melted["type"] = [x[1] for x in parsed]
        melted["stat"] = [x[2] for x in parsed]
    
    return melted

def plot_eeg_visualizations(de_df, dasm_df, reg_df, out_dir):
    print("  -> Plotting EEG visuals...")
    # 1. DE distribution by band
    if not de_df.empty:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=de_df, x="band", y="value", hue="band", palette="Set2", legend=False)
        plt.title("EEG DE Distribution by Band")
        plt.savefig(out_dir / "eeg_de_band_boxplot.png", dpi=150)
        plt.close()
        
        # 2. Channel x Band Heatmaps (Mean & Std)
        mean_heatmap = de_df.groupby(["channel", "band"])["value"].mean().unstack()
        std_heatmap = de_df.groupby(["channel", "band"])["value"].std().unstack()
        
        if not mean_heatmap.empty:
            plt.figure(figsize=(8, 10))
            sns.heatmap(mean_heatmap, annot=True, cmap="YlGnBu", fmt=".2f")
            plt.title("Mean DE: Channel x Band")
            plt.savefig(out_dir / "eeg_channel_band_mean_heatmap.png", dpi=150)
            plt.close()
            
            plt.figure(figsize=(8, 10))
            sns.heatmap(std_heatmap, annot=True, cmap="YlOrRd", fmt=".2f")
            plt.title("Std DE: Channel x Band")
            plt.savefig(out_dir / "eeg_channel_band_std_heatmap.png", dpi=150)
            plt.close()

    # 3. Region DE Boxplot
    if not reg_df.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=reg_df, x="region", y="value", hue="band", palette="Set3")
        plt.title("EEG Region DE Distribution by Band")
        plt.savefig(out_dir / "eeg_region_band_boxplot.png", dpi=150)
        plt.close()

    # 4. DASM Boxplot
    if not dasm_df.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=dasm_df, x="feature", y="value", hue="band", palette="Set1")
        plt.title("EEG DASM Distribution")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(out_dir / "eeg_dasm_distribution.png", dpi=150)
        plt.close()

def plot_cv_visualizations(cv_df, out_dir):
    print("  -> Plotting CV (AU) visuals...")
    # AU_r stats
    au_r = cv_df[cv_df["type"] == "r"]
    if not au_r.empty:
        # Median
        med_df = au_r[au_r["stat"] == "median"]
        if not med_df.empty:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=med_df, x="au", y="value", color="lightblue")
            plt.title("AU_r Median Distribution per AU")
            plt.savefig(out_dir / "cv_au_r_median_boxplot.png", dpi=150)
            plt.close()
            
        # P95
        p95_df = au_r[au_r["stat"] == "p95"]
        if not p95_df.empty:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=p95_df, x="au", y="value", color="salmon")
            plt.title("AU_r P95 Distribution per AU")
            plt.savefig(out_dir / "cv_au_r_p95_boxplot.png", dpi=150)
            plt.close()

    # AU_c stats
    au_c = cv_df[cv_df["type"] == "c"]
    if not au_c.empty:
        # Presence
        pres_df = au_c[au_c["stat"] == "presence"]
        if not pres_df.empty:
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=pres_df, x="au", y="value", color="lightgreen")
            plt.title("AU_c Presence Rate per AU")
            plt.savefig(out_dir / "cv_au_c_presence_rate_boxplot.png", dpi=150)
            plt.close()

    # Mean Heatmap
    mean_heat = cv_df.groupby(["au", "stat"])["value"].mean().unstack()
    if not mean_heat.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(mean_heat, cmap="magma", annot=False)
        plt.title("Mean AU Features (AU x Statistic)")
        plt.savefig(out_dir / "cv_au_feature_mean_heatmap.png", dpi=150)
        plt.close()

def main():
    feature_dir = Path(r"results\demo\01D_core18_first3s_sw_segments_ica_clip")
    out_dir = Path(r"results\demo\v6D_eda_reports\phase2_domain_viz")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data for Phase 2...")
    eeg_x, eeg_names, cv_x, cv_names = load_data(feature_dir)
    
    print("\n--- Processing EEG ---")
    de_df, dasm_df, reg_df = parse_eeg_features(eeg_x, eeg_names)
    plot_eeg_visualizations(de_df, dasm_df, reg_df, out_dir)
    
    print("\n--- Processing CV ---")
    cv_df = parse_cv_features(cv_x, cv_names)
    plot_cv_visualizations(cv_df, out_dir)
    
    print(f"\nPhase 2 Complete. Check results in: {out_dir}")

if __name__ == "__main__":
    main()
