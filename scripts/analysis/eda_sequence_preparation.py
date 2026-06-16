import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
import json
import warnings

warnings.filterwarnings("ignore")

def load_data(feature_dir: Path):
    registry_path = feature_dir / "feature_registry_clean.json"
    with open(registry_path, "r") as f:
        registry = json.load(f)
        
    eeg_set = registry.get("eeg", {})
    cv_set = registry.get("cv", {})
    
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

def check_completeness(meta, out_dir):
    print("\n--- 1. Segment Completeness & Order ---")
    
    # Check completeness
    counts = meta.groupby("original_trial_uid").size()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x=counts.values)
    plt.title("Segments per Trial")
    plt.xlabel("Number of Segments")
    plt.ylabel("Trial Count")
    
    plt.subplot(1, 2, 2)
    sns.countplot(data=meta, x="segment_idx")
    plt.title("Count per Segment Index")
    
    plt.tight_layout()
    plt.savefig(out_dir / "segment_completeness.png", dpi=150)
    plt.close()
    
    # Table of missing
    missing_trials = counts[counts != 5]
    if len(missing_trials) > 0:
        print(f"WARNING: Found {len(missing_trials)} trials with missing segments!")
        missing_trials.to_csv(out_dir / "missing_segment_trials.csv")
    else:
        print("PASS: All trials have exactly 5 segments.")
        
    # Check Order/Offset
    if "segment_start_offset_sec" in meta.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=meta, x="segment_idx", y="segment_start_offset_sec")
        plt.title("Segment Index vs Offset Time")
        plt.savefig(out_dir / "segment_order.png", dpi=150)
        plt.close()
        
        plt.figure(figsize=(8, 6))
        sns.histplot(meta["segment_duration_sec"], bins=30)
        plt.title("Segment Duration Distribution")
        plt.savefig(out_dir / "segment_duration.png", dpi=150)
        plt.close()
    else:
        print("Note: 'segment_start_offset_sec' not found in metadata.")

def check_label_consistency(meta, out_dir):
    print("\n--- 2. Label Consistency Check ---")
    a_unique = meta.groupby("original_trial_uid")["f_emotion_a_3class_v2"].nunique()
    v_unique = meta.groupby("original_trial_uid")["f_emotion_v_3class_v2"].nunique()
    
    max_a = a_unique.max()
    max_v = v_unique.max()
    
    print(f"Max unique Arousal labels per trial: {max_a}")
    print(f"Max unique Valence labels per trial: {max_v}")
    
    inconsistent = meta[
        meta["original_trial_uid"].isin(a_unique[a_unique > 1].index) | 
        meta["original_trial_uid"].isin(v_unique[v_unique > 1].index)
    ]
    if len(inconsistent) > 0:
        print(f"WARNING: Found {len(inconsistent)} segments with inconsistent labels in the same trial!")
        inconsistent.to_csv(out_dir / "label_inconsistent_trials.csv", index=False)
    else:
        print("PASS: Label consistency verified.")

def plot_trial_distribution(meta, out_dir):
    print("\n--- 3. Trial-Level Label Distribution ---")
    
    # We take the first segment of each trial to represent the trial label
    trial_meta = meta.drop_duplicates(subset=["original_trial_uid"], keep="first")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(data=trial_meta, x="f_emotion_a_3class_v2", order=["Low", "Medium", "High"])
    plt.title(f"Arousal (Trial-Level, N={len(trial_meta)})")
    
    plt.subplot(1, 2, 2)
    sns.countplot(data=trial_meta, x="f_emotion_v_3class_v2", order=["Low", "Medium", "High"])
    plt.title(f"Valence (Trial-Level, N={len(trial_meta)})")
    
    plt.tight_layout()
    plt.savefig(out_dir / "trial_label_distribution.png", dpi=150)
    plt.close()
    
    print(trial_meta["f_emotion_a_3class_v2"].value_counts())
    print(trial_meta["f_emotion_v_3class_v2"].value_counts())

def check_data_split(meta, out_dir):
    print("\n--- 4. Cross-Trial Split Check ---")
    trial_meta = meta.drop_duplicates(subset=["original_trial_uid"], keep="first")
    X_dummy = np.zeros(len(trial_meta))
    y_dummy = trial_meta["f_emotion_a_3class_v2"].values
    groups = trial_meta["original_trial_uid"].values
    
    gkf = GroupKFold(n_splits=5)
    
    # Test just the first split
    train_idx, test_idx = next(gkf.split(X_dummy, y_dummy, groups))
    
    train_uids = set(groups[train_idx])
    test_uids = set(groups[test_idx])
    
    overlap = train_uids & test_uids
    print(f"Train Trials: {len(train_uids)}, Test Trials: {len(test_uids)}")
    print(f"Overlap between Train and Test: {len(overlap)}")
    
    if len(overlap) == 0:
        print("PASS: Cross-Trial split logic is sound. No data leakage across folds.")
    else:
        print("WARNING: Data leakage detected!")

def plot_temporal_trend(eeg_x, eeg_names, cv_x, cv_names, meta, out_dir):
    print("\n--- 5. Temporal Trend EDA ---")
    
    # Combine into DF for easier grouping
    eeg_df = pd.DataFrame(eeg_x, columns=eeg_names)
    cv_df = pd.DataFrame(cv_x, columns=cv_names)
    
    eeg_df["segment_idx"] = meta["segment_idx"].values
    cv_df["segment_idx"] = meta["segment_idx"].values
    
    eeg_df["arousal"] = meta["f_emotion_a_3class_v2"].values
    cv_df["arousal"] = meta["f_emotion_a_3class_v2"].values
    
    eeg_df["valence"] = meta["f_emotion_v_3class_v2"].values
    cv_df["valence"] = meta["f_emotion_v_3class_v2"].values
    
    cv_features_a = ["cv_AU07_r_p95", "cv_AU04_c_presence_rate", "cv_AU10_c_presence_rate"]
    cv_features_v = ["cv_AU12_c_presence_rate", "cv_AU12_r_p95", "cv_AU06_r_p95"]
    eeg_features_a = ["eeg_dasm_de_F3_F4_beta", "eeg_de_O2_beta", "eeg_dasm_de_AF3_AF4_gamma_low"]
    eeg_features_v = ["eeg_de_C3_gamma_low", "eeg_region_central_de_gamma_low", "eeg_de_C3_alpha"]
    
    def plot_trend(df, features, target_col, title, filename):
        plt.figure(figsize=(15, 5))
        for i, feat in enumerate(features):
            if feat in df.columns:
                plt.subplot(1, 3, i+1)
                sns.lineplot(data=df, x="segment_idx", y=feat, hue=target_col, marker="o", errorbar=None)
                plt.title(feat)
                plt.xticks([0, 1, 2, 3, 4])
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=150)
        plt.close()
        
    plot_trend(cv_df, cv_features_a, "arousal", "CV Arousal Trends", "trend_cv_arousal.png")
    plot_trend(cv_df, cv_features_v, "valence", "CV Valence Trends", "trend_cv_valence.png")
    
    plot_trend(eeg_df, eeg_features_a, "arousal", "EEG Arousal Trends", "trend_eeg_arousal.png")
    plot_trend(eeg_df, eeg_features_v, "valence", "EEG Valence Trends", "trend_eeg_valence.png")
    print("Trend plots generated.")

def check_3d_scaling(eeg_x, eeg_names, meta, out_dir):
    print("\n--- 6. Feature Scaling Simulation (3D) ---")
    # First drop trials that don't have exactly 5 segments so we can reshape cleanly
    counts = meta.groupby("original_trial_uid").size()
    valid_trials = counts[counts == 5].index
    
    valid_idx = meta["original_trial_uid"].isin(valid_trials)
    eeg_valid = eeg_x[valid_idx]
    meta_valid = meta[valid_idx].sort_values(by=["original_trial_uid", "segment_idx"])
    
    N_trials = len(valid_trials)
    F = eeg_valid.shape[1]
    
    try:
        # Reshape to [N_trials, 5, F]
        X_3d = eeg_valid.reshape(N_trials, 5, F)
        print(f"Successfully reshaped to 3D Tensor: {X_3d.shape}")
        
        # Simulate Train/Test Split (80/20)
        split = int(0.8 * N_trials)
        X_train_3d = X_3d[:split]
        X_test_3d = X_3d[split:]
        
        # Apply RobustScaler properly (fit on train 2D, transform all)
        scaler = RobustScaler()
        X_train_2d = X_train_3d.reshape(-1, F)
        scaler.fit(X_train_2d)
        
        X_train_scaled = scaler.transform(X_train_2d).reshape(split, 5, F)
        X_test_scaled = scaler.transform(X_test_3d.reshape(-1, F)).reshape(N_trials - split, 5, F)
        print("Successfully applied 3D Scaling logic (RobustScaler).")
        
        # Plot before and after for a specific feature
        feat_idx = eeg_names.index("eeg_de_C3_gamma_low") if "eeg_de_C3_gamma_low" in eeg_names else 0
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(X_train_3d[:, :, feat_idx].flatten(), bins=50)
        plt.title("Before Scaling (eeg_de_C3_gamma_low)")
        
        plt.subplot(1, 2, 2)
        sns.histplot(X_train_scaled[:, :, feat_idx].flatten(), bins=50)
        plt.title("After RobustScaler")
        
        plt.tight_layout()
        plt.savefig(out_dir / "scaling_simulation.png", dpi=150)
        plt.close()
        
    except Exception as e:
        print(f"Error during 3D reshape/scale simulation: {e}")

def main():
    feature_dir = Path(r"results\demo\01D_core18_first3s_sw_segments_ica_clip")
    out_dir = Path(r"results\demo\v6D_eda_reports\sequence_preparation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading Data...")
    eeg_x, eeg_names, cv_x, cv_names, meta = load_data(feature_dir)
    print(f"Data shapes - EEG: {eeg_x.shape}, CV: {cv_x.shape}, Meta: {meta.shape}")
    
    check_completeness(meta, out_dir)
    check_label_consistency(meta, out_dir)
    plot_trial_distribution(meta, out_dir)
    check_data_split(meta, out_dir)
    plot_temporal_trend(eeg_x, eeg_names, cv_x, cv_names, meta, out_dir)
    check_3d_scaling(eeg_x, eeg_names, meta, out_dir)
    
    print("\nEDA Completed! Check the sequence_preparation directory.")

if __name__ == "__main__":
    main()
