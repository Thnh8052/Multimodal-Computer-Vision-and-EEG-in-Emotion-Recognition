#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_data(feature_dir: Path) -> Tuple[np.ndarray, List[str], np.ndarray, List[str], pd.DataFrame]:
    registry_path = feature_dir / "feature_registry_clean.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"Feature registry not found in {feature_dir}")
        
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

def filter_features(X, names, modality, args):
    if not args.filter_corr and args.drop_top_id <= 0:
        return X, names
        
    print(f"[{modality.upper()}] Initial features: {len(names)}")
    features_to_drop = set()
    eda_dir = Path(args.eda_dir)
    
    if args.filter_corr:
        corr_file = eda_dir / f"high_corr_pairs_{modality}.csv"
        if corr_file.exists():
            df_corr = pd.read_csv(corr_file)
            for f2 in df_corr["Feature_2"]:
                features_to_drop.add(f2)
            print(f"  -> Marked {len(df_corr)} features for removal due to high correlation.")
            
    if args.drop_top_id > 0:
        id_file = eda_dir / f"subject_fingerprint_top_features_{modality}.csv"
        if id_file.exists():
            df_id = pd.read_csv(id_file)
            top_id_features = df_id["feature"].head(args.drop_top_id).tolist()
            for f in top_id_features:
                features_to_drop.add(f)
            print(f"  -> Marked top {args.drop_top_id} identity-leaking features for removal.")
            
    keep_indices = [i for i, name in enumerate(names) if name not in features_to_drop]
    X_filtered = X[:, keep_indices]
    names_filtered = [names[i] for i in keep_indices]
    print(f"[{modality.upper()}] Filtered out {len(names) - len(names_filtered)} features. Remaining: {len(names_filtered)}")
    
    return X_filtered, names_filtered

def aggregate_trial_proba(proba_seg, meta_seg, classes):
    """
    Predict trên segment rows -> soft voting về original_trial_uid -> tính metric ở trial-level
    """
    classes_list = list(classes)
    proba_df = pd.DataFrame(proba_seg, columns=classes_list)
    proba_df["original_trial_uid"] = np.asarray(meta_seg["original_trial_uid"])
    trial_proba = proba_df.groupby("original_trial_uid")[classes_list].mean()
    trial_pred = trial_proba.idxmax(axis=1).astype(int)
    return trial_pred, trial_proba

def get_trial_true_labels(meta_seg, trial_pred, target_col):
    return meta_seg.groupby("original_trial_uid")[target_col].first().loc[trial_pred.index].astype(int)

def eval_trial_metrics(y_true, y_pred, is_binary: bool) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    avg_mode = "macro" if is_binary else "macro"
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average=avg_mode, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

def print_metrics(metrics_dict):
    print(f"  Macro F1: {metrics_dict['macro_f1']:.4f} | Acc: {metrics_dict['acc']:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default=r"results\demo\01D_core18_first3s_sw_segments_ica_clip")
    ap.add_argument("--task", choices=["3class", "binary"], required=True)
    ap.add_argument("--targets", nargs="+", default=["f_emotion_a_3class_v2", "f_emotion_v_3class_v2"])
    ap.add_argument("--k_values_single", nargs="+", type=int, default=[40, 60, 80, 100, 133], help="k values for EEG(133) and CV(136)")
    ap.add_argument("--k_values_fusion", nargs="+", type=int, default=[60, 100, 120, 140,180, 220, 269], help="k values for Fusion(269)")
    ap.add_argument("--C_values", nargs="+", type=float, default=[0.4, 1.0, 2.0])
    ap.add_argument("--modalities", nargs="+", default=["eeg", "cv", "fusion"], choices=["eeg", "cv", "fusion"])
    ap.add_argument("--out_prefix", type=str, default="v6_robust_svm", help="Prefix for the output directory")
    
    # Feature Filtering
    ap.add_argument("--filter_corr", action="store_true", help="Filter out Feature_2 from high correlation pairs")
    ap.add_argument("--drop_top_id", type=int, default=0, help="Number of top identity-leaking features to drop")
    ap.add_argument("--eda_dir", type=str, default=r"results\demo\v6D_eda_reports\phase1_qa_corr", help="Path to EDA phase1 report dir")
    
    args = ap.parse_args()

    feature_dir = Path(args.feature_dir)
    out_dir = mkdir(feature_dir.parent / f"{args.out_prefix}_{args.task}_paper_protocol")
    mkdir(out_dir / "oof_reports")

    all_summary_rows = []
    
    print("Loading data...")
    eeg_x, eeg_names, cv_x, cv_names, meta = load_data(feature_dir)
    
    if args.filter_corr or args.drop_top_id > 0:
        print("\n--- Applying Feature Filtering ---")
        if "eeg" in args.modalities or "fusion" in args.modalities:
            eeg_x, eeg_names = filter_features(eeg_x, eeg_names, "eeg", args)
        if "cv" in args.modalities or "fusion" in args.modalities:
            cv_x, cv_names = filter_features(cv_x, cv_names, "cv", args)
        print("----------------------------------\n")
        
    for modality in args.modalities:
        print(f"\n{'='*60}")
        print(f" MODALITY: {modality.upper()}")
        print(f"{'='*60}")
        
        if modality == "eeg":
            X_all, names = eeg_x, eeg_names
        elif modality == "cv":
            X_all, names = cv_x, cv_names
        elif modality == "fusion":
            X_all = np.hstack([eeg_x, cv_x])
            names = eeg_names + cv_names
            
        print(f"Loaded feature matrix for {modality}: {X_all.shape}")
        
        for target in args.targets:
            print(f"\n--- Target: {target} ---")
            
            y_raw = pd.to_numeric(meta[target], errors="coerce")
            if args.task == "binary":
                y_series = y_raw.copy()
                y_series[y_raw == 1] = np.nan
                y_series[y_raw == 2] = 1
            else:
                y_series = y_raw.copy()

            X_mod = np.where(np.isfinite(X_all), X_all, np.nan)
            keep = y_series.notna().to_numpy()
            X = X_mod[keep]
            y = y_series.loc[keep].astype(int).to_numpy()
            meta_kept = meta.loc[keep].copy()
            meta_kept["y_target"] = y
            
            # 1. Trial-level table
            trial_df = (
                meta_kept.sort_values(["original_trial_uid", "segment_idx"])
                .groupby("original_trial_uid")
                .first()
                .reset_index()
            )
            trial_uids = np.asarray(trial_df["original_trial_uid"])
            y_trial = np.asarray(trial_df["y_target"], dtype=int)
            
            # 2. Split 60/20/20 at trial level
            train_val_uids, test_uids, y_train_val, y_test_trial = train_test_split(
                trial_uids, y_trial, test_size=0.20, stratify=y_trial, random_state=42
            )
            
            train_uids, val_uids, y_train_trial, y_val_trial = train_test_split(
                train_val_uids, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42
            )
            
            assert len(set(train_uids) & set(val_uids)) == 0
            assert len(set(train_uids) & set(test_uids)) == 0
            assert len(set(val_uids) & set(test_uids)) == 0
            
            print(f"Trial Splits: Train={len(train_uids)} (60%), Val={len(val_uids)} (20%), Test={len(test_uids)} (20%)")
            
            # Mappings
            train_mask = meta_kept["original_trial_uid"].isin(train_uids).to_numpy()
            val_mask = meta_kept["original_trial_uid"].isin(val_uids).to_numpy()
            test_mask = meta_kept["original_trial_uid"].isin(test_uids).to_numpy()
            
            X_train, y_train_seg = X[train_mask], y[train_mask]
            X_val, y_val_seg = X[val_mask], y[val_mask]
            X_test, y_test_seg = X[test_mask], y[test_mask]
            
            meta_train = meta_kept.loc[train_mask].copy()
            meta_val = meta_kept.loc[val_mask].copy()
            meta_test = meta_kept.loc[test_mask].copy()
            
            # 3. 5-Fold Tuning on Train
            print("\n[Phase 1] 5-Fold Hyperparameter Tuning on 60% Train Set (Preprocessing: 256Hz, 4-45Hz, ICA)")
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            train_trial_df = trial_df[trial_df["original_trial_uid"].isin(train_uids)].copy()
            train_trial_y = np.asarray(train_trial_df["y_target"], dtype=int)
            
            k_values = args.k_values_single if modality in ["eeg", "cv"] else args.k_values_fusion
            
            param_scores = []
            total_combos = len(k_values) * len(args.C_values)
            combo_idx = 0
            
            for k in k_values:
                kk = min(int(k), X.shape[1])
                for C in args.C_values:
                    combo_idx += 1
                    fold_scores = []
                    fold_accs = []
                    fold_per_class_f1s = []
                    
                    for fold, (tr_i, cv_i) in enumerate(skf.split(train_trial_df, train_trial_y)):
                        # CHAY STRATIFIED_KFOLD TREN TRAIN UIDS, KHONG CHAY TREN SEGMENT ROWS
                        fold_train_uids = train_trial_df.iloc[tr_i]["original_trial_uid"].values
                        fold_cv_uids = train_trial_df.iloc[cv_i]["original_trial_uid"].values
                        
                        tr_mask = meta_train["original_trial_uid"].isin(fold_train_uids)
                        cv_mask = meta_train["original_trial_uid"].isin(fold_cv_uids)
                        
                        X_tr, y_tr = X_train[tr_mask], y_train_seg[tr_mask]
                        X_cv, y_cv = X_train[cv_mask], y_train_seg[cv_mask]
                        meta_cv = meta_train[cv_mask].copy()
                        
                        print(f"  [{combo_idx}/{total_combos}] k={kk}, C={C} | Fold {fold+1}/5 ({len(X_tr)} train, {len(X_cv)} val segs)", end="", flush=True)
                        
                        pipe = Pipeline([
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("selector", SelectKBest(score_func=f_classif, k=kk)),
                            ("clf", SVC(kernel="rbf", C=float(C), gamma="scale", class_weight="balanced", probability=True)),
                        ])
                        
                        pipe.fit(X_tr, y_tr)
                        proba_cv = pipe.predict_proba(X_cv)
                        classes = pipe.classes_
                        
                        trial_pred, _ = aggregate_trial_proba(proba_cv, meta_cv, classes)
                        trial_true = get_trial_true_labels(meta_cv, trial_pred, "y_target")
                        
                        score = f1_score(trial_true, trial_pred, average="macro")
                        acc = accuracy_score(trial_true, trial_pred)
                        per_class = f1_score(trial_true, trial_pred, average=None, zero_division=0)
                        
                        fold_scores.append(score)
                        fold_accs.append(acc)
                        fold_per_class_f1s.append(per_class)
                        print(f" -> F1={score:.4f}", flush=True)
                        
                    mean_f1 = np.mean(fold_scores)
                    std_f1 = np.std(fold_scores)
                    mean_acc = np.mean(fold_accs)
                    std_acc = np.std(fold_accs)
                    mean_per_class = np.mean(fold_per_class_f1s, axis=0)
                    std_per_class = np.std(fold_per_class_f1s, axis=0)
                    
                    param_scores.append({
                        "k": kk, "C": C, 
                        "mean_macro_f1": mean_f1, "std_macro_f1": std_f1,
                        "mean_acc": mean_acc, "std_acc": std_acc,
                        "mean_per_class": mean_per_class, "std_per_class": std_per_class
                    })
                    print(f"  => k={kk}, C={C}: Mean F1={mean_f1:.4f} ± {std_f1:.4f}", flush=True)
            
            best_param = max(param_scores, key=lambda d: d["mean_macro_f1"])
            print(f"\n-> BEST PARAMS: k={best_param['k']}, C={best_param['C']}")
            print(f"--- PAPER-STYLE CV RESULTS (5-Fold on Train) ---")
            print(f"Macro Avg:  {best_param['mean_macro_f1']:.4f} ± {best_param['std_macro_f1']:.4f}")
            print(f"Accuracy:   {best_param['mean_acc']:.4f} ± {best_param['std_acc']:.4f}")
            
            class_names = ["Low", "Medium", "High"] if args.task == "3class" else ["Class 0", "Class 1"]
            for i, cname in enumerate(class_names):
                if i < len(best_param['mean_per_class']):
                    print(f"{cname:<10}: {best_param['mean_per_class'][i]:.4f} ± {best_param['std_per_class'][i]:.4f}")
            print("------------------------------------------------\n")
            
            # 4. Train Final Model
            print("[Phase 2] Train Final Model & Validation")
            final_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif, k=best_param['k'])),
                ("clf", SVC(kernel="rbf", C=float(best_param['C']), gamma="scale", class_weight="balanced", probability=True)),
            ])
            # Train model 1 lan duy nhat tren 60% Train segments
            final_pipe.fit(X_train, y_train_seg)
            
            # Predict tren Val segments -> soft voting ve original_trial_uid -> tinh metric o trial-level
            val_proba = final_pipe.predict_proba(X_val)
            val_trial_pred, val_trial_proba_df = aggregate_trial_proba(val_proba, meta_val, final_pipe.classes_)
            val_trial_true = get_trial_true_labels(meta_val, val_trial_pred, "y_target")
            
            val_metrics = eval_trial_metrics(val_trial_true, val_trial_pred, args.task=="binary")
            
            print("[Phase 3] Final Test Evaluation (Strictly Unseen)")
            # Predict tren Test segments -> soft voting ve original_trial_uid -> tinh metric o trial-level
            test_proba = final_pipe.predict_proba(X_test)
            test_trial_pred, test_trial_proba_df = aggregate_trial_proba(test_proba, meta_test, final_pipe.classes_)
            test_trial_true = get_trial_true_labels(meta_test, test_trial_pred, "y_target")
            
            test_metrics = eval_trial_metrics(test_trial_true, test_trial_pred, args.task=="binary")
            print("Final Test Results:")
            print_metrics(test_metrics)
            
            # Save results
            row = {
                "modality": modality,
                "target": target,
                "best_k": best_param["k"],
                "best_C": best_param["C"],
                "cv_mean_macro_f1": best_param["mean_macro_f1"],
                "cv_std_macro_f1": best_param["std_macro_f1"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_acc": val_metrics["acc"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_acc": test_metrics["acc"],
            }
            all_summary_rows.append(row)
            
            base_name = f"{target}_{modality}_k{best_param['k']}_C{best_param['C']}"
            
            cm = confusion_matrix(test_trial_true, test_trial_pred)
            per_class_f1 = f1_score(test_trial_true, test_trial_pred, average=None, zero_division=0)
            
            detail_report = {
                "protocol": "paper_style_60_20_20_trial_split_5fold_tuning",
                "split_level": "original_trial_uid",
                "aggregation": "segment_predict_proba_mean_soft_voting",
                "modality": modality,
                "target": target,
                "task": args.task,
                "best_k": int(best_param["k"]),
                "best_C": float(best_param["C"]),
                "cv_mean_macro_f1": float(best_param["mean_macro_f1"]),
                "cv_std_macro_f1": float(best_param["std_macro_f1"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_acc": float(val_metrics["acc"]),
                "test_macro_f1": float(test_metrics["macro_f1"]),
                "test_acc": float(test_metrics["acc"]),
                "test_weighted_f1": float(test_metrics["weighted_f1"]),
                "test_per_class_f1": per_class_f1.tolist(),
                "test_confusion_matrix": cm.tolist(),
                "n_train_trials": int(len(train_uids)),
                "n_val_trials": int(len(val_uids)),
                "n_test_trials": int(len(test_uids)),
                "n_train_segments": int(len(X_train)),
                "n_val_segments": int(len(X_val)),
                "n_test_segments": int(len(X_test)),
            }

            with open(out_dir / "oof_reports" / f"{base_name}_summary.json", "w", encoding="utf-8") as f:
                json.dump(detail_report, f, indent=2, ensure_ascii=False)

            pd.DataFrame({
                "original_trial_uid": test_trial_true.index,
                "true_label": test_trial_true.values,
                "pred_label": test_trial_pred.loc[test_trial_true.index].values,
            }).to_csv(out_dir / "oof_reports" / f"{base_name}_test_predictions.csv", index=False)
            
            val_trial_proba_df.to_csv(out_dir / "oof_reports" / f"{base_name}_val_proba.csv")
            test_trial_proba_df.to_csv(out_dir / "oof_reports" / f"{base_name}_test_proba.csv")

    df = pd.DataFrame(all_summary_rows)
    summary_path = out_dir / "summary_paper_protocol.csv"
    
    if summary_path.exists() and not df.empty:
        old_df = pd.read_csv(summary_path)
        df = pd.concat([old_df, df], ignore_index=True)
        if all(c in df.columns for c in ["modality", "target", "best_k", "best_C"]):
            df = df.drop_duplicates(subset=["modality", "target", "best_k", "best_C"], keep="last")
            
    df.to_csv(summary_path, index=False)
    print(f"\nSaved final summary to {summary_path}")

if __name__ == "__main__":
    main()
