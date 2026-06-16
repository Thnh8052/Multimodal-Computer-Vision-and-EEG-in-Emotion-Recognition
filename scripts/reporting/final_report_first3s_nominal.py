#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Final nominal report for AFFEC first3s experiments.

What this script does
---------------------
1) Load TRAIN01-A2 RBF-SVM summaries and TRAIN01-A2B XGB/HGB summaries from one
   or more run directories.
2) Build a final comparison table for EEG-only / CV-only / Early Fusion.
3) Select performance-best and robust-best rows per target/scenario/modality.
4) If OOF artifacts exist, export:
   - confusion matrix CSV + PNG
   - classification report CSV
   - OOF metric table
   - optional prediction CSV
5) Write a Markdown summary for the report.

Expected folder examples
------------------------
results\affec_clean\train01_A2_first3s_eeg_finetune_small
results\affec_clean\train01_A2_first3s_early_fusion
results\affec_clean\train01_A2_first3s_boosting_cv

Important
---------
This is still NOMINAL 3-class classification. It reports ordinal_mae as a metric,
but it does not train an ordinal classifier.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)


CLASS_NAMES = {0: "Low", 1: "Med", 2: "High"}
TARGET_SHORT = {
    "f_emotion_a_3class_v2": "arousal",
    "f_emotion_v_3class_v2": "valence",
}

# Default branches you decided to compare in final report.
DEFAULT_KEEP_BRANCHES = [
    # CV
    "cv_au_first3s_no_valid_rate",
    "cv_au_first3s_no_valid_no_p95",
    "cv_au_first3s_no_au04_static",
    # EEG
    "eeg_first3s_de_relbp_dasm_dynamic",
    "eeg_first3s_de_relbp_dasm_rasm_dynamic",
    "eeg_first3s_asym_dasm_rasm",
    # Early fusion
    "fusion_eeg_first3s_de_relbp_dasm_dynamic__cv_au_first3s_no_valid_rate",
    "fusion_eeg_first3s_de_relbp_dasm_dynamic__cv_au_first3s_no_valid_no_p95",
    "fusion_eeg_first3s_asym_dasm_rasm__cv_au_first3s_no_valid_no_p95",
    "fusion_eeg_first3s_asym_dasm_rasm__cv_au_first3s_no_au04_static",
]


# -----------------------------------------------------------------------------
# I/O utilities
# -----------------------------------------------------------------------------

def safe_name(s: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_list_arg(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def find_manifest(input_dir: Path, manifest_name: str = "") -> Tuple[Dict[str, Any], Path]:
    candidates: List[Path] = []
    if manifest_name:
        candidates.append(input_dir / manifest_name)
    candidates += [
        input_dir / "train01_00B_input_manifest.json",
        input_dir / "feature_registry_train01_00B.json",
        input_dir / "train01_00_input_manifest.json",
    ]
    for p in candidates:
        if p.exists():
            return read_json(p), p
    raise FileNotFoundError(f"Could not find manifest in {input_dir}")


def load_target_and_masks(input_dir: Path, manifest: Dict[str, Any], target: str, scenario: str) -> Tuple[np.ndarray, np.ndarray]:
    if "targets" not in manifest or target not in manifest["targets"]:
        raise KeyError(f"Target {target} not found in manifest['targets']")
    y_path = input_dir / manifest["targets"][target]["path"]
    y = np.load(y_path)

    valid = np.ones(len(y), dtype=bool)
    target_mask_path = manifest["targets"][target].get("mask_valid")
    if target_mask_path:
        valid &= np.load(input_dir / target_mask_path).astype(bool)

    if "masks" in manifest and scenario in manifest["masks"]:
        valid &= np.load(input_dir / manifest["masks"][scenario]["path"]).astype(bool)
    elif scenario == "main_all_trials":
        pass
    else:
        raise KeyError(f"Scenario {scenario} not found in manifest['masks']")
    return y.astype(int), valid


# -----------------------------------------------------------------------------
# Load experiment summaries
# -----------------------------------------------------------------------------

def modality_from_branch(branch: str, existing: str = "") -> str:
    if isinstance(existing, str) and existing and existing != "unknown":
        # Normalize older labels.
        if existing in {"eeg", "cv", "fusion", "late_fusion"}:
            return existing
    b = str(branch)
    if b.startswith("eeg_"):
        return "eeg"
    if b.startswith("cv_"):
        return "cv"
    if b.startswith("fusion_"):
        return "fusion"
    if b.startswith("late_"):
        return "late_fusion"
    return existing or "unknown"


def summary_kind_from_filename(path: Path) -> str:
    name = path.name.lower()
    if "boosting" in name:
        return "boosting_tree"
    if "late_fusion" in name:
        return "late_fusion"
    return "svm_rbf_mlp"


def find_summary_files(run_dir: Path) -> List[Path]:
    candidates = [
        run_dir / "train01A2_first3s_all_summary.csv",
        run_dir / "train01A2B_first3s_boosting_summary.csv",
        run_dir / "train01A2_first3s_late_fusion_summary.csv",
    ]
    existing = [p for p in candidates if p.exists()]
    if existing:
        return existing
    # Fallback: any likely summary file, but avoid per_fold and overfit/debug.
    files = []
    for p in run_dir.glob("*.csv"):
        n = p.name.lower()
        if "summary" in n and "per_fold" not in n and "overfit" not in n and "debug" not in n:
            files.append(p)
    return sorted(files)


def load_all_summaries(run_dirs: Sequence[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for rd in run_dirs:
        files = find_summary_files(rd)
        if not files:
            print(f"[WARN] No summary CSV found in: {rd}")
            continue
        for p in files:
            try:
                df = pd.read_csv(p)
            except Exception as e:
                print(f"[WARN] Could not read {p}: {e}")
                continue
            if df.empty:
                continue
            df["source_run_dir"] = str(rd)
            df["source_summary_csv"] = str(p)
            df["source_kind"] = summary_kind_from_filename(p)
            if "branch" not in df.columns and "eeg_branch" in df.columns and "cv_branch" in df.columns:
                df["branch"] = "late_" + df["eeg_branch"].astype(str) + "__" + df["cv_branch"].astype(str)
            if "modality" not in df.columns:
                df["modality"] = df["branch"].map(lambda b: modality_from_branch(str(b)))
            else:
                df["modality"] = [modality_from_branch(b, m) for b, m in zip(df["branch"], df["modality"])]
            if "normalization" not in df.columns:
                df["normalization"] = "unknown"
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    # Numeric coercion for common metrics.
    for c in out.columns:
        if c.endswith("_mean") or c.endswith("_std") or c in {"n_folds", "test_n_total"}:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def filter_summary(
    df: pd.DataFrame,
    scenarios: Sequence[str],
    targets: Sequence[str],
    normalizations: Sequence[str],
    keep_branches: Sequence[str],
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if scenarios:
        out = out[out["scenario"].isin(scenarios)]
    if targets:
        out = out[out["target"].isin(targets)]
    if normalizations and "normalization" in out.columns:
        out = out[out["normalization"].isin(normalizations)]
    if keep_branches:
        out = out[out["branch"].isin(keep_branches)]
    if "test_macro_f1_mean" in out.columns:
        out = out.sort_values(
            ["test_macro_f1_mean", "gap_macro_f1_mean"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Select final candidates
# -----------------------------------------------------------------------------

def select_candidates(df: pd.DataFrame, robust_gap: float, top_n: int = 1) -> pd.DataFrame:
    """Return performance-best and robust-best rows by scenario/target/modality."""
    if df.empty:
        return pd.DataFrame()
    required = {"scenario", "target", "modality", "test_macro_f1_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in summary: {missing}")

    rows: List[pd.DataFrame] = []
    group_cols = ["scenario", "target", "modality"]
    for keys, g in df.groupby(group_cols, dropna=False):
        gg = g.sort_values(["test_macro_f1_mean", "gap_macro_f1_mean"], ascending=[False, True], na_position="last")
        perf = gg.head(top_n).copy()
        perf["selection_type"] = "performance_best"
        rows.append(perf)

        if "gap_macro_f1_mean" in gg.columns:
            rg = gg[gg["gap_macro_f1_mean"].fillna(999) <= robust_gap]
        else:
            rg = pd.DataFrame()
        if not rg.empty:
            rob = rg.sort_values(["test_macro_f1_mean", "gap_macro_f1_mean"], ascending=[False, True]).head(top_n).copy()
            rob["selection_type"] = f"robust_gap_le_{robust_gap:g}"
            rows.append(rob)

    if not rows:
        return pd.DataFrame()
    cand = pd.concat(rows, ignore_index=True, sort=False)
    # Drop exact duplicate candidates but keep multiple selection labels if needed.
    key_cols = [
        "selection_type", "scenario", "target", "normalization", "modality", "branch", "model_name", "source_run_dir"
    ]
    existing = [c for c in key_cols if c in cand.columns]
    cand = cand.drop_duplicates(existing).reset_index(drop=True)
    return cand.sort_values(["target", "scenario", "modality", "selection_type"]).reset_index(drop=True)


def compact_report_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "selection_type", "scenario", "target", "modality", "branch", "model_name", "model_spec",
        "test_macro_f1_mean", "test_macro_f1_std", "test_weighted_f1_mean", "test_balanced_acc_mean",
        "test_ordinal_mae_mean", "gap_macro_f1_mean", "n_folds", "test_n_total", "source_kind", "source_run_dir",
    ]
    return df[[c for c in cols if c in df.columns]].copy()


# -----------------------------------------------------------------------------
# OOF report generation
# -----------------------------------------------------------------------------

def find_oof_dir(row: pd.Series) -> Optional[Path]:
    rd = Path(str(row["source_run_dir"]))
    scenario = safe_name(row["scenario"])
    target = safe_name(row["target"])
    branch = safe_name(row["branch"])
    model_name = safe_name(row["model_name"])
    p = rd / "oof" / scenario / target / branch / model_name
    if p.exists() and (p / "oof_pred.npy").exists():
        return p
    return None


def compute_nominal_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> Dict[str, float]:
    return {
        "n": int(len(y_true)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(labels), average="weighted", zero_division=0)),
        "ordinal_mae": float(mean_absolute_error(y_true, y_pred)),
        "far_error_rate": float(np.mean(np.abs(y_true - y_pred) >= 2)),
        "within_1_rate": float(np.mean(np.abs(y_true - y_pred) <= 1)),
    }


def save_confusion_matrix_png(cm: np.ndarray, labels: Sequence[int], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    names = [CLASS_NAMES.get(int(x), str(x)) for x in labels]
    ax.set_xticks(np.arange(len(labels)), labels=names)
    ax.set_yticks(np.arange(len(labels)), labels=names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    max_val = cm.max() if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def export_oof_for_candidate(
    row: pd.Series,
    input_dir: Path,
    manifest: Dict[str, Any],
    out_dir: Path,
    save_predictions: bool = False,
) -> Optional[Dict[str, Any]]:
    oof_dir = find_oof_dir(row)
    if oof_dir is None:
        return None

    scenario = str(row["scenario"])
    target = str(row["target"])
    branch = str(row["branch"])
    model_name = str(row["model_name"])
    modality = str(row.get("modality", modality_from_branch(branch)))
    selection_type = str(row.get("selection_type", "selected"))

    y_all, valid_mask = load_target_and_masks(input_dir, manifest, target, scenario)
    pred = np.load(oof_dir / "oof_pred.npy")
    mask = valid_mask & (pred >= 0)
    if mask.sum() == 0:
        return None
    y_true = y_all[mask].astype(int)
    y_pred = pred[mask].astype(int)

    meta = {}
    if (oof_dir / "oof_meta.json").exists():
        meta = read_json(oof_dir / "oof_meta.json")
    labels = [int(x) for x in meta.get("classes", sorted(np.unique(np.concatenate([y_true, y_pred])).tolist()))]
    labels = sorted(labels)

    base_name = "__".join([
        safe_name(selection_type),
        safe_name(scenario),
        safe_name(TARGET_SHORT.get(target, target)),
        safe_name(modality),
        safe_name(branch)[:80],
        safe_name(model_name)[:80],
    ])
    d = mkdir(out_dir / "oof_reports" / base_name)

    metrics = compute_nominal_metrics(y_true, y_pred, labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{CLASS_NAMES.get(i, i)}" for i in labels], columns=[f"pred_{CLASS_NAMES.get(i, i)}" for i in labels])
    cm_df.to_csv(d / "confusion_matrix.csv", encoding="utf-8-sig")
    save_confusion_matrix_png(cm, labels, f"{TARGET_SHORT.get(target, target)} | {modality}\n{branch}\n{model_name}", d / "confusion_matrix.png")

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[CLASS_NAMES.get(i, str(i)) for i in labels],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    report_df.to_csv(d / "classification_report.csv", encoding="utf-8-sig")

    if save_predictions:
        pred_df = pd.DataFrame({
            "index": np.where(mask)[0],
            "y_true": y_true,
            "y_pred": y_pred,
            "abs_error": np.abs(y_true - y_pred),
        })
        proba_path = oof_dir / "oof_proba.npy"
        if proba_path.exists():
            proba = np.load(proba_path)[mask]
            for j, lab in enumerate(labels):
                if j < proba.shape[1]:
                    pred_df[f"proba_{CLASS_NAMES.get(lab, lab)}"] = proba[:, j]
        pred_df.to_csv(d / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    info = {
        "selection_type": selection_type,
        "scenario": scenario,
        "target": target,
        "target_short": TARGET_SHORT.get(target, target),
        "modality": modality,
        "branch": branch,
        "model_name": model_name,
        "model_spec": str(row.get("model_spec", "")),
        "source_run_dir": str(row.get("source_run_dir", "")),
        "oof_dir": str(oof_dir),
        "report_dir": str(d.relative_to(out_dir)),
        **metrics,
    }
    write_json(info, d / "oof_metrics.json")
    return info


# -----------------------------------------------------------------------------
# Markdown summary
# -----------------------------------------------------------------------------

def fmt_float(x: Any, digits: int = 4) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def write_markdown_summary(
    out_dir: Path,
    all_filtered: pd.DataFrame,
    candidates: pd.DataFrame,
    oof_metrics: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines: List[str] = []
    lines.append("# AFFEC First3s Final Nominal Report")
    lines.append("")
    lines.append(f"Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- EEG/CV/Fusion are all based on **first 3 seconds**.")
    lines.append("- Classification type: **nominal 3-class** (`Low`, `Med`, `High`).")
    lines.append("- `ordinal_mae` is reported as an evaluation metric only; this script does not train ordinal classification.")
    lines.append("- Main split: subject-dependent pooled joint-stratified 5-fold, according to the supplied input manifest/split.")
    lines.append("")

    lines.append("## Selected candidates")
    lines.append("")
    if candidates.empty:
        lines.append("No candidates selected.")
    else:
        show = compact_report_columns(candidates)
        display_cols = [c for c in [
            "selection_type", "scenario", "target", "modality", "branch", "model_name",
            "test_macro_f1_mean", "test_weighted_f1_mean", "test_balanced_acc_mean",
            "test_ordinal_mae_mean", "gap_macro_f1_mean"
        ] if c in show.columns]
        lines.append(show[display_cols].to_markdown(index=False))
    lines.append("")

    if not oof_metrics.empty:
        lines.append("## OOF nominal metrics")
        lines.append("")
        display_cols = [
            "selection_type", "scenario", "target_short", "modality", "branch", "model_name",
            "macro_f1", "weighted_f1", "balanced_acc", "ordinal_mae", "far_error_rate", "n", "report_dir",
        ]
        lines.append(oof_metrics[[c for c in display_cols if c in oof_metrics.columns]].to_markdown(index=False))
        lines.append("")
        lines.append("Confusion matrix PNG/CSV and classification report CSV are saved under `oof_reports/`.")
        lines.append("")
    else:
        lines.append("## OOF reports")
        lines.append("")
        lines.append("No OOF artifacts were found for selected candidates. The comparison table was still generated from summary CSVs.")
        lines.append("")

    lines.append("## Best rows by macro-F1 from loaded summaries")
    lines.append("")
    if not all_filtered.empty:
        top_cols = [c for c in [
            "scenario", "target", "modality", "branch", "model_name", "test_macro_f1_mean",
            "test_weighted_f1_mean", "test_balanced_acc_mean", "test_ordinal_mae_mean", "gap_macro_f1_mean", "source_kind"
        ] if c in all_filtered.columns]
        lines.append(all_filtered.sort_values("test_macro_f1_mean", ascending=False).head(args.top_rows_md)[top_cols].to_markdown(index=False))
    lines.append("")

    (out_dir / "final_nominal_report.md").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate final nominal report from first3s SVM/XGB/HGB runs.")
    ap.add_argument("--input_dir", required=True, help="TRAIN01-00B first3s input directory")
    ap.add_argument("--manifest", default="", help="Manifest filename inside input_dir")
    ap.add_argument("--run_dirs", required=True, help="Comma-separated run directories containing summary CSVs and optionally oof/")
    ap.add_argument("--out_dir", required=True, help="Output report directory")
    ap.add_argument("--scenarios", default="main_all_trials,cv_valid_only")
    ap.add_argument("--targets", default="f_emotion_a_3class_v2,f_emotion_v_3class_v2")
    ap.add_argument("--normalizations", default="none")
    ap.add_argument("--keep_branches", default=",".join(DEFAULT_KEEP_BRANCHES), help="Comma-separated branches to keep. Use empty string to keep all.")
    ap.add_argument("--robust_gap", type=float, default=0.18, help="Max gap_macro_f1_mean for robust candidate")
    ap.add_argument("--save_predictions", default="0", choices=["0", "1"], help="Also save OOF per-sample predictions")
    ap.add_argument("--top_rows_md", type=int, default=30)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = mkdir(Path(args.out_dir))
    run_dirs = [Path(p) for p in parse_list_arg(args.run_dirs)]
    scenarios = parse_list_arg(args.scenarios)
    targets = parse_list_arg(args.targets)
    normalizations = parse_list_arg(args.normalizations)
    keep_branches = parse_list_arg(args.keep_branches)
    save_predictions = args.save_predictions == "1"

    manifest, manifest_path = find_manifest(input_dir, args.manifest)

    print("[INFO] Loading summaries...")
    all_summary = load_all_summaries(run_dirs)
    if all_summary.empty:
        raise SystemExit("[ERROR] No summary rows loaded. Check --run_dirs.")

    all_summary.to_csv(out_dir / "all_loaded_summary_raw.csv", index=False, encoding="utf-8-sig")

    filtered = filter_summary(
        all_summary,
        scenarios=scenarios,
        targets=targets,
        normalizations=normalizations,
        keep_branches=keep_branches,
    )
    if filtered.empty:
        raise SystemExit("[ERROR] No rows after filtering. Check scenarios/targets/branches/normalizations.")

    filtered.to_csv(out_dir / "all_loaded_summary_filtered.csv", index=False, encoding="utf-8-sig")

    print("[INFO] Selecting candidates...")
    candidates = select_candidates(filtered, robust_gap=args.robust_gap, top_n=1)
    compact = compact_report_columns(candidates)
    compact.to_csv(out_dir / "final_selected_candidates.csv", index=False, encoding="utf-8-sig")

    # Also useful: a concise best-per-target/scenario/modality table without selection duplicates.
    perf_only = candidates[candidates["selection_type"] == "performance_best"].copy()
    compact_report_columns(perf_only).to_csv(out_dir / "final_performance_best_table.csv", index=False, encoding="utf-8-sig")

    robust_only = candidates[candidates["selection_type"].astype(str).str.startswith("robust")].copy()
    compact_report_columns(robust_only).to_csv(out_dir / "final_robust_best_table.csv", index=False, encoding="utf-8-sig")

    print("[INFO] Exporting OOF reports when available...")
    oof_rows: List[Dict[str, Any]] = []
    for _, row in candidates.iterrows():
        try:
            info = export_oof_for_candidate(row, input_dir, manifest, out_dir, save_predictions=save_predictions)
            if info is None:
                print(f"[WARN] No OOF found for {row.get('scenario')} | {row.get('target')} | {row.get('branch')} | {row.get('model_name')}")
            else:
                oof_rows.append(info)
        except Exception as e:
            print(f"[WARN] Failed OOF export for {row.get('branch')} | {row.get('model_name')}: {e}")

    oof_metrics = pd.DataFrame(oof_rows)
    if not oof_metrics.empty:
        oof_metrics.to_csv(out_dir / "final_oof_metrics.csv", index=False, encoding="utf-8-sig")
    else:
        # Create empty file for consistency.
        pd.DataFrame().to_csv(out_dir / "final_oof_metrics.csv", index=False, encoding="utf-8-sig")

    write_json({
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_dir": str(input_dir),
        "manifest": str(manifest_path),
        "run_dirs": [str(x) for x in run_dirs],
        "out_dir": str(out_dir),
        "scenarios": scenarios,
        "targets": targets,
        "normalizations": normalizations,
        "keep_branches": keep_branches,
        "robust_gap": args.robust_gap,
        "n_summary_rows_raw": int(len(all_summary)),
        "n_summary_rows_filtered": int(len(filtered)),
        "n_candidates": int(len(candidates)),
        "n_oof_reports": int(len(oof_metrics)),
    }, out_dir / "final_report_config.json")

    write_markdown_summary(out_dir, filtered, candidates, oof_metrics, args)

    print("[DONE]")
    print(f"  - {out_dir / 'final_nominal_report.md'}")
    print(f"  - {out_dir / 'final_selected_candidates.csv'}")
    print(f"  - {out_dir / 'final_oof_metrics.csv'}")
    print(f"  - {out_dir / 'oof_reports'}")


if __name__ == "__main__":
    main()
