#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
AFFEC CLEAN EDA 01D - Expand first-3s into sliding-window samples (ICA+Clip) + AU-only CV features
================================================================

Mục tiêu
--------
Extract trial-level feature theo thiết kế clean sau EDA00:

EEG main:
  - CORE18 channels
  - interval main = first 3 seconds of video, expanded into 1s sliding-window samples by default
  - bands: theta/alpha/beta/gamma_low, filter 4-45 Hz
  - actual EDF sfreq -> resample 256 Hz
  - band power internals, DE, relative power
  - DASM_de + logRASM from band power ratio
  - region average
  - per-segment features by default: one sample per 1s window; no trial-level temporal aggregation
  - không dùng all63, không dùng delta band, không dùng absolute logBP feature, không dùng raw max/min/peak

CV main:
  - AU-only, 17 AUs
  - interval main = same sliding-window sample interval as EEG
  - AU_r: p95 / median / iqr / slope / valid_rate, clip [0,5]
  - AU_c: presence_rate / longest_on_ratio / transition_rate
  - dùng timestamp thật, không resample CV về EEG

Output chính
------------
  metadata_trials_clean.csv
  trial_feature_quality_clean.csv
  eeg_processing_log.csv
  cv_processing_log.csv
  error_log.csv
  feature_registry_clean.json
  eda_clean_01_summary.txt

  eeg/X_eeg_core18_first3s_clean.npy
  eeg/feature_names_eeg_core18_first3s_clean.json
  eeg/X_eeg_core18_first3s_second_fix_delta.npy       # optional; NaN nếu thiếu second_fix
  eeg/feature_names_eeg_core18_first3s_second_fix_delta.json

  cv/X_cv_au_only_first3s.npy
  cv/feature_names_cv_au_only_first3s.json

Ví dụ chạy
----------
python eda_clean_01_extract_core18_auonly_features.py ^
  --raw_root data\raw ^
  --eda00_dir results\affec_clean\00_manifest_feature_audit_video ^
  --out_dir results\affec_clean\01_clean_features_core18_auonly

Chạy nhanh debug:
python eda_clean_01_extract_core18_auonly_features.py ^
  --raw_root data\raw ^
  --eda00_dir results\affec_clean\00_manifest_feature_audit_video ^
  --out_dir results\affec_clean\01_debug ^
  --max_runs 5
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# Clean design constants
# =============================================================================

CORE18 = [
    "Fp1", "Fp2",
    "AF3", "AF4",
    "F3", "F4",
    "F7", "F8",
    "FC5", "FC6",
    "T7", "T8",
    "C3", "C4",
    "P3", "P4",
    "O1", "O2",
]

ASYM_PAIRS = [
    ("Fp1", "Fp2"),
    ("AF3", "AF4"),
    ("F3", "F4"),
    ("F7", "F8"),
    ("FC5", "FC6"),
    ("T7", "T8"),
    ("C3", "C4"),
    ("P3", "P4"),
    ("O1", "O2"),
]

REGION_GROUPS = {
    "frontopolar_frontal": ["Fp1", "Fp2", "AF3", "AF4", "F3", "F4", "F7", "F8", "FC5", "FC6"],
    "temporal": ["T7", "T8"],
    "central": ["C3", "C4"],
    "parietal": ["P3", "P4"],
    "occipital": ["O1", "O2"],
}

EEG_BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma_low": (30.0, 45.0),
}

CORE_CV_AU = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07",
    "AU09", "AU10", "AU12", "AU14", "AU15", "AU17",
    "AU20", "AU23", "AU25", "AU26", "AU45",
]

AU_R_STATS = ["p95", "median", "iqr", "slope", "valid_rate"]
AU_C_STATS = ["presence_rate", "longest_on_ratio", "transition_rate"]
EEG_ROBUST_STATS = ["median", "iqr", "trimmed_mean", "slope", "last_minus_first", "p95"]
MAIN_TARGETS = ["f_emotion_a_3class_v2", "f_emotion_v_3class_v2"]

EPS = 1e-12

# =============================================================================
# Generic helpers
# =============================================================================

def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def norm_subject(x: Any) -> str:
    s = str(x).strip()
    if not s.startswith("sub-") and re.match(r"^[A-Za-z0-9]+$", s):
        s = "sub-" + s
    return s


def norm_run(x: Any) -> str:
    try:
        return str(int(float(x)))
    except Exception:
        m = re.search(r"run[-_]?(\d+)", str(x), flags=re.IGNORECASE)
        if m:
            return str(int(m.group(1)))
        return str(x).strip()


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s)).strip("_")


def to_float(x: Any, default: float = np.nan) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def parse_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_path(raw_root: Path, value: Any) -> Optional[Path]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute() and p.exists():
        return p
    p2 = raw_root / s
    if p2.exists():
        return p2
    # Last fallback: sometimes CSV stores path with raw_root prefix but not absolute.
    p3 = Path(s.replace("\\", "/"))
    if p3.exists():
        return p3
    return None


def dicts_to_matrix(rows: List[Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
    keys = sorted({k for r in rows for k in r.keys()})
    X = np.full((len(rows), len(keys)), np.nan, dtype=np.float32)
    key_to_j = {k: j for j, k in enumerate(keys)}
    for i, r in enumerate(rows):
        for k, v in r.items():
            j = key_to_j.get(k)
            if j is None:
                continue
            try:
                fv = float(v)
                X[i, j] = fv if math.isfinite(fv) else np.nan
            except Exception:
                X[i, j] = np.nan
    return X, keys


def save_feature_matrix(out_dir: Path, subdir: str, stem: str, rows: List[Dict[str, float]]) -> Tuple[Path, Path, np.ndarray, List[str]]:
    d = mkdir(out_dir / subdir)
    X, names = dicts_to_matrix(rows)
    x_path = d / f"X_{stem}.npy"
    n_path = d / f"feature_names_{stem}.json"
    np.save(x_path, X)
    write_json(names, n_path)
    return x_path, n_path, X, names


def finite_summary(X: np.ndarray) -> Dict[str, Any]:
    finite = np.isfinite(X)
    vals = X[finite]
    return {
        "shape": list(X.shape),
        "nan_rate": float(1.0 - finite.mean()) if X.size else np.nan,
        "all_nan_rows": int(np.sum(~np.isfinite(X).any(axis=1))) if X.ndim == 2 else 0,
        "finite_min": float(np.min(vals)) if vals.size else np.nan,
        "finite_max": float(np.max(vals)) if vals.size else np.nan,
        "finite_mean": float(np.mean(vals)) if vals.size else np.nan,
    }

# =============================================================================
# Load EDA00 outputs
# =============================================================================

def load_manifest(eda00_dir: Path, manifest_csv: str = "") -> pd.DataFrame:
    path = Path(manifest_csv) if manifest_csv else eda00_dir / "final_trial_index_clean.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    df = pd.read_csv(path)
    df["subject"] = df["subject"].map(norm_subject)
    df["run"] = df["run"].map(norm_run)
    for c in [
        "video_interval_start", "video_interval_end", "video_interval_duration",
        "baseline_second_fix_start", "baseline_second_fix_end", "baseline_second_fix_duration",
        "keep_clean_rawstim_video", "keep_clean_second_fix_delta",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "video_interval_start" not in df.columns and "video_start" in df.columns:
        df["video_interval_start"] = pd.to_numeric(df["video_start"], errors="coerce")
        df["video_interval_end"] = pd.to_numeric(df["video_end"], errors="coerce")
        df["video_interval_duration"] = df["video_interval_end"] - df["video_interval_start"]
    keep_col = "keep_clean_rawstim_video"
    if keep_col in df.columns:
        df = df[df[keep_col].astype(float) == 1].copy()
    df = df.sort_values(["subject", "run", "trial"]).reset_index(drop=True)
    return df


def load_run_paths(raw_root: Path, eda00_dir: Path, run_file_coverage_csv: str = "") -> Dict[Tuple[str, str], Dict[str, Path]]:
    path = Path(run_file_coverage_csv) if run_file_coverage_csv else eda00_dir / "run_file_coverage.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing run_file_coverage.csv: {path}")
    df = pd.read_csv(path)
    paths: Dict[Tuple[str, str], Dict[str, Path]] = {}
    for _, r in df.iterrows():
        subject = norm_subject(r.get("subject"))
        run = norm_run(r.get("run"))
        item: Dict[str, Path] = {}
        for out_name, col in [
            ("edf", "path_eeg_edf"),
            ("cv_tsv", "path_videostream_tsv"),
            ("cv_json", "path_videostream_json"),
        ]:
            if col in df.columns:
                p = resolve_path(raw_root, r.get(col))
                if p is not None:
                    item[out_name] = p
        paths[(subject, run)] = item
    return paths


def load_cv_offsets(path_str: str) -> Dict[Tuple[str, str], float]:
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: Dict[Tuple[str, str], float] = {}
    for _, r in df.iterrows():
        subject = norm_subject(r.get("subject"))
        run = norm_run(r.get("run"))
        # Prefer commonly used columns.
        offset = np.nan
        for c in ["offset_sec", "best_offset_sec", "cv_offset_sec"]:
            if c in df.columns:
                offset = to_float(r.get(c))
                if math.isfinite(offset):
                    break
        if math.isfinite(offset):
            out[(subject, run)] = float(offset)
    return out


# =============================================================================
# Segment expansion: first-3s -> multiple independent sliding-window rows
# =============================================================================

def _sliding_offsets(total_sec: float, window_sec: float, step_sec: float, include_tail: bool = False) -> List[float]:
    if total_sec <= 0 or window_sec <= 0 or step_sec <= 0:
        return []
    if total_sec + 1e-9 < window_sec:
        return []
    max_start = max(0.0, float(total_sec) - float(window_sec))
    offsets: List[float] = []
    cur = 0.0
    while cur <= max_start + 1e-9:
        offsets.append(round(cur, 10))
        cur += float(step_sec)
    if include_tail:
        tail = round(max_start, 10)
        if offsets and abs(offsets[-1] - tail) > 1e-6:
            offsets.append(tail)
        elif not offsets:
            offsets = [tail]
    return sorted(set(offsets))


def expand_manifest_first3s_to_segment_rows(
    manifest: pd.DataFrame,
    first_sec: float,
    segment_window_sec: float,
    segment_step_sec: float,
    include_tail: bool = False,
    max_segments_per_trial: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Expand each original trial into multiple segment rows inside the first-N seconds.

    Default first_sec=3, segment_window_sec=1, segment_step_sec=0.5 gives 5 rows:
      [0.0,1.0], [0.5,1.5], [1.0,2.0], [1.5,2.5], [2.0,3.0]

    Each row inherits the trial labels. For training/evaluation, group by subject or
    original_trial_uid; never random-split these generated rows directly.
    """
    if first_sec <= 0:
        raise ValueError("first_sec must be > 0")
    if segment_window_sec <= 0:
        raise ValueError("segment_window_sec must be > 0")
    if segment_step_sec <= 0:
        raise ValueError("segment_step_sec must be > 0")
    if segment_window_sec > first_sec + 1e-9:
        raise ValueError("segment_window_sec must be <= first_sec")

    df = manifest.copy()
    if "video_interval_duration" not in df.columns:
        df["video_interval_duration"] = pd.to_numeric(df["video_interval_end"], errors="coerce") - pd.to_numeric(df["video_interval_start"], errors="coerce")

    rows: List[pd.Series] = []
    dropped: List[pd.Series] = []
    offsets = _sliding_offsets(float(first_sec), float(segment_window_sec), float(segment_step_sec), include_tail=include_tail)
    if max_segments_per_trial and max_segments_per_trial > 0:
        offsets = offsets[: int(max_segments_per_trial)]

    for _, r in df.iterrows():
        v_start = to_float(r.get("video_interval_start"))
        v_end = to_float(r.get("video_interval_end"))
        dur = to_float(r.get("video_interval_duration"), v_end - v_start if math.isfinite(v_start) and math.isfinite(v_end) else np.nan)
        if not (math.isfinite(v_start) and math.isfinite(v_end) and v_end > v_start and math.isfinite(dur)):
            dropped.append(r)
            continue
        if dur + 1e-9 < float(first_sec):
            dropped.append(r)
            continue

        original_uid = str(r.get("trial_uid", f"{norm_subject(r.get('subject'))}_run-{norm_run(r.get('run'))}_trial-{r.get('trial')}"))
        for wi, off in enumerate(offsets):
            seg_start = float(v_start) + float(off)
            seg_end = seg_start + float(segment_window_sec)
            if seg_end > float(v_start) + float(first_sec) + 1e-9:
                continue
            rr = r.copy()
            rr["original_trial_uid"] = original_uid
            rr["original_trial"] = r.get("trial")
            rr["segment_idx"] = int(wi)
            rr["segment_start_offset_sec"] = float(off)
            rr["segment_end_offset_sec"] = float(off + segment_window_sec)
            rr["segment_start_abs"] = float(seg_start)
            rr["segment_end_abs"] = float(seg_end)
            rr["segment_duration_sec"] = float(segment_window_sec)
            rr["first_sec_context"] = float(first_sec)
            # Keep old names too, so the rest of the extraction loop can reuse them.
            rr["eeg_first3s_start"] = float(seg_start)
            rr["eeg_first3s_end"] = float(seg_end)
            rr["eeg_first3s_duration"] = float(segment_window_sec)
            rr["eeg_branch_interval"] = f"first_{first_sec:g}s_as_{segment_window_sec:g}s_step_{segment_step_sec:g}s_segments"
            rr["trial_uid"] = f"{original_uid}_seg{wi:03d}_{off:.2f}-{off + segment_window_sec:.2f}s"
            rows.append(rr)

    out = pd.DataFrame(rows).reset_index(drop=True) if rows else df.iloc[0:0].copy()
    dropped_df = pd.DataFrame(dropped).reset_index(drop=True) if dropped else df.iloc[0:0].copy()
    if len(out):
        out["n_segments_for_original_trial"] = out.groupby("original_trial_uid")["segment_idx"].transform("count").astype(int)
    return out, dropped_df

# =============================================================================
# CV/OpenFace AU-only feature extraction
# =============================================================================

def sidecar_columns(json_path: Optional[Path]) -> List[str]:
    if json_path is None or not json_path.exists():
        return []
    try:
        obj = read_json(json_path)
        cols = obj.get("Columns", []) if isinstance(obj, dict) else []
        return [str(c).strip() for c in cols]
    except Exception:
        return []


def cv_selected_columns(all_cols: Sequence[str]) -> List[str]:
    cols = set(map(str, all_cols))
    selected: List[str] = []
    for c in ["onset", "timestamp", "frame", "confidence", "success", "face_id"]:
        if c in cols:
            selected.append(c)
    for au in CORE_CV_AU:
        for suffix in ["_r", "_c"]:
            c = au + suffix
            if c in cols:
                selected.append(c)
    out = []
    seen = set()
    for c in selected:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def read_cv_selected(cv_path: Path, json_path: Optional[Path]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cols = sidecar_columns(json_path)
    log: Dict[str, Any] = {
        "cv_path": str(cv_path),
        "cv_json_path": str(json_path) if json_path else "",
        "sidecar_columns": len(cols),
        "read_mode": "",
    }
    selected = cv_selected_columns(cols)
    if cols and selected:
        col_to_idx = {c: i for i, c in enumerate(cols)}
        ordered_pairs = sorted((col_to_idx[c], c) for c in selected)
        use_idx = [idx for idx, _ in ordered_pairs]
        ordered_names = [name for _, name in ordered_pairs]
        try:
            df = pd.read_csv(cv_path, sep="\t", header=None, usecols=use_idx, low_memory=False)
            df.columns = ordered_names
            df = df[selected].copy()
            log["read_mode"] = "sidecar_header_none_usecols_ordered"
            log["selected_columns"] = ",".join(selected)
            log["n_rows"] = int(len(df))
            log["n_cols_loaded"] = int(df.shape[1])
            return df, log
        except Exception as e:
            log["sidecar_usecols_error"] = repr(e)

    df = pd.read_csv(cv_path, sep="\t", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    selected2 = cv_selected_columns(list(df.columns))
    if selected2:
        df = df[selected2].copy()
    log["read_mode"] = "fallback_header"
    log["selected_columns"] = ",".join(selected2)
    log["n_rows"] = int(len(df))
    log["n_cols_loaded"] = int(df.shape[1])
    return df, log


def infer_cv_time(df: pd.DataFrame, fps_hint: float = 0.0) -> Tuple[np.ndarray, str, float]:
    t: Optional[np.ndarray] = None
    time_col = ""
    for c in ["onset", "timestamp", "time", "frame_time", "seconds"]:
        if c in df.columns:
            arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(arr).sum() >= 2:
                t = arr - np.nanmin(arr)
                time_col = c
                break
    if t is None:
        if "frame" not in df.columns or fps_hint <= 0:
            raise ValueError("Cannot infer CV time: no onset/timestamp and no frame+fps_hint")
        frame = pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=float)
        t = frame / float(fps_hint)
        time_col = "frame/fps"
    ok = np.isfinite(t)
    if ok.sum() < 2:
        raise ValueError("CV time has fewer than 2 finite values")
    diffs = np.diff(np.sort(t[ok]))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0) & (diffs < 2.0)]
    fps_est = float(1.0 / np.median(diffs)) if len(diffs) else (float(fps_hint) if fps_hint > 0 else np.nan)
    return t, time_col, fps_est


def sort_cv_by_time(df: pd.DataFrame, t: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    ok = np.isfinite(t)
    df2 = df.loc[ok].copy()
    t2 = np.asarray(t[ok], dtype=float)
    order = np.argsort(t2)
    return df2.iloc[order].reset_index(drop=True), t2[order]


def continuous_au_r_stats(values: np.ndarray, times_rel: np.ndarray, denom_n: int) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    t = np.asarray(times_rel, dtype=float)
    ok = np.isfinite(v) & np.isfinite(t)
    valid_n = int(ok.sum())
    out = {k: np.nan for k in AU_R_STATS}
    if valid_n == 0:
        return out
    vv = np.clip(v[ok], 0.0, 5.0)
    tt = t[ok]
    q25 = float(np.nanpercentile(vv, 25))
    q75 = float(np.nanpercentile(vv, 75))
    out.update({
        "p95": float(np.nanpercentile(vv, 95)),
        "median": float(np.nanmedian(vv)),
        "iqr": float(q75 - q25),
        "valid_rate": float(valid_n / max(1, int(denom_n))),
    })
    if valid_n >= 2 and float(np.nanmax(tt) - np.nanmin(tt)) > 1e-9:
        try:
            tt0 = tt - tt[0]
            out["slope"] = float(np.polyfit(tt0, vv, 1)[0])
        except Exception:
            out["slope"] = np.nan
    return out


def binary_duration_stats(values: np.ndarray, times_abs: np.ndarray, interval_start: float, interval_end: float) -> Dict[str, float]:
    out = {k: np.nan for k in AU_C_STATS}
    v = np.asarray(values, dtype=float)
    t = np.asarray(times_abs, dtype=float)
    ok = np.isfinite(v) & np.isfinite(t)
    if int(ok.sum()) == 0:
        return out
    b = (v[ok] >= 0.5).astype(np.int8)
    tt = t[ok]
    order = np.argsort(tt)
    tt = tt[order]
    b = b[order]
    duration = max(float(interval_end - interval_start), 1e-9)

    if len(tt) >= 2:
        dt = np.diff(tt)
        valid_dt = dt[np.isfinite(dt) & (dt > 0) & (dt < 2.0)]
        med_dt = float(np.median(valid_dt)) if len(valid_dt) else duration / max(1, len(tt))
        frame_dur = np.r_[dt, med_dt]
        frame_dur = np.clip(frame_dur, 0.0, max(0.0, duration))
    else:
        frame_dur = np.asarray([duration], dtype=float)

    on_dur = float(np.sum(frame_dur[b == 1]))
    out["presence_rate"] = float(on_dur / duration)

    # Longest continuous ON duration.
    best = 0.0
    cur = 0.0
    for bi, di in zip(b, frame_dur):
        if bi == 1:
            cur += float(di)
            best = max(best, cur)
        else:
            cur = 0.0
    out["longest_on_ratio"] = float(best / duration)
    transitions = int(np.sum(np.diff(b) != 0)) if len(b) >= 2 else 0
    out["transition_rate"] = float(transitions / duration)
    return out


def extract_cv_features_for_segment(
    cv_df: pd.DataFrame,
    cv_t: np.ndarray,
    start: float,
    end: float,
    fps_est: float,
    min_frames: int,
    min_frame_ratio: float,
    use_success_filter: bool,
    min_success_frames: int,
    confidence_min: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    left = int(np.searchsorted(cv_t, start, side="left"))
    right = int(np.searchsorted(cv_t, end, side="right"))
    seg_all = cv_df.iloc[left:right].copy()
    times_all = cv_t[left:right]
    duration = max(0.0, float(end - start))
    expected = duration * fps_est if math.isfinite(fps_est) and fps_est > 0 else np.nan
    frame_ratio = len(seg_all) / expected if math.isfinite(expected) and expected > 0 else np.nan

    q: Dict[str, Any] = {
        "cv_status": "ok",
        "cv_n_frames_all": int(len(seg_all)),
        "cv_expected_frames": float(expected) if math.isfinite(expected) else np.nan,
        "cv_frame_ratio": float(frame_ratio) if math.isfinite(frame_ratio) else np.nan,
        "cv_first_frame_offset_sec": float(times_all[0] - start) if len(times_all) else np.nan,
        "cv_last_frame_offset_sec": float(end - times_all[-1]) if len(times_all) else np.nan,
        "cv_max_frame_gap_sec": float(np.nanmax(np.diff(times_all))) if len(times_all) >= 2 else np.nan,
    }
    if len(seg_all) < min_frames:
        q["cv_status"] = "too_few_frames"
        return {}, q
    if math.isfinite(frame_ratio) and frame_ratio < min_frame_ratio:
        q["cv_status"] = "low_frame_ratio"

    if "confidence" in seg_all.columns:
        conf = pd.to_numeric(seg_all["confidence"], errors="coerce").to_numpy(dtype=float)
        q["cv_confidence_median"] = float(np.nanmedian(conf)) if np.isfinite(conf).any() else np.nan
        q["cv_confidence_mean"] = float(np.nanmean(conf)) if np.isfinite(conf).any() else np.nan
    else:
        conf = np.full(len(seg_all), np.nan)
        q["cv_confidence_median"] = np.nan
        q["cv_confidence_mean"] = np.nan

    if "success" in seg_all.columns:
        success = pd.to_numeric(seg_all["success"], errors="coerce").fillna(0).to_numpy(dtype=float) >= 0.5
        q["cv_success_rate"] = float(np.mean(success)) if len(success) else np.nan
    else:
        success = np.ones(len(seg_all), dtype=bool)
        q["cv_success_rate"] = np.nan

    conf_mask = np.ones(len(seg_all), dtype=bool)
    if confidence_min > 0 and "confidence" in seg_all.columns:
        conf_mask = np.isfinite(conf) & (conf >= confidence_min)

    use_mask = np.ones(len(seg_all), dtype=bool)
    used_mode = "all_rows"
    if use_success_filter:
        candidate = success & conf_mask
        if int(candidate.sum()) >= min_success_frames:
            use_mask = candidate
            used_mode = "success_confidence_rows"
        else:
            used_mode = "fallback_all_rows"
            if q["cv_status"] == "ok":
                q["cv_status"] = "fallback_low_success"

    seg = seg_all.loc[use_mask].copy()
    times = times_all[use_mask]
    q["cv_n_frames_used"] = int(len(seg))
    q["cv_rows_used_mode"] = used_mode
    if len(seg) < min_frames:
        q["cv_status"] = "too_few_frames_after_filter"
        return {}, q

    feats: Dict[str, float] = {}
    denom_n = int(len(seg))
    times_rel = np.asarray(times, dtype=float) - float(start)
    for au in CORE_CV_AU:
        c = au + "_r"
        if c in seg.columns:
            vals = pd.to_numeric(seg[c], errors="coerce").to_numpy(dtype=float)
            stats = continuous_au_r_stats(vals, times_rel, denom_n)
        else:
            stats = {k: np.nan for k in AU_R_STATS}
        for stat in AU_R_STATS:
            feats[f"cv_{c}_{stat}"] = stats.get(stat, np.nan)

        c = au + "_c"
        if c in seg.columns:
            vals = pd.to_numeric(seg[c], errors="coerce").to_numpy(dtype=float)
            stats2 = binary_duration_stats(vals, times, start, end)
        else:
            stats2 = {k: np.nan for k in AU_C_STATS}
        for stat in AU_C_STATS:
            feats[f"cv_{c}_{stat}"] = stats2.get(stat, np.nan)

    q["cv_feature_duration"] = float(duration)
    q["cv_feature_rows"] = int(len(seg))
    return feats, q

# =============================================================================
# EEG feature extraction
# =============================================================================

def canonical_ch(name: str) -> str:
    # Make common EDF channel names comparable to CORE18 names.
    s = str(name).strip()
    s = re.sub(r"^EEG\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"-REF$|-LE$|-A1$|-A2$", "", s, flags=re.IGNORECASE)
    return s.strip()


def channel_lookup(ch_names: Sequence[str]) -> Dict[str, int]:
    out = {}
    for i, ch in enumerate(ch_names):
        out[canonical_ch(ch).lower()] = i
        out[str(ch).strip().lower()] = i
    return out


def robust_stats_over_time(values: np.ndarray, times: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    t = np.asarray(times, dtype=float)
    ok = np.isfinite(v) & np.isfinite(t)
    out = {k: np.nan for k in EEG_ROBUST_STATS}
    if int(ok.sum()) == 0:
        return out
    vv = v[ok]
    tt = t[ok]
    q25 = float(np.nanpercentile(vv, 25))
    q75 = float(np.nanpercentile(vv, 75))
    out["median"] = float(np.nanmedian(vv))
    out["iqr"] = float(q75 - q25)
    out["p95"] = float(np.nanpercentile(vv, 95))
    # 10% trimmed mean, implemented without scipy.stats dependency.
    if len(vv) >= 5:
        lo = int(math.floor(0.10 * len(vv)))
        hi = int(math.ceil(0.90 * len(vv)))
        sv = np.sort(vv)
        arr = sv[lo:hi] if hi > lo else sv
        out["trimmed_mean"] = float(np.nanmean(arr))
    else:
        out["trimmed_mean"] = float(np.nanmean(vv))
    out["last_minus_first"] = float(vv[-1] - vv[0]) if len(vv) >= 2 else np.nan
    if len(vv) >= 2 and float(np.nanmax(tt) - np.nanmin(tt)) > 1e-9:
        try:
            tt0 = tt - tt[0]
            out["slope"] = float(np.polyfit(tt0, vv, 1)[0])
        except Exception:
            out["slope"] = np.nan
    return out


def make_windows(n_samples: int, sfreq: float, window_sec: float, step_sec: float, min_window_sec: float) -> List[Tuple[int, int, float]]:
    if n_samples <= 0 or sfreq <= 0:
        return []
    win = int(round(window_sec * sfreq))
    step = int(round(step_sec * sfreq))
    min_win = int(round(min_window_sec * sfreq))
    win = max(min_win, min(win, n_samples))
    step = max(1, step)
    if n_samples < min_win:
        return []
    starts = list(range(0, max(1, n_samples - win + 1), step))
    if not starts:
        starts = [0]
    # Ensure tail coverage if last window misses > half step.
    tail_start = max(0, n_samples - win)
    if tail_start not in starts and (tail_start - starts[-1]) > max(1, step // 2):
        starts.append(tail_start)
    return [(s, min(n_samples, s + win), (s + min(n_samples, s + win)) / 2.0 / sfreq) for s in starts]


def compute_window_band_features(
    data_uv: np.ndarray,
    sfreq: float,
    bands: Dict[str, Tuple[float, float]],
    window_sec: float,
    step_sec: float,
    min_window_sec: float,
    eeg_clip_uv: float,
) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
    from scipy.signal import detrend, welch

    x = np.asarray(data_uv, dtype=float)
    q: Dict[str, Any] = {
        "eeg_status": "ok",
        "eeg_n_channels": int(x.shape[0]) if x.ndim == 2 else 0,
        "eeg_n_samples": int(x.shape[1]) if x.ndim == 2 else 0,
        "eeg_duration_sec": float(x.shape[1] / sfreq) if x.ndim == 2 and sfreq > 0 else np.nan,
    }
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < int(max(16, min_window_sec * sfreq)):
        q["eeg_status"] = "segment_too_short"
        return None, q
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if eeg_clip_uv > 0:
        x = np.clip(x, -float(eeg_clip_uv), float(eeg_clip_uv))
    q.update({
        "eeg_max_abs_uv": float(np.nanmax(np.abs(x))),
        "eeg_p99_abs_uv": float(np.nanpercentile(np.abs(x), 99)),
        "eeg_std_uv": float(np.nanstd(x)),
    })

    windows = make_windows(x.shape[1], sfreq, window_sec, step_sec, min_window_sec)
    if not windows:
        q["eeg_status"] = "no_valid_windows"
        return None, q

    n_ch = x.shape[0]
    n_bands = len(bands)
    bp_all = np.full((len(windows), n_ch, n_bands), np.nan, dtype=float)
    centers = np.asarray([c for _, _, c in windows], dtype=float)
    band_names = list(bands.keys())

    for wi, (s, e, _) in enumerate(windows):
        seg = x[:, s:e]
        if seg.shape[1] < int(max(16, min_window_sec * sfreq)):
            continue
        seg = seg - np.nanmean(seg, axis=1, keepdims=True)
        try:
            seg = detrend(seg, axis=1, type="linear")
        except Exception:
            pass
        nperseg = min(seg.shape[1], int(max(32, min(1.0 * sfreq, seg.shape[1]))))
        freqs, psd = welch(seg, fs=sfreq, nperseg=nperseg, axis=1)
        for bi, band in enumerate(band_names):
            lo, hi = bands[band]
            mask = (freqs >= lo) & (freqs < hi)
            if not mask.any():
                continue
            try:
                bp = np.trapezoid(psd[:, mask], freqs[mask], axis=1)
            except AttributeError:
                bp = np.trapz(psd[:, mask], freqs[mask], axis=1)
            bp_all[wi, :, bi] = bp

    total = np.nansum(bp_all, axis=2, keepdims=True) + EPS
    logbp = np.log(bp_all + EPS)
    de = 0.5 * np.log(2.0 * np.pi * np.e * (bp_all + EPS))
    relbp = bp_all / total

    q["eeg_n_windows"] = int(len(windows))
    q["eeg_window_sec"] = float(window_sec)
    q["eeg_step_sec"] = float(step_sec)
    return {
        "bp": bp_all,
        "logbp": logbp,
        "de": de,
        "relbp": relbp,
        "centers": centers,
        "band_names": np.asarray(band_names, dtype=object),
    }, q


def eeg_features_from_band_arrays(ch_names: Sequence[str], arrays: Dict[str, np.ndarray], prefix: str = "eeg") -> Dict[str, float]:
    """Create cleaned CORE18 EEG features.

    Clean first-3s design:
      - ONLY Keep DE channel/region features.
      - Robust temporal stats include median/trimmed_mean/iqr/slope/last_minus_first/p95.
      - Compute DASM (Differential Asymmetry) from DE for symmetric pairs.
    """
    band_names = [str(x) for x in arrays["band_names"].tolist()]
    centers = arrays["centers"].astype(float)
    lookup = channel_lookup(ch_names)
    feats: Dict[str, float] = {}

    # Per-channel band robust stats: ONLY DE
    for measure in ["de"]:
        arr = arrays[measure]  # windows, channels, bands
        for ch in CORE18:
            ci = lookup.get(ch.lower())
            if ci is None:
                for band in band_names:
                    for stat in EEG_ROBUST_STATS:
                        feats[f"{prefix}_{measure}_{ch}_{band}_{stat}"] = np.nan
                continue
            for bi, band in enumerate(band_names):
                stats = robust_stats_over_time(arr[:, ci, bi], centers)
                for stat in EEG_ROBUST_STATS:
                    feats[f"{prefix}_{measure}_{ch}_{band}_{stat}"] = stats.get(stat, np.nan)

    # DASM (Differential Asymmetry): DE(Left) - DE(Right)
    for measure in ["de"]:
        arr = arrays[measure]
        for left_ch, right_ch in ASYM_PAIRS:
            li = lookup.get(left_ch.lower())
            ri = lookup.get(right_ch.lower())
            for bi, band in enumerate(band_names):
                if li is not None and ri is not None:
                    dasm_vals = arr[:, li, bi] - arr[:, ri, bi]
                else:
                    dasm_vals = np.full(len(centers), np.nan)
                stats = robust_stats_over_time(dasm_vals, centers)
                for stat in EEG_ROBUST_STATS:
                    feats[f"{prefix}_dasm_{measure}_{left_ch}_{right_ch}_{band}_{stat}"] = stats.get(stat, np.nan)

    # Region averages: ONLY DE
    for measure in ["de"]:
        arr = arrays[measure]
        for region, channels in REGION_GROUPS.items():
            idx = [lookup[c.lower()] for c in channels if lookup.get(c.lower()) is not None]
            for bi, band in enumerate(band_names):
                if idx:
                    vals = np.nanmean(arr[:, idx, bi], axis=1)
                else:
                    vals = np.full(len(centers), np.nan)
                stats = robust_stats_over_time(vals, centers)
                for stat in EEG_ROBUST_STATS:
                    feats[f"{prefix}_region_{region}_{measure}_{band}_{stat}"] = stats.get(stat, np.nan)
            feats[f"{prefix}_region_{region}_n_channels"] = float(len(idx))
    return feats




def eeg_features_from_single_window_arrays(ch_names: Sequence[str], arrays: Dict[str, np.ndarray], prefix: str = "eeg") -> Dict[str, float]:
    """Create one feature vector for one generated segment row.

    This intentionally avoids the old trial-level temporal aggregation over the 5 internal
    windows. If a segment contains more than one PSD window because of config changes,
    it averages them inside that segment only.
    """
    band_names = [str(x) for x in arrays["band_names"].tolist()]
    lookup = channel_lookup(ch_names)
    feats: Dict[str, float] = {}
    arr = arrays["de"]  # windows, channels, bands

    # Per-channel DE value.
    for ch in CORE18:
        ci = lookup.get(ch.lower())
        for bi, band in enumerate(band_names):
            if ci is None:
                feats[f"{prefix}_de_{ch}_{band}"] = np.nan
            else:
                feats[f"{prefix}_de_{ch}_{band}"] = float(np.nanmean(arr[:, ci, bi]))

    # DASM: DE(left) - DE(right).
    for left_ch, right_ch in ASYM_PAIRS:
        li = lookup.get(left_ch.lower())
        ri = lookup.get(right_ch.lower())
        for bi, band in enumerate(band_names):
            if li is not None and ri is not None:
                vals = arr[:, li, bi] - arr[:, ri, bi]
                feats[f"{prefix}_dasm_de_{left_ch}_{right_ch}_{band}"] = float(np.nanmean(vals))
            else:
                feats[f"{prefix}_dasm_de_{left_ch}_{right_ch}_{band}"] = np.nan

    # Region mean DE value.
    for region, channels in REGION_GROUPS.items():
        idx = [lookup[c.lower()] for c in channels if lookup.get(c.lower()) is not None]
        for bi, band in enumerate(band_names):
            if idx:
                vals = np.nanmean(arr[:, idx, bi], axis=1)
                feats[f"{prefix}_region_{region}_de_{band}"] = float(np.nanmean(vals))
            else:
                feats[f"{prefix}_region_{region}_de_{band}"] = np.nan
        feats[f"{prefix}_region_{region}_n_channels"] = float(len(idx))
    return feats

def subtract_feature_dicts(video: Dict[str, float], base: Dict[str, float], prefix: str = "eeg_delta") -> Dict[str, float]:
    out: Dict[str, float] = {}
    keys = sorted(set(video.keys()) | set(base.keys()))
    for k in keys:
        v = to_float(video.get(k))
        b = to_float(base.get(k))
        # Remove original prefix to keep names shorter.
        kk = re.sub(r"^eeg_", "", k)
        out[f"{prefix}_{kk}"] = v - b if math.isfinite(v) and math.isfinite(b) else np.nan
    return out


def load_preprocess_eeg_run(
    edf_path: Path,
    crop_tmin_abs: float,
    crop_tmax_abs: float,
    l_freq: float,
    h_freq: float,
    resample_sfreq: float,
    average_reference: bool,
) -> Tuple[Any, float, Dict[str, Any]]:
    import mne

    log: Dict[str, Any] = {
        "edf_path": str(edf_path),
        "crop_tmin_abs": float(crop_tmin_abs),
        "crop_tmax_abs": float(crop_tmax_abs),
    }
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    log["sfreq_original"] = float(raw.info["sfreq"])
    log["n_channels_original"] = int(len(raw.ch_names))
    log["duration_original_sec"] = float(raw.n_times / raw.info["sfreq"])

    # Crop to relevant run-level segment to save memory
    tmax_possible = raw.n_times / raw.info["sfreq"]
    tmin = max(0.0, float(crop_tmin_abs))
    tmax = min(float(crop_tmax_abs), tmax_possible)
    if tmax <= tmin:
        raise ValueError(f"Invalid crop {tmin}..{tmax}; run duration={tmax_possible}")
    raw.crop(tmin=tmin, tmax=tmax, include_tmax=True)
    raw.load_data(verbose="ERROR")

    # 1. Normalize Channel Names to standard 10-20 format
    raw.rename_channels({ch: canonical_ch(ch) for ch in raw.ch_names})

    # 2. Set channel types + pick EEG
    # Only keep channels that match standard 10-20 or known proxies to prevent junk from breaking ICA
    montage = mne.channels.make_standard_montage("standard_1020")
    std_ch = set(montage.ch_names)
    ch_types = {}
    for ch in raw.ch_names:
        if ch in std_ch or ch.lower() in ["fp1", "fp2", "eog"]:
            ch_types[ch] = "eeg"
        else:
            ch_types[ch] = "misc"
    raw.set_channel_types(ch_types, verbose="ERROR")
    raw.pick("eeg")

    # 3. Set montage (standard_1020) - Need this before interpolating
    try:
        raw.set_montage(montage, on_missing="ignore", verbose="ERROR")
    except Exception as e:
        log["montage_error"] = repr(e)

    log["n_channels_before_ica"] = int(len(raw.ch_names))

    # 4. Filter
    raw.filter(l_freq=float(l_freq), h_freq=float(h_freq), fir_design="firwin", verbose="ERROR")
    log["filter"] = f"{l_freq}-{h_freq}Hz"

    # 5. Resample BEFORE bad channel detection and ICA to save memory and speed up computation
    if resample_sfreq > 0 and abs(float(raw.info["sfreq"]) - float(resample_sfreq)) > 1e-6:
        raw.resample(float(resample_sfreq), npad="auto", verbose="ERROR")

    # 4. Detect Bad Channels using Log-Variance + MAD
    data = raw.get_data()
    variances = np.var(data, axis=1)
    variances[variances < EPS] = EPS
    log_var = np.log(variances)
    med_log_var = np.median(log_var)
    mad = np.median(np.abs(log_var - med_log_var))
    if mad < EPS:
        mad = EPS
    
    threshold = 4.0
    bads = []
    for i, ch in enumerate(raw.ch_names):
        # Do not mark Fp1/Fp2 as bad since they are used for EOG proxy in ICA
        if ch.lower() in ["fp1", "fp2"]:
            continue
        if np.abs(log_var[i] - med_log_var) > threshold * mad:
            bads.append(ch)
            
    log["bad_channels"] = ",".join(bads) if bads else "none"
    log["n_bad_channels"] = len(bads)

    # 5. Interpolate bad channels
    if bads:
        raw.info["bads"] = bads
        try:
            raw.interpolate_bads(reset_bads=True, verbose="ERROR")
            log["interpolation"] = "success"
        except Exception as e:
            log["interpolation"] = f"failed:{repr(e)}"
            raw.info["bads"] = []
    else:
        log["interpolation"] = "none"

    # 6. Fit ICA on the whole run
    log["eog_reference_used"] = "none"
    log["ica_excluded_components"] = 0
    try:
        from mne.preprocessing import ICA
        ica = ICA(n_components=15, random_state=42, max_iter='auto')
        ica.fit(raw, verbose="ERROR")
        
        # Use Fp1 and Fp2 as proxy EOG channels
        eog_ch = [ch for ch in raw.ch_names if ch.lower() in ["fp1", "fp2"]]
        eog_indices = []
        if eog_ch:
            eog_inds, eog_scores = ica.find_bads_eog(raw, ch_name=eog_ch, verbose="ERROR")
            eog_indices.extend(eog_inds)
            log["eog_reference_used"] = ",".join(eog_ch)
        
        eog_indices = list(set(eog_indices))
        ica.exclude = eog_indices
        log["ica_excluded_components"] = len(eog_indices)
        ica.apply(raw, verbose="ERROR")
        log["ica_status"] = "success"
    except Exception as e:
        log["ica_status"] = f"failed:{repr(e)}"

    # 7. Finally, pick CORE18
    lookup = channel_lookup(raw.ch_names)
    missing = [ch for ch in CORE18 if lookup.get(ch.lower()) is None]
    if missing:
        raise ValueError(f"Missing CORE18 EEG channels: {missing}")
    picks = [raw.ch_names[lookup[ch.lower()]] for ch in CORE18]
    raw.pick(picks)

    # 8. Average Reference (Applied after picking CORE18)
    if average_reference:
        try:
            raw.set_eeg_reference("average", projection=False, verbose="ERROR")
            log["reference"] = "average"
        except Exception as e:
            log["reference"] = f"failed:{repr(e)}"
    else:
        log["reference"] = "original"

    # 9. Resampling was already moved to step 5 before ICA
    log["sfreq_final"] = float(raw.info["sfreq"])
    log["n_channels_final"] = int(len(raw.ch_names))
    log["duration_loaded_sec"] = float(raw.n_times / raw.info["sfreq"])
    
    return raw, tmin, log


def eeg_segment_uv_from_cropped(raw: Any, crop_tmin_abs: float, start_abs: float, end_abs: float) -> Tuple[np.ndarray, float, int, int]:
    sf = float(raw.info["sfreq"])
    rel_start = max(0.0, float(start_abs) - float(crop_tmin_abs))
    rel_end = max(rel_start, float(end_abs) - float(crop_tmin_abs))
    n_times = int(raw.n_times)
    start = int(max(0, math.floor(rel_start * sf)))
    stop = int(min(n_times, math.ceil(rel_end * sf)))
    if stop <= start:
        return np.empty((0, 0), dtype=float), sf, start, stop
    data = raw.get_data(start=start, stop=stop, verbose="ERROR") * 1e6
    return data, sf, start, stop


def extract_eeg_segment_features(
    data_uv: np.ndarray,
    sfreq: float,
    ch_names: Sequence[str],
    window_sec: float,
    step_sec: float,
    min_window_sec: float,
    eeg_clip_uv: float,
    feature_mode: str = "single_values",
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    arrays, q = compute_window_band_features(
        data_uv=data_uv,
        sfreq=sfreq,
        bands=EEG_BANDS,
        window_sec=window_sec,
        step_sec=step_sec,
        min_window_sec=min_window_sec,
        eeg_clip_uv=eeg_clip_uv,
    )
    if arrays is None:
        return {}, q
    q["eeg_segment_feature_mode"] = str(feature_mode)
    if str(feature_mode).lower() == "single_values":
        feats = eeg_features_from_single_window_arrays(ch_names, arrays, prefix="eeg")
    else:
        # Backward-compatible mode: aggregate temporal stats inside the generated segment.
        feats = eeg_features_from_band_arrays(ch_names, arrays, prefix="eeg")
    return feats, q

# =============================================================================
# Main extraction loop
# =============================================================================

def process_run(
    subject: str,
    run: str,
    trials: pd.DataFrame,
    paths: Dict[str, Path],
    cv_offset_sec: float,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], List[Dict[str, float]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return eeg_video_rows, eeg_delta_rows, cv_rows, quality_rows, eeg_log_rows, cv_log_rows, errors."""
    eeg_video_rows: List[Dict[str, float]] = []
    eeg_delta_rows: List[Dict[str, float]] = []
    cv_rows: List[Dict[str, float]] = []
    quality_rows: List[Dict[str, Any]] = []
    eeg_logs: List[Dict[str, Any]] = []
    cv_logs: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    # -----------------------------
    # Load/preprocess EEG once/run
    # -----------------------------
    raw = None
    crop_tmin_abs = np.nan
    eeg_load_log: Dict[str, Any] = {"subject": subject, "run": run, "eeg_run_status": "not_loaded"}
    try:
        if "edf" not in paths:
            raise FileNotFoundError("Missing EDF path")
        eeg_start_col = "segment_start_abs" if "segment_start_abs" in trials.columns else ("eeg_first3s_start" if "eeg_first3s_start" in trials.columns else "video_interval_start")
        eeg_end_col = "segment_end_abs" if "segment_end_abs" in trials.columns else ("eeg_first3s_end" if "eeg_first3s_end" in trials.columns else "video_interval_end")
        starts = trials[eeg_start_col].astype(float).tolist()
        ends = trials[eeg_end_col].astype(float).tolist()
        if parse_bool(args.extract_second_fix_delta):
            if "baseline_second_fix_start" in trials.columns:
                starts += trials["baseline_second_fix_start"].astype(float).tolist()
            if "baseline_second_fix_end" in trials.columns:
                ends += trials["baseline_second_fix_end"].astype(float).tolist()
        finite_starts = [x for x in starts if math.isfinite(x)]
        finite_ends = [x for x in ends if math.isfinite(x)]
        if not finite_starts or not finite_ends:
            raise ValueError("No finite EEG crop intervals")
        crop_start = max(0.0, min(finite_starts) - float(args.eeg_crop_pad_sec))
        crop_end = max(finite_ends) + float(args.eeg_crop_pad_sec)
        raw, crop_tmin_abs, load_log = load_preprocess_eeg_run(
            edf_path=paths["edf"],
            crop_tmin_abs=crop_start,
            crop_tmax_abs=crop_end,
            l_freq=float(args.eeg_l_freq),
            h_freq=float(args.eeg_h_freq),
            resample_sfreq=float(args.eeg_resample_sfreq),
            average_reference=parse_bool(args.eeg_average_reference),
        )
        eeg_load_log.update(load_log)
        eeg_load_log["eeg_run_status"] = "ok"
    except Exception as e:
        eeg_load_log["eeg_run_status"] = "load_failed"
        eeg_load_log["error"] = repr(e)
        errors.append({"type": "eeg_run_load_failed", "subject": subject, "run": run, "message": repr(e)})
    eeg_logs.append(eeg_load_log)

    # -----------------------------
    # Load CV once/run
    # -----------------------------
    cv_df = None
    cv_t = None
    fps_est = np.nan
    cv_run_log: Dict[str, Any] = {"subject": subject, "run": run, "cv_run_status": "not_loaded", "cv_offset_sec": float(cv_offset_sec)}
    try:
        if "cv_tsv" not in paths:
            raise FileNotFoundError("Missing videostream TSV path")
        cv_df0, read_log = read_cv_selected(paths["cv_tsv"], paths.get("cv_json"))
        t, time_col, fps_est = infer_cv_time(cv_df0, fps_hint=0.0)
        # Apply optional offset so CV time is in same coordinate as event time.
        t = np.asarray(t, dtype=float) + float(cv_offset_sec)
        cv_df, cv_t = sort_cv_by_time(cv_df0, t)
        cv_run_log.update(read_log)
        cv_run_log.update({
            "cv_run_status": "ok",
            "cv_time_col": time_col,
            "cv_fps_est": float(fps_est) if math.isfinite(fps_est) else np.nan,
            "cv_t_min_aligned": float(np.nanmin(cv_t)) if cv_t is not None and len(cv_t) else np.nan,
            "cv_t_max_aligned": float(np.nanmax(cv_t)) if cv_t is not None and len(cv_t) else np.nan,
        })
    except Exception as e:
        cv_run_log["cv_run_status"] = "load_failed"
        cv_run_log["error"] = repr(e)
        errors.append({"type": "cv_run_load_failed", "subject": subject, "run": run, "message": repr(e)})
    cv_logs.append(cv_run_log)

    # -----------------------------
    # Per-trial extraction
    # -----------------------------
    for _, tr in trials.iterrows():
        base_meta = {
            "subject": subject,
            "run": run,
            "trial": int(tr.get("trial")),
            "trial_uid": str(tr.get("trial_uid", f"{subject}_run-{run}_trial-{tr.get('trial')}")),
        }
        qrow: Dict[str, Any] = dict(base_meta)

        # The expanded manifest already defines one generated segment row for BOTH EEG and CV.
        v_start_full = to_float(tr.get("video_interval_start"))
        v_end_full = to_float(tr.get("video_interval_end"))
        v_start = to_float(tr.get("segment_start_abs", tr.get("eeg_first3s_start", v_start_full)))
        v_end = to_float(tr.get("segment_end_abs", tr.get("eeg_first3s_end", v_end_full)))
        sf_start = to_float(tr.get("baseline_second_fix_start"))
        sf_end = to_float(tr.get("baseline_second_fix_end"))

        # EEG video.
        eeg_video_feat: Dict[str, float] = {}
        if raw is not None and math.isfinite(v_start) and math.isfinite(v_end) and v_end > v_start:
            try:
                data_uv, sf, seg_start, seg_stop = eeg_segment_uv_from_cropped(raw, crop_tmin_abs, v_start, v_end)
                eeg_video_feat, q_eeg = extract_eeg_segment_features(
                    data_uv=data_uv,
                    sfreq=sf,
                    ch_names=raw.ch_names,
                    window_sec=float(args.eeg_window_sec),
                    step_sec=float(args.eeg_step_sec),
                    min_window_sec=float(args.eeg_min_window_sec),
                    eeg_clip_uv=float(args.eeg_clip_uv),
                    feature_mode=str(args.eeg_segment_feature_mode),
                )
                qrow.update({f"video_{k}": v for k, v in q_eeg.items()})
                qrow["eeg_video_segment_start_sample"] = int(seg_start)
                qrow["eeg_video_segment_stop_sample"] = int(seg_stop)
            except Exception as e:
                qrow["video_eeg_status"] = "extract_failed"
                errors.append({**base_meta, "type": "eeg_video_extract_failed", "message": repr(e)})
        else:
            qrow["video_eeg_status"] = "run_not_loaded_or_bad_interval"
        eeg_video_rows.append(eeg_video_feat)

        # EEG second_fix_delta.
        eeg_delta_feat: Dict[str, float] = {}
        if parse_bool(args.extract_second_fix_delta):
            if eeg_video_feat and raw is not None and math.isfinite(sf_start) and math.isfinite(sf_end) and sf_end > sf_start:
                try:
                    data_uv_b, sf_b, _, _ = eeg_segment_uv_from_cropped(raw, crop_tmin_abs, sf_start, sf_end)
                    eeg_base_feat, q_base = extract_eeg_segment_features(
                        data_uv=data_uv_b,
                        sfreq=sf_b,
                        ch_names=raw.ch_names,
                        window_sec=float(args.eeg_window_sec),
                        step_sec=float(args.eeg_step_sec),
                        min_window_sec=min(float(args.eeg_min_window_sec), max(0.2, sf_end - sf_start)),
                        eeg_clip_uv=float(args.eeg_clip_uv),
                        feature_mode=str(args.eeg_segment_feature_mode),
                    )
                    qrow.update({f"secondfix_{k}": v for k, v in q_base.items()})
                    eeg_delta_feat = subtract_feature_dicts(eeg_video_feat, eeg_base_feat, prefix="eeg_delta_secondfix")
                    qrow["eeg_delta_status"] = "ok" if eeg_delta_feat else "empty"
                except Exception as e:
                    qrow["eeg_delta_status"] = "extract_failed"
                    errors.append({**base_meta, "type": "eeg_delta_extract_failed", "message": repr(e)})
            else:
                qrow["eeg_delta_status"] = "missing_second_fix_or_video"
        eeg_delta_rows.append(eeg_delta_feat)

        # CV AU-only now uses the SAME 3-second interval as EEG for strict alignment
        cv_feat: Dict[str, float] = {}
        if cv_df is not None and cv_t is not None and math.isfinite(v_start) and math.isfinite(v_end) and v_end > v_start:
            try:
                cv_feat, q_cv = extract_cv_features_for_segment(
                    cv_df=cv_df,
                    cv_t=cv_t,
                    start=v_start,
                    end=v_end,
                    fps_est=float(fps_est),
                    min_frames=int(args.cv_min_frames),
                    min_frame_ratio=float(args.cv_min_frame_ratio),
                    use_success_filter=parse_bool(args.cv_use_success_filter),
                    min_success_frames=int(args.cv_min_success_frames),
                    confidence_min=float(args.cv_confidence_min),
                )
                qrow.update(q_cv)
            except Exception as e:
                qrow["cv_status"] = "extract_failed"
                errors.append({**base_meta, "type": "cv_extract_failed", "message": repr(e)})
        else:
            qrow["cv_status"] = "run_not_loaded_or_bad_interval"
        cv_rows.append(cv_feat)

        quality_rows.append(qrow)

    return eeg_video_rows, eeg_delta_rows, cv_rows, quality_rows, eeg_logs, cv_logs, errors


def write_summary(
    out_path: Path,
    args: argparse.Namespace,
    metadata: pd.DataFrame,
    registry: Dict[str, Any],
    quality: pd.DataFrame,
    eeg_log: pd.DataFrame,
    cv_log: pd.DataFrame,
    error_log: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("=" * 100)
    lines.append("AFFEC CLEAN EDA 01 - CORE18 EEG + AU-only CV FEATURE EXTRACTION SUMMARY")
    lines.append("=" * 100)
    lines.append("")
    lines.append("[CONFIG]")
    for k in [
        "raw_root", "eda00_dir", "out_dir", "eeg_fixed_first_sec", "segment_window_sec",
        "segment_step_sec", "segment_include_tail", "segment_max_windows_per_trial", "eeg_segment_feature_mode",
        "eeg_l_freq", "eeg_h_freq", "eeg_resample_sfreq",
        "eeg_window_sec", "eeg_step_sec", "eeg_min_window_sec", "eeg_average_reference",
        "eeg_clip_uv", "cv_use_success_filter", "cv_confidence_min", "extract_second_fix_delta",
    ]:
        lines.append(f"  - {k}: {getattr(args, k)}")
    lines.append("")
    lines.append("[DATA]")
    lines.append(f"  - n_rows/generated segments: {len(metadata)}")
    if "original_trial_uid" in metadata.columns:
        lines.append(f"  - original trials: {metadata['original_trial_uid'].nunique()}")
    lines.append(f"  - subjects: {metadata['subject'].nunique() if 'subject' in metadata else 'NA'}")
    lines.append(f"  - subject-run pairs: {metadata[['subject','run']].drop_duplicates().shape[0] if {'subject','run'}.issubset(metadata.columns) else 'NA'}")
    if "video_interval_duration" in metadata.columns:
        lines.append(f"  - video full duration median: {pd.to_numeric(metadata['video_interval_duration'], errors='coerce').median():.3f}s")
    if "eeg_branch_interval" in metadata.columns and len(metadata):
        lines.append(f"  - EEG/CV branch interval: {metadata['eeg_branch_interval'].iloc[0]}; kept generated rows: {len(metadata)}")
    if "n_segments_for_original_trial" in metadata.columns and len(metadata):
        w = pd.to_numeric(metadata.drop_duplicates("original_trial_uid")["n_segments_for_original_trial"], errors="coerce")
        lines.append(f"  - segments per original trial: median={w.median():.1f}, min={w.min():.0f}, max={w.max():.0f}")
    if "keep_clean_second_fix_delta" in metadata.columns:
        lines.append(f"  - second_fix_delta available rows: {int(pd.to_numeric(metadata['keep_clean_second_fix_delta'], errors='coerce').fillna(0).sum())}/{len(metadata)}")
    lines.append("")
    lines.append("[FEATURE MATRICES]")
    for name, item in registry.get("feature_sets", {}).items():
        s = item.get("summary", {})
        lines.append(f"  - {name}: shape={tuple(s.get('shape', []))}, nan_rate={s.get('nan_rate'):.6f}, all_nan_rows={s.get('all_nan_rows')}, finite_min={s.get('finite_min')}, finite_max={s.get('finite_max')}")
    lines.append("")
    lines.append("[EEG LOG]")
    if len(eeg_log):
        if "eeg_run_status" in eeg_log.columns:
            lines.append("  - run status:")
            lines.append(str(eeg_log["eeg_run_status"].value_counts(dropna=False)))
        if "sfreq_original" in eeg_log.columns:
            lines.append("  - original sfreq values:")
            lines.append(str(eeg_log["sfreq_original"].value_counts(dropna=False).sort_index()))
        if "sfreq_final" in eeg_log.columns:
            lines.append("  - final sfreq values:")
            lines.append(str(eeg_log["sfreq_final"].value_counts(dropna=False).sort_index()))
    lines.append("")
    lines.append("[CV LOG]")
    if len(cv_log):
        if "cv_run_status" in cv_log.columns:
            lines.append("  - run status:")
            lines.append(str(cv_log["cv_run_status"].value_counts(dropna=False)))
        if "cv_fps_est" in cv_log.columns:
            lines.append("  - estimated FPS summary:")
            lines.append(str(pd.to_numeric(cv_log["cv_fps_est"], errors="coerce").describe()))
        if "read_mode" in cv_log.columns:
            lines.append("  - read modes:")
            lines.append(str(cv_log["read_mode"].value_counts(dropna=False)))
    lines.append("")
    lines.append("[QUALITY]")
    if len(quality):
        if "cv_status" in quality.columns:
            lines.append("  - cv_status:")
            lines.append(str(quality["cv_status"].value_counts(dropna=False)))
        if "video_eeg_status" in quality.columns:
            lines.append("  - video_eeg_status:")
            lines.append(str(quality["video_eeg_status"].value_counts(dropna=False)))
        if "eeg_delta_status" in quality.columns:
            lines.append("  - eeg_delta_status:")
            lines.append(str(quality["eeg_delta_status"].value_counts(dropna=False)))
    lines.append("")
    lines.append("[ERRORS]")
    lines.append(f"  - n_errors: {len(error_log)}")
    if len(error_log) and "type" in error_log.columns:
        lines.append(str(error_log["type"].value_counts(dropna=False).head(20)))
    lines.append("")
    lines.append("[NEXT]")
    lines.append("  - Kiểm tra summary/quality trước.")
    lines.append("  - Nếu feature sạch: chạy EDA03/Train trên feature keys trong feature_registry_clean.json.")
    lines.append("  - second_fix_delta chỉ là ablation; không thay main nếu chưa kiểm chứng.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", default=r"data\raw")
    ap.add_argument("--eda00_dir", default=r"results\affec_clean\00_manifest_feature_audit_video")
    ap.add_argument("--out_dir", default=r"results\demo\01D_core18_first3s_sw_segments_ica_clip")
    ap.add_argument("--manifest_csv", default="", help="Override final_trial_index_clean.csv")
    ap.add_argument("--run_file_coverage_csv", default="", help="Override run_file_coverage.csv")
    ap.add_argument("--cv_offset_csv", default="", help="Optional run-level CV offset CSV with offset_sec column")
    ap.add_argument("--max_runs", type=int, default=0, help="Debug: process only first N subject-run groups")
    ap.add_argument("--max_trials", type=int, default=0, help="Debug: process only first N trials after filtering")

    # EEG settings.
    ap.add_argument("--eeg_l_freq", type=float, default=4.0)
    ap.add_argument("--eeg_h_freq", type=float, default=45.0)
    ap.add_argument("--eeg_resample_sfreq", type=float, default=256.0)
    ap.add_argument("--eeg_average_reference", default="1")
    ap.add_argument("--eeg_crop_pad_sec", type=float, default=0.5)
    ap.add_argument("--eeg_window_sec", type=float, default=1.0)
    ap.add_argument("--eeg_step_sec", type=float, default=1.0, help="Inside each generated segment. Leave 1.0 for one PSD window per 1s sample.")
    ap.add_argument("--eeg_min_window_sec", type=float, default=0.5)
    ap.add_argument("--eeg_clip_uv", type=float, default=100.0, help="Fixed pre-PSD clipping in microvolts; 0 disables")
    ap.add_argument("--eeg_fixed_first_sec", "--fixed_first_sec", dest="eeg_fixed_first_sec", type=float, default=3.0, help="Use the first N seconds as context before expanding into segment rows")
    ap.add_argument("--segment_window_sec", type=float, default=1.0, help="Generated sample/window length inside first-N seconds")
    ap.add_argument("--segment_step_sec", type=float, default=0.5, help="Generated sample/window step. 3s + 1s window + 0.5s step => 5 samples/trial")
    ap.add_argument("--segment_include_tail", default="0", help="Usually 0 for exact 5 windows from 3s with 1s/0.5s")
    ap.add_argument("--segment_max_windows_per_trial", type=int, default=0, help="0 = no cap; use for debugging only")
    ap.add_argument("--eeg_segment_feature_mode", choices=["single_values", "robust_stats"], default="single_values", help="single_values avoids aggregating the 5 windows back into one trial vector")
    ap.add_argument("--extract_second_fix_delta", default="0")

    # CV settings.
    ap.add_argument("--cv_min_frames", type=int, default=5)
    ap.add_argument("--cv_min_frame_ratio", type=float, default=0.20)
    ap.add_argument("--cv_use_success_filter", default="1")
    ap.add_argument("--cv_min_success_frames", type=int, default=5)
    ap.add_argument("--cv_confidence_min", type=float, default=0.0)

    args = ap.parse_args()
    raw_root = Path(args.raw_root)
    eda00_dir = Path(args.eda00_dir)
    out_dir = mkdir(Path(args.out_dir))
    mkdir(out_dir / "eeg")
    mkdir(out_dir / "cv")

    manifest = load_manifest(eda00_dir, args.manifest_csv)

    # Expand each original trial into multiple generated sample rows inside first-N seconds.
    # Default: first 3s -> five 1s windows with 0.5s hop.
    manifest, dropped_first3 = expand_manifest_first3s_to_segment_rows(
        manifest=manifest,
        first_sec=float(args.eeg_fixed_first_sec),
        segment_window_sec=float(args.segment_window_sec),
        segment_step_sec=float(args.segment_step_sec),
        include_tail=parse_bool(args.segment_include_tail),
        max_segments_per_trial=int(args.segment_max_windows_per_trial),
    )

    if args.max_trials and int(args.max_trials) > 0:
        manifest = manifest.head(int(args.max_trials)).copy()
    run_paths = load_run_paths(raw_root, eda00_dir, args.run_file_coverage_csv)
    cv_offsets = load_cv_offsets(args.cv_offset_csv)

    # Save metadata early.
    metadata = manifest.copy().reset_index(drop=True)
    metadata.insert(0, "row_id", np.arange(len(metadata), dtype=int))
    try:
        dropped_first3.to_csv(out_dir / "dropped_trials_shorter_than_first3s_for_segments.csv", index=False)
    except Exception:
        pass

    eeg_video_all: List[Dict[str, float]] = []
    eeg_delta_all: List[Dict[str, float]] = []
    cv_all: List[Dict[str, float]] = []
    quality_all: List[Dict[str, Any]] = []
    eeg_logs_all: List[Dict[str, Any]] = []
    cv_logs_all: List[Dict[str, Any]] = []
    errors_all: List[Dict[str, Any]] = []

    groups = list(metadata.groupby(["subject", "run"], sort=True))
    if args.max_runs and int(args.max_runs) > 0:
        keep_keys = set(k for k, _ in groups[: int(args.max_runs)])
        metadata = metadata[metadata.apply(lambda r: (r["subject"], r["run"]) in keep_keys, axis=1)].reset_index(drop=True)
        groups = list(metadata.groupby(["subject", "run"], sort=True))

    print(f"Processing {len(metadata)} generated segment rows from {len(groups)} subject-run groups")
    for i, ((subject, run), trials) in enumerate(groups, 1):
        print(f"[{i}/{len(groups)}] {subject} run {run} | n_segment_rows={len(trials)}")
        paths = run_paths.get((subject, run), {})
        cv_offset = cv_offsets.get((subject, run), 0.0)
        ev, ed, cv, q, elog, clog, errs = process_run(
            subject=str(subject),
            run=str(run),
            trials=trials,
            paths=paths,
            cv_offset_sec=float(cv_offset),
            args=args,
        )
        eeg_video_all.extend(ev)
        eeg_delta_all.extend(ed)
        cv_all.extend(cv)
        quality_all.extend(q)
        eeg_logs_all.extend(elog)
        cv_logs_all.extend(clog)
        errors_all.extend(errs)

    # Save matrices and tables.
    # --- clean_strict Filtering ---
    # We only keep trials where BOTH EEG and CV are valid (not all-NaN, properly extracted).
    # This ensures the matrices are perfectly aligned for FUSION and contain no missing modality data.
    clean_indices = []
    for i, q in enumerate(quality_all):
        if q.get("video_eeg_status") == "ok" and q.get("cv_status") == "ok":
            clean_indices.append(i)
    
    print(f"clean_strict filtering: kept {len(clean_indices)} / {len(metadata)} generated segment rows.")
    
    metadata = metadata.iloc[clean_indices].reset_index(drop=True)
    if "row_id" in metadata.columns:
        metadata["row_id"] = np.arange(len(metadata), dtype=int)
    
    # Recalculate actual segments left after strict clean
    if "original_trial_uid" in metadata.columns:
        metadata["n_segments_for_original_trial"] = metadata.groupby("original_trial_uid")["original_trial_uid"].transform("count").astype(int)

    eeg_video_all = [eeg_video_all[i] for i in clean_indices]
    cv_all = [cv_all[i] for i in clean_indices]
    if parse_bool(args.extract_second_fix_delta):
        eeg_delta_all = [eeg_delta_all[i] for i in clean_indices]
    quality_all = [quality_all[i] for i in clean_indices]
    # ------------------------------

    metadata.to_csv(out_dir / "metadata_trials_clean.csv", index=False)
    quality_df = pd.DataFrame(quality_all)
    eeg_log_df = pd.DataFrame(eeg_logs_all)
    cv_log_df = pd.DataFrame(cv_logs_all)
    error_df = pd.DataFrame(errors_all)
    quality_df.to_csv(out_dir / "trial_feature_quality_clean.csv", index=False)
    eeg_log_df.to_csv(out_dir / "eeg_processing_log.csv", index=False)
    cv_log_df.to_csv(out_dir / "cv_processing_log.csv", index=False)
    error_df.to_csv(out_dir / "error_log.csv", index=False)

    segment_stem = safe_name(f"first{float(args.eeg_fixed_first_sec):g}s_seg{float(args.segment_window_sec):g}s_step{float(args.segment_step_sec):g}s")
    interval_label = f"first_{float(args.eeg_fixed_first_sec):g}s_expanded_to_{float(args.segment_window_sec):g}s_step_{float(args.segment_step_sec):g}s_segments"
    eeg_key = f"eeg_core18_{segment_stem}_clean"
    eeg_delta_key = f"eeg_core18_{segment_stem}_second_fix_delta"
    cv_key = f"cv_au_only_{segment_stem}"

    registry: Dict[str, Any] = {
        "version": "affec_clean_eda01D_core18_first3s_expanded_segment_samples_ica_clip_de_auonly",
        "n_rows": int(len(metadata)),
        "n_original_trials": int(metadata["original_trial_uid"].nunique()) if "original_trial_uid" in metadata.columns and len(metadata) else int(len(metadata)),
        "metadata_csv": "metadata_trials_clean.csv",
        "quality_csv": "trial_feature_quality_clean.csv",
        "feature_sets": {},
        "config": vars(args),
        "split_warning": "Generated segment rows from the same original_trial_uid are highly correlated. Split/group by subject or original_trial_uid; never random-split rows directly.",
    }

    x_path, n_path, X, names = save_feature_matrix(out_dir, "eeg", eeg_key, eeg_video_all)
    registry["feature_sets"][eeg_key] = {
        "modality": "eeg",
        "path": str(x_path.relative_to(out_dir)),
        "feature_names": str(n_path.relative_to(out_dir)),
        "n_features": int(len(names)),
        "interval": interval_label,
        "feature_design": "ICA + Clip 100uV + CORE18 DE; each row is one generated segment, not a trial-level aggregation",
        "summary": finite_summary(X),
    }

    if parse_bool(args.extract_second_fix_delta):
        x_path, n_path, X, names = save_feature_matrix(out_dir, "eeg", eeg_delta_key, eeg_delta_all)
        registry["feature_sets"][eeg_delta_key] = {
            "modality": "eeg",
            "path": str(x_path.relative_to(out_dir)),
            "feature_names": str(n_path.relative_to(out_dir)),
            "n_features": int(len(names)),
            "interval": interval_label + "_minus_second_fix",
            "summary": finite_summary(X),
        }

    x_path, n_path, X, names = save_feature_matrix(out_dir, "cv", cv_key, cv_all)
    registry["feature_sets"][cv_key] = {
        "modality": "cv",
        "path": str(x_path.relative_to(out_dir)),
        "feature_names": str(n_path.relative_to(out_dir)),
        "n_features": int(len(names)),
        "interval": interval_label,
        "feature_design": "AU-only on the same generated segment interval as EEG",
        "summary": finite_summary(X),
    }

    write_json(registry, out_dir / "feature_registry_clean.json")
    write_summary(
        out_path=out_dir / "eda_clean_01_summary.txt",
        args=args,
        metadata=metadata,
        registry=registry,
        quality=quality_df,
        eeg_log=eeg_log_df,
        cv_log=cv_log_df,
        error_log=error_df,
    )
    print("Done.")
    print(f"Summary: {out_dir / 'eda_clean_01_summary.txt'}")
    print(f"Registry: {out_dir / 'feature_registry_clean.json'}")


if __name__ == "__main__":
    main()
