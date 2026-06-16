import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default=r"results\demo\01C_core18_first3s_ica_clip")
    args = ap.parse_args()
    
    feature_dir = Path(args.feature_dir)
    meta = pd.read_csv(feature_dir / "metadata_trials_clean.csv")
    
    eeg_x = np.load(feature_dir / "eeg" / "X_eeg_core18_first3s_clean.npy")
    eeg_names = read_json(feature_dir / "eeg" / "feature_names_eeg_core18_first3s_clean.json")
    
    print("="*60)
    print("1. TÌM ĐẶC TRƯNG TỐT NHẤT (FEATURE IMPORTANCE)")
    print("="*60)
    
    for target_name, label in [("f_emotion_a_3class_v2", "AROUSAL"), ("f_emotion_v_3class_v2", "VALENCE")]:
        y_raw = pd.to_numeric(meta[target_name], errors="coerce")
        keep = np.isfinite(eeg_x).any(axis=1) & y_raw.notna().to_numpy()
        
        X = eeg_x[keep]
        y = y_raw.loc[keep].astype(int).to_numpy()
        
        # Impute
        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X)
        
        selector = SelectKBest(score_func=f_classif, k="all")
        selector.fit(X_imp, y)
        
        scores = selector.scores_
        scores[np.isnan(scores)] = 0
        top_idx = np.argsort(scores)[::-1][:15]
        
        print(f"\n[ TOP 15 EEG FEATURES CHO {label} ]")
        for rank, idx in enumerate(top_idx, start=1):
            print(f"  #{rank:02d} | F-score: {scores[idx]:.2f} | Feature: {eeg_names[idx]}")

    print("\n" + "="*60)
    print("2. ĐÁNH GIÁ DẤU VÂN TAY NGƯỜI DÙNG (SUBJECT FINGERPRINT)")
    print("="*60)
    
    keep = np.isfinite(eeg_x).any(axis=1)
    X = eeg_x[keep]
    y_sub = meta["subject"].astype("category").cat.codes.to_numpy()[keep]
    
    # StratifiedKFold để đo xem dễ đoán user cỡ nào
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", C=0.1, max_iter=1000))
    ])
    
    print(f"Đang huấn luyện mô hình dự đoán Subject ID trên {X.shape[0]} trials (tổng {len(np.unique(y_sub))} subjects)...")
    cv_scores = cross_val_score(pipe, X, y_sub, cv=skf, scoring="accuracy", n_jobs=-1)
    acc = np.mean(cv_scores)
    
    print(f"-> Khả năng dự đoán đúng danh tính người dùng (Subject Accuracy): {acc*100:.2f}%")
    print("Nếu chỉ đoán bừa, Accuracy khoảng {:.2f}%".format(100.0/len(np.unique(y_sub))))
    if acc > 0.8:
        print("=> CẢNH BÁO: Rò rỉ Dấu vân tay người dùng rất nặng! Bắt buộc phải dùng StratifiedGroupKFold khi train Cảm xúc.")
    else:
        print("=> Tạm ổn định.")

if __name__ == "__main__":
    main()
