import sys
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy.signal import hilbert

def compute_features(raw_data, sfreq, l_freq, h_freq):
    # Filter data
    filtered = mne.filter.filter_data(raw_data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False)
    # Envelope via Hilbert
    envelope = np.abs(hilbert(filtered, axis=-1))
    
    # 1. PSD (approximated by squared envelope)
    psd = envelope ** 2
    
    # 2. Log(PSD)
    log_psd = np.log10(psd + 1e-15)
    
    # 3. DE (Differential Entropy)
    de = 0.5 * np.log(2 * np.pi * np.e * (psd + 1e-15))
    
    return psd, log_psd, de

def analyze_trial(sub, run, trial_num, channels):
    sub = sub.replace("sub-", "")
    base_dir = Path("data/raw")
    sub_dir = base_dir / f"sub-{sub}"
    
    events_path = sub_dir / f"sub-{sub}_task-fer_run-{run}_events.tsv"
    eeg_edf_path = sub_dir / "eeg" / f"sub-{sub}_task-fer_run-{run}_eeg.edf"
    
    df_events = pd.read_csv(events_path, sep="\t")
    trial_events = df_events[df_events["trial"] == trial_num]
    
    base_row = trial_events[trial_events["flag"] == "second_fix"].iloc[0]
    stim_row = trial_events[trial_events["flag"] == "video"].iloc[0]
    
    base_start, base_end = base_row["onset"], base_row["onset"] + base_row["duration"]
    stim_start, stim_end = stim_row["onset"], stim_row["onset"] + stim_row["duration"]
    
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(eeg_edf_path, preload=True, verbose=False)
    raw.pick_channels(channels)
    raw.reorder_channels(channels)
    raw_data = raw.get_data() * 1e6 # Convert to microvolts for readability of PSD
    eeg_times = raw.times
    sfreq = raw.info["sfreq"]
    
    bands = {"Beta": (13.0, 30.0)} # Focus only on Beta to keep output clean
    
    base_mask = (eeg_times >= base_start) & (eeg_times <= base_end)
    stim_mask = (eeg_times >= stim_start) & (eeg_times <= stim_end)
    
    results = {}
    for band_name, (l, h) in bands.items():
        psd_series, log_psd_series, de_series = compute_features(raw_data, sfreq, l, h)
        
        # Calculate Delta (Stimulus - Baseline) for each feature
        delta_psd = np.mean(psd_series[:, stim_mask], axis=-1) - np.mean(psd_series[:, base_mask], axis=-1)
        delta_log_psd = np.mean(log_psd_series[:, stim_mask], axis=-1) - np.mean(log_psd_series[:, base_mask], axis=-1)
        delta_de = np.mean(de_series[:, stim_mask], axis=-1) - np.mean(de_series[:, base_mask], axis=-1)
        
        results[band_name] = {
            "PSD": dict(zip(channels, delta_psd)),
            "Log_PSD": dict(zip(channels, delta_log_psd)),
            "DE": dict(zip(channels, delta_de)),
        }
        
    return results

def print_feature_comparison(feat_name, res_t1, res_t3, channels):
    print(f"\n[{feat_name}]")
    print(f"{'Channel':<10} | {'T1 (Sad)':<15} | {'T3 (Happy)':<15}")
    print("-" * 46)
    for ch in channels:
        v1 = res_t1["Beta"][feat_name][ch]
        v3 = res_t3["Beta"][feat_name][ch]
        print(f"{ch:<10} | {v1:>15.4f} | {v3:>15.4f}")
        
    fai_t1 = res_t1["Beta"][feat_name]["Fp2"] - res_t1["Beta"][feat_name]["Fp1"]
    fai_t3 = res_t3["Beta"][feat_name]["Fp2"] - res_t3["Beta"][feat_name]["Fp1"]
    print("-" * 46)
    print(f"Bất đối xứng (Fp2 - Fp1):")
    print(f"Sad   : {fai_t1:+.4f}")
    print(f"Happy : {fai_t3:+.4f}")

def main():
    sub = "acl"
    run = 0
    channels = ["Fp1", "Fp2", "F3", "F4", "O1", "O2"]
    
    print("Extracting Data...")
    res_t1 = analyze_trial(sub, run, 1, channels)
    res_t3 = analyze_trial(sub, run, 3, channels)
    
    print("\n" + "="*50)
    print("SO SÁNH CÁC ĐẶC TRƯNG EEG: DẢI SÓNG BETA (13-30 Hz)")
    print("Mức thay đổi so với Baseline (Delta)")
    print("="*50)
    
    print_feature_comparison("PSD", res_t1, res_t3, channels)
    print_feature_comparison("Log_PSD", res_t1, res_t3, channels)
    print_feature_comparison("DE", res_t1, res_t3, channels)

if __name__ == "__main__":
    main()
