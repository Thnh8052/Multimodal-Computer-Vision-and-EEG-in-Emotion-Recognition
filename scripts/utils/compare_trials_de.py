import sys
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy.signal import hilbert

def compute_band_de(raw_data, sfreq, l_freq, h_freq):
    # Filter data
    filtered = mne.filter.filter_data(raw_data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False)
    # Envelope via Hilbert
    envelope = np.abs(hilbert(filtered, axis=-1))
    # PSD approx
    psd = envelope ** 2
    # Differential Entropy: 0.5 * ln(2 * pi * e * psd)
    de = 0.5 * np.log(2 * np.pi * np.e * (psd + 1e-15))
    return de

def analyze_trial(sub, run, trial_num, channels):
    sub = sub.replace("sub-", "")
    base_dir = Path("data/raw")
    sub_dir = base_dir / f"sub-{sub}"
    
    events_path = sub_dir / f"sub-{sub}_task-fer_run-{run}_events.tsv"
    eeg_edf_path = sub_dir / "eeg" / f"sub-{sub}_task-fer_run-{run}_eeg.edf"
    
    df_events = pd.read_csv(events_path, sep="\t")
    trial_events = df_events[df_events["trial"] == trial_num]
    
    # Get times for baseline (second_fix) and stimulus (video)
    base_row = trial_events[trial_events["flag"] == "second_fix"].iloc[0]
    stim_row = trial_events[trial_events["flag"] == "video"].iloc[0]
    
    base_start, base_end = base_row["onset"], base_row["onset"] + base_row["duration"]
    stim_start, stim_end = stim_row["onset"], stim_row["onset"] + stim_row["duration"]
    
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(eeg_edf_path, preload=True, verbose=False)
    raw.pick_channels(channels)
    
    # Ensure channels are in the requested order
    raw.reorder_channels(channels)
    raw_data = raw.get_data()
    eeg_times = raw.times
    sfreq = raw.info["sfreq"]
    
    bands = {
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 13.0),
        "Beta": (13.0, 30.0),
        "Gamma": (30.0, 45.0)
    }
    
    base_mask = (eeg_times >= base_start) & (eeg_times <= base_end)
    stim_mask = (eeg_times >= stim_start) & (eeg_times <= stim_end)
    
    results = {}
    for band_name, (l, h) in bands.items():
        de_series = compute_band_de(raw_data, sfreq, l, h) # shape (n_channels, n_times)
        
        base_de = np.mean(de_series[:, base_mask], axis=-1)
        stim_de = np.mean(de_series[:, stim_mask], axis=-1)
        
        # We care about the change from baseline (ERD/ERS)
        delta_de = stim_de - base_de
        
        results[band_name] = dict(zip(channels, delta_de))
        
    return results

def main():
    sub = "acl"
    run = 0
    channels = ["Fp1", "Fp2", "F3", "F4", "O1", "O2"]
    
    print("Extracting Data...")
    res_t1 = analyze_trial(sub, run, 1, channels) # Sad
    res_t3 = analyze_trial(sub, run, 3, channels) # Happy
    
    print("\n" + "="*50)
    print("TRIAL 1 (SAD) vs TRIAL 3 (HAPPY)")
    print("Giá trị thể hiện: Mức tăng/giảm Differential Entropy (DE) so với Baseline")
    print("="*50)
    
    for band in ["Theta", "Alpha", "Beta", "Gamma"]:
        print(f"\n[{band} Band]")
        print(f"{'Channel':<10} | {'T1 (Sad)':<12} | {'T3 (Happy)':<12}")
        print("-" * 40)
        for ch in channels:
            v1 = res_t1[band][ch]
            v3 = res_t3[band][ch]
            print(f"{ch:<10} | {v1:>12.3f} | {v3:>12.3f}")
            
        # Tính Frontal Asymmetry (Phải - Trái) cho Fp2-Fp1
        # Nếu (Phải - Trái) > 0 => Não phải hoạt động mạnh hơn => Tiêu cực
        fai_t1 = res_t1[band]["Fp2"] - res_t1[band]["Fp1"]
        fai_t3 = res_t3[band]["Fp2"] - res_t3[band]["Fp1"]
        print("-" * 40)
        print(f"Bất đối xứng (Fp2 - Fp1):")
        print(f"Sad   : {fai_t1:+.3f} (Dương = Tiêu cực, Âm = Tích cực)")
        print(f"Happy : {fai_t3:+.3f} (Dương = Tiêu cực, Âm = Tích cực)")

if __name__ == "__main__":
    main()
