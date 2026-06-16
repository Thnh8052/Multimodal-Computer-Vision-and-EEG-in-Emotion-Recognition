import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import mne
from scipy.signal import hilbert


def main():
    print("="*60)
    print("EDA VISUALIZATION: EEG BANDS TIMELINE (Theta, Alpha, Beta, Gamma)")
    print("="*60)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=str, help="Tên subject (vd: acl, acm)", default="")
    parser.add_argument("--run", type=str, help="Số run (vd: 0, 1)", default="")
    parser.add_argument("--trial", type=str, help="Số trial (vd: 1, 2) hoặc để trống", default="")
    parser.add_argument("--channel", type=str, help="Kênh điện cực (vd: Fp1, hoặc 'all' để vẽ cả 63 kênh)", default="all")
    parser.add_argument("--feature", type=str, help="Đặc trưng muốn vẽ: 'psd' (Log PSD) hoặc 'de' (Differential Entropy)", choices=["psd", "de"], default="psd")
    args = parser.parse_args()

    sub = args.sub
    run = args.run
    trial_input = args.trial
    target_channel = args.channel
    target_feature = args.feature.lower()

    if not sub:
        try:
            sub = input("Hãy nhập sub (vd: acl, acm): ").strip().lower()
        except EOFError:
            print("LỖI: Môi trường của bạn không hỗ trợ gõ phím trực tiếp. Vui lòng chạy lệnh kèm tham số, ví dụ:")
            print(f"python visualize_run_timeline.py --sub acl --run 0 --trial 1 --channel all")
            sys.exit(1)
            
    if not run:
        try:
            run = input("Hãy nhập run (vd: 0, 1): ").strip()
        except EOFError:
            pass
            
    if not sub or not run:
        print("LỖI: Bạn phải cung cấp sub và run!")
        sys.exit(1)
        
    if not trial_input:
        try:
            trial_input = input("Hãy nhập trial (vd: 1, 2) hoặc nhấn Enter để xem toàn bộ Run: ").strip()
        except EOFError:
            pass

    sub = sub.replace("sub-", "")
    run = run.replace("run-", "")
    
    base_dir = Path("data/raw")
    sub_dir = base_dir / f"sub-{sub}"
    
    if not sub_dir.exists():
        print(f"LỖI: Không tìm thấy thư mục {sub_dir}")
        sys.exit(1)
        
    events_path = sub_dir / f"sub-{sub}_task-fer_run-{run}_events.tsv"
    eeg_edf_path = sub_dir / "eeg" / f"sub-{sub}_task-fer_run-{run}_eeg.edf"
    
    for p in [events_path, eeg_edf_path]:
        if not p.exists():
            print(f"LỖI: Không tìm thấy file {p}")
            sys.exit(1)
            
    print("\n[1/2] Đang nạp Events (Cờ đánh dấu)...")
    df_events = pd.read_csv(events_path, sep="\t")
    
    print(f"[2/2] Đang nạp Sóng não EEG và trích xuất 4 dải tần (Theta, Alpha, Beta, Gamma)...")
    mne.set_log_level("ERROR")
    raw = mne.io.read_raw_edf(eeg_edf_path, preload=True)
    raw.pick_types(eeg=True) # Chỉ lấy các kênh EEG
    
    CORE18 = [
        "Fp1", "Fp2", 
        "AF3", "AF4", 
        "F3", "F4", 
        "F7", "F8", 
        "FC5", "FC6", 
        "T7", "T8", 
        "C3", "C4", 
        "P3", "P4", 
        "O1", "O2"
    ]
    
    if target_channel.lower() == "all":
        channels_to_process = raw.ch_names
    elif target_channel.lower() == "core":
        channels_to_process = [ch for ch in CORE18 if ch in raw.ch_names]
    else:
        if target_channel not in raw.ch_names:
            print(f"LỖI: Kênh {target_channel} không tồn tại. Các kênh hợp lệ: {raw.ch_names[:10]}...")
            sys.exit(1)
        channels_to_process = [target_channel]
        
    # Lấy dữ liệu của các kênh
    raw_data = raw.get_data(picks=channels_to_process) # shape: (n_channels, n_times)
    eeg_times = raw.times
    sfreq = raw.info["sfreq"]
    
    # Định nghĩa 4 dải tần
    bands = {
        "Theta (4-8 Hz)": (4.0, 8.0),
        "Alpha (8-13 Hz)": (8.0, 13.0),
        "Beta (13-30 Hz)": (13.0, 30.0),
        "Gamma Low (30-45 Hz)": (30.0, 45.0)
    }
    
    # Lọc và tính Envelope (Đường bao biên độ) dùng Hilbert transform
    band_envelopes = {}
    for name, (l_freq, h_freq) in bands.items():
        print(f"  -> Đang lọc dải {name} cho {len(channels_to_process)} kênh...")
        filtered = mne.filter.filter_data(raw_data, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False)
        envelope = np.abs(hilbert(filtered, axis=-1))
        
        # Smooth envelope (Moving average ~ 0.5s)
        window_size = int(sfreq * 0.5)
        kernel = np.ones(window_size) / window_size
        smoothed_env = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=-1, arr=envelope)
        
        band_envelopes[name] = smoothed_env
    
    # Cắt dữ liệu theo Trial
    target_trial_str = "FullRun"
    if trial_input:
        target_trial = int(trial_input)
        target_trial_str = f"Trial_{target_trial}"
        trial_events = df_events[df_events["trial"] == target_trial]
        if trial_events.empty:
            print(f"\nLỖI: Không tìm thấy Trial {target_trial} trong dữ liệu.")
            sys.exit(1)
            
        t_start = trial_events["onset"].min()
        t_end = (trial_events["onset"] + trial_events["duration"]).max()
        
        t_start = max(0, t_start - 2.0)
        t_end = t_end + 2.0
        
        df_events = trial_events
        
        mask = (eeg_times >= t_start) & (eeg_times <= t_end)
        eeg_times = eeg_times[mask]
        for name in bands.keys():
            band_envelopes[name] = band_envelopes[name][:, mask]
                
        title_suffix = f"| Trial: {target_trial}"
    else:
        title_suffix = f"| Toàn bộ Run"
        
    out_dir = Path(f"results/affec_v3/03_timeline_plots/sub-{sub}_run-{run}_{target_trial_str}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nĐang vẽ biểu đồ cho {len(channels_to_process)} kênh...")
    print(f"Hình ảnh sẽ được lưu tại: {out_dir}")
    
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"] # Blue, Green, Orange, Red
    
    flag_colors = {
        "trial": "#e7cb94",               
        "first_fix": "#aec7e8",           
        "scenario": "#ffbb78",            
        "second_fix": "#9edae5",          
        "video": "#ff9896",               
        "last_frame_video": "#c5b0d5",    
        "f_emotion_labelling": "#98df8a", 
        "p_emotion_labelling": "#f7b6d2"  
    }
    
    for i, ch_name in enumerate(channels_to_process):
        fig, ax2 = plt.subplots(1, 1, figsize=(16, 8))
        
        for idx, name in enumerate(bands.keys()):
            env_data = band_envelopes[name][i]
            psd = env_data ** 2
            
            if target_feature == "de":
                # Differential Entropy: DE = 0.5 * ln(2 * pi * e * sigma^2)
                # where sigma^2 is the variance (approximated by local PSD)
                val = 0.5 * np.log(2 * np.pi * np.e * (psd + 1e-15))
                offset = idx * 5.0 # DE values might need a slightly larger offset
                ylabel = "Differential Entropy (DE) + Offset"
            else:
                val = np.log10(psd + 1e-15)
                offset = idx * 3.0  
                ylabel = "Log10(PSD) + Offset"
                
            ax2.plot(eeg_times, val + offset, label=f"{name}", color=colors[idx], linewidth=1.5, alpha=0.9)
            
        ax2.set_ylabel(ylabel, fontsize=12)
        ax2.set_xlabel("Time (Seconds)", fontsize=12)
        ax2.set_title(f"EEG Band Power Envelopes ({ch_name}) - {target_feature.upper()} | Sub: {sub} | Run: {run} {title_suffix}", fontsize=16)
        ax2.set_yticks([]) 
        ax2.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
        ax2.grid(True, alpha=0.2, axis='x')
        
        for _, row in df_events.iterrows():
            onset = row["onset"]
            duration = row["duration"]
            flag = row["flag"]
            
            color = flag_colors.get(flag, "#dddddd")
            if flag != "trial":
                ax2.axvspan(onset, onset + duration, color=color, alpha=0.4, lw=0)
            else:
                ax2.axvline(onset, color="black", linestyle="--", alpha=0.6, lw=1.5)
            
            if trial_input and flag != "trial":
                ax2.text(onset + duration/2, ax2.get_ylim()[0] + (ax2.get_ylim()[1] - ax2.get_ylim()[0])*0.1, 
                         flag, color="black", fontsize=10, rotation=90, 
                         ha="center", va="bottom", alpha=0.8, fontweight="bold")
            
            if flag == "trial" and not trial_input:
                ax2.text(onset, ax2.get_ylim()[1]*0.9, f" T{int(row['trial'])}", 
                         color="black", fontweight="bold", ha="left", va="top")

        patches = [mpatches.Patch(color=c, label=l, alpha=0.6) for l, c in flag_colors.items()]
        fig.legend(handles=patches, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.95), fontsize=11)
        
        plt.tight_layout(rect=[0, 0, 0.88, 0.9])
        
        # Save figure and close to prevent memory leak
        out_file = out_dir / f"{ch_name}.png"
        plt.savefig(out_file, dpi=100)
        plt.close(fig)
        
        # Print progress bar
        sys.stdout.write(f"\r  Đã xuất {i+1}/{len(channels_to_process)} ảnh ({ch_name})...")
        sys.stdout.flush()
        
    print("\n[Hoàn tất!] Bạn có thể mở thư mục trên để xem ảnh của toàn bộ các kênh.")

if __name__ == "__main__":
    main()
