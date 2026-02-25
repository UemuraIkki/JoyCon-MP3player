import numpy as np
import librosa
import csv
import scipy.signal  # メディアンフィルタ用に追加
from pathlib import Path

# ==========================================
# エンジン1：STFT解析（旧方式・高速・全音域抽出）
# ==========================================
def analyze_with_stft(file_path: str, fps: int = 66) -> list:
    print(f"[{file_path}] のSTFT解析(高速・ピーク抽出)を開始します...")
    y, sr = librosa.load(file_path, sr=None, mono=True)
    
    y_harmonic, _ = librosa.effects.hpss(y, margin=1.2)
    
    hop_length = int(sr / fps)
    stft_matrix = np.abs(librosa.stft(y_harmonic, hop_length=hop_length))
    frequencies = librosa.fft_frequencies(sr=sr)
    
    lf_mask = (frequencies >= 40.0) & (frequencies < 160.0)
    hf_mask = (frequencies >= 160.0) & (frequencies <= 1000.0)
    
    commands = []
    for t in range(stft_matrix.shape[1]):
        frame = stft_matrix[:, t]
        
        lf_spectrum = frame[lf_mask]
        if len(lf_spectrum) > 0 and np.max(lf_spectrum) > 0.05:
            lf_f = float(frequencies[lf_mask][np.argmax(lf_spectrum)])
            lf_a = min(1.0, float(np.max(lf_spectrum)) / 80.0)
        else:
            lf_f, lf_a = 0.0, 0.0
            
        hf_spectrum = frame[hf_mask]
        if len(hf_spectrum) > 0 and np.max(hf_spectrum) > 0.05:
            hf_f = float(frequencies[hf_mask][np.argmax(hf_spectrum)])
            hf_a = min(1.0, float(np.max(hf_spectrum)) / 10.0)
        else:
            hf_f, hf_a = 0.0, 0.0
            
        commands.append((hf_f, hf_a, lf_f, lf_a))
        
    print(f"STFT解析完了: {len(commands)} frames")
    return commands

# ==========================================
# エンジン2：F0推定（新方式・高精度・メロディ特化・オートスケーリング付き）
# ==========================================
def analyze_with_f0(file_path: str, fps: int = 66) -> list:
    print(f"[{file_path}] のF0推定(高精度・メロディ抽出)を開始します...")
    y, sr = librosa.load(file_path, sr=None, mono=True)
    hop_length = int(sr / fps)
    
    y_harmonic, _ = librosa.effects.hpss(y, margin=1.2)
    rms = librosa.feature.rms(y=y_harmonic, hop_length=hop_length)[0]
    rms_normalized = rms / np.max(rms) if np.max(rms) > 0 else rms
    
    f0, voiced_flag, _ = librosa.pyin(
        y_harmonic, fmin=40.0, fmax=1200.0, sr=sr, hop_length=hop_length, fill_na=0.0
    )
    
    valid_f0 = f0[voiced_flag & (f0 > 0.0)]
    if len(valid_f0) > 0:
        p10_freq = np.percentile(valid_f0, 10)
        p90_freq = np.percentile(valid_f0, 90)
        shift_ratio = 400.0 / p10_freq
        if p90_freq * shift_ratio > 1100.0:
            shift_ratio = 1100.0 / p90_freq
        print(f"オートスケーリング適用: x{shift_ratio:.2f}倍")
    else:
        shift_ratio = 1.0

    commands = []
    num_frames = min(len(f0), len(rms_normalized))
    
    for t in range(num_frames):
        freq = float(f0[t])
        is_voiced = voiced_flag[t]
        raw_amp = float(rms_normalized[t])
        amp = raw_amp ** 0.7 if raw_amp > 0 else 0.0 
        
        hf_f, hf_a, lf_f, lf_a = 0.0, 0.0, 0.0, 0.0
        
        if is_voiced and amp > 0.05 and freq > 0.0:
            shifted_freq = freq * shift_ratio
            shifted_freq = max(300.0, min(1200.0, shifted_freq))
            
            distance_from_center = abs(shifted_freq - 600.0)
            eq_boost = 1.0 + ((distance_from_center / 400.0) ** 1.5) * 2.0
            
            hf_f = shifted_freq
            hf_a = min(1.0, amp * 1.5 * eq_boost)
            lf_f, lf_a = 0.0, 0.0

        commands.append((hf_f, hf_a, lf_f, lf_a))
        
    print(f"F0解析完了: {len(commands)} frames")
    return commands

# ==========================================
# 後処理：メディアンフィルタ（スパイクノイズ除去）
# ==========================================
def apply_median_filter(commands: list, kernel_size: int = 5) -> list:
    """
    配列データにメディアンフィルタを適用し、突発的な周波数・振幅のブレを平滑化する。
    kernel_size は必ず奇数（5フレーム = 約75msのノイズを無視する）
    """
    print(f"メディアンフィルタ（カーネルサイズ: {kernel_size}）を適用中...")
    
    # データを列ごとに分解
    hf_f_list = [c[0] for c in commands]
    hf_a_list = [c[1] for c in commands]
    lf_f_list = [c[2] for c in commands]
    lf_a_list = [c[3] for c in commands]

    # それぞれにフィルタをかける
    hf_f_filtered = scipy.signal.medfilt(hf_f_list, kernel_size)
    hf_a_filtered = scipy.signal.medfilt(hf_a_list, kernel_size)
    lf_f_filtered = scipy.signal.medfilt(lf_f_list, kernel_size)
    lf_a_filtered = scipy.signal.medfilt(lf_a_list, kernel_size)

    # 再びタプルのリストに結合
    filtered_commands = []
    for i in range(len(commands)):
        filtered_commands.append((
            float(hf_f_filtered[i]),
            float(hf_a_filtered[i]),
            float(lf_f_filtered[i]),
            float(lf_a_filtered[i])
        ))
        
    print("平滑化処理が完了しました。")
    return filtered_commands

# ==========================================
# 共通ロジック：CSV保存
# ==========================================
def save_commands_to_csv(commands: list, output_path: str):
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['hf_freq', 'hf_amp', 'lf_freq', 'lf_amp'])
        for cmd in commands:
            writer.writerow(cmd)
    print(f"CSVファイルを出力しました: {output_path}")



# ==========================================
# メイン実行ブロック
# ==========================================
if __name__ == '__main__':
    script_dir = Path(__file__).parent
    mp3_name = "shunkan_skeleton.wav"
    mp3_path = script_dir / mp3_name
    csv_path = script_dir / f"{mp3_name.split('.')[0]}_commands.csv"

    if not mp3_path.exists():
        print(f"エラー: {mp3_path} が見つかりません。")
        exit()

    # ★★★ ここでアルゴリズムと後処理を設計（選択）します ★★★
    # True = F0推定 (高精度)、False = STFT (高速)
    USE_F0_ALGORITHM = False 
    
    # True = メディアンフィルタを適用（STFTのノイズ除去に極めて有効）
    APPLY_MEDIAN_FILTER = True
    
    print("--- オーディオ解析パイプライン起動 ---")
    
    # 1. 解析フェーズ
    if USE_F0_ALGORITHM:
        audio_commands = analyze_with_f0(str(mp3_path), fps=66)
    else:
        audio_commands = analyze_with_stft(str(mp3_path), fps=66)
        
    # 2. 後処理フェーズ
    if APPLY_MEDIAN_FILTER:
        audio_commands = apply_median_filter(audio_commands, kernel_size=5)
    
    # 3. 出力フェーズ
    save_commands_to_csv(audio_commands, str(csv_path))