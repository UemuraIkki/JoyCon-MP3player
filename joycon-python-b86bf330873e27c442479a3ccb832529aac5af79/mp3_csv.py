import numpy as np
import librosa
import csv
from pathlib import Path

def analyze_audio_for_joycon_dsp(file_path: str, fps: int = 66) -> list:
    print(f"[{file_path}] の処理開始")

    y, sr = librosa.load(file_path, sr=None, mono=True)
    
    # 2. HPSS（調波・打楽器音分離）の実行
    print("打楽器成分（ノイズ）の分離フィルターを適用中...")
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=1.2)
    
    # 3. STFT解析（打楽器が除去されたクリーンな y_harmonic を使用）
    print("純粋なメロディ成分のFFT解析を実行中...")
    hop_length = int(sr / fps)
    stft_matrix = np.abs(librosa.stft(y_harmonic, hop_length=hop_length))
    frequencies = librosa.fft_frequencies(sr=sr)
    
    # 余計な高音域はモーターの追従を妨げるため、上限を1000Hzに制限
    lf_mask = (frequencies >= 40.0) & (frequencies < 160.0)
    hf_mask = (frequencies >= 160.0) & (frequencies <= 1250.0)
    
    commands = []
    
    for t in range(stft_matrix.shape[1]):
        frame = stft_matrix[:, t]
        lf_spectrum = frame[lf_mask]
        if len(lf_spectrum) > 0 and np.max(lf_spectrum) > 0.05:
            lf_idx = np.argmax(lf_spectrum)
            lf_freq = frequencies[lf_mask][lf_idx]
            lf_amp = float(np.max(lf_spectrum))
        else:
            lf_freq, lf_amp = 0.0, 0.0

        hf_spectrum = frame[hf_mask]
        if len(hf_spectrum) > 0 and np.max(hf_spectrum) > 0.05:
            hf_idx = np.argmax(hf_spectrum)
            hf_freq = frequencies[hf_mask][hf_idx]
            hf_amp = float(np.max(hf_spectrum))
        else:
            hf_freq, hf_amp = 0.0, 0.0
            
        lf_amp = min(1.0, lf_amp / 80.0) 
        hf_amp = min(1.0, hf_amp / 10.0)
        if hf_amp > 0.3:
            lf_amp = lf_amp * 0.2

        commands.append((float(hf_freq), float(hf_amp), float(lf_freq), float(lf_amp)))
        
    print(f"解析終了: 合計 {len(commands)} frames")
    return commands

def save_commands_to_csv(commands: list, output_path: str):
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['hf_freq', 'hf_amp', 'lf_freq', 'lf_amp'])
        for cmd in commands:
            writer.writerow(cmd)
    print(f"出力しました: {output_path}")

if __name__ == '__main__':
    script_dir = Path(__file__).parent
    mp3 = "HJ.mp3"


    mp3_path = script_dir / mp3
    csv_path = script_dir / f"{mp3.split('.')[0]}_commands.csv"

    if not mp3_path.exists():
        print(f"エラー: {mp3_path} が見つかりません。")
        exit()

    audio_commands = analyze_audio_for_joycon_dsp(str(mp3_path), fps=66)
    
    save_commands_to_csv(audio_commands, str(csv_path))