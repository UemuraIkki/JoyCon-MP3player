import numpy as np
import librosa
import soundfile as sf
import scipy.signal
from pathlib import Path

def create_skeleton_audio(input_path: str, output_path: str):
    print(f"[{input_path}] の骨格化（解析用プレプロセス）を開始します...")
    
    # 音声の読み込み
    y, sr = librosa.load(input_path, sr=None, mono=True)
    
    # 1. HPSS: ここで先に打楽器成分を完全に消し去る
    print("打楽器・ノイズ成分を物理的に除去中...")
    y_harmonic, _ = librosa.effects.hpss(y, margin=2.0) # marginを強めにして徹底的に分離
    
    # 2. バンドパスフィルタ (300Hz - 1200Hz)
    print("Joy-Conの再生可能帯域外の音を殺棄中...")
    nyq = 0.5 * sr
    low = 300.0 / nyq
    high = 1200.0 / nyq
    b, a = scipy.signal.butter(4, [low, high], btype='band')
    y_filtered = scipy.signal.filtfilt(b, a, y_harmonic)
    
    # 3. ノイズゲート（閾値以下の音を無音化）
    print("ノイズゲート適用中...")
    threshold = 0.05
    y_gated = np.where(np.abs(y_filtered) > threshold, y_filtered, 0.0)
    
    # 4. ハードコンプレッション（音量の均一化）
    # 小さな音を持ち上げ、大きすぎる音を抑え込んで平坦にする
    print("ダイナミクスを破壊し、音圧を最大化中...")
    y_compressed = np.tanh(y_gated * 5.0) # tanhによるソフトクリッピングと増幅
    
    # 5. 音量の最終正規化
    y_normalized = y_compressed / np.max(np.abs(y_compressed))
    
    # WAVファイルとして書き出し
    sf.write(output_path, y_normalized, sr)
    print(f"骨格化オーディオの生成完了: {output_path}")

if __name__ == '__main__':
    script_dir = Path(__file__).parent
    mp3_name = "hakujitu.mp3"
    input_path = script_dir / mp3_name
    
    # 出力ファイル名（_skeleton.wav）
    output_path = script_dir / f"{mp3_name.split('.')[0]}_skeleton.wav"
    
    if not input_path.exists():
        print(f"エラー: {input_path} が見つかりません。")
        exit()
        
    create_skeleton_audio(str(input_path), str(output_path))