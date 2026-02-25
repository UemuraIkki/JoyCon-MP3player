# csv_emulator.py
import numpy as np
import sounddevice as sd
import csv
from pathlib import Path

def synthesize_joycon_audio(csv_path: str, fps: int = 66, sample_rate: int = 44100):
    """
    CSVのモーター制御コマンドから、PC再生用のオーディオ波形（サイン波）を数学的に合成する
    """
    commands = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # ヘッダーをスキップ
        for row in reader:
            if len(row) == 4:
                commands.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))

    print(f"CSV読み込み完了: {len(commands)} フレーム")
    print("仮想Joy-Con波形を合成中... ")

    # 1フレームあたりのサンプル数を計算（44100Hz / 66fps ≒ 668サンプル）
    samples_per_frame = int(sample_rate / fps)
    total_samples = len(commands) * samples_per_frame
    audio_data = np.zeros(total_samples, dtype=np.float32)

    # 波の切れ目で「ブチッ」というノイズ（ポップノイズ）が入るのを防ぐため、
    # 前のフレームからの波の角度（位相）を保持する
    hf_phase = 0.0
    lf_phase = 0.0

    for i, cmd in enumerate(commands):
        hf_f, hf_a, lf_f, lf_a = cmd
        start_idx = i * samples_per_frame
        end_idx = start_idx + samples_per_frame

        # 時間軸の配列を作成
        t = np.arange(samples_per_frame) / sample_rate

        # 高周波（HF）モーターのサイン波合成
        if hf_f > 0 and hf_a > 0:
            hf_wave = hf_a * np.sin(2 * np.pi * hf_f * t + hf_phase)
            hf_phase += 2 * np.pi * hf_f * (samples_per_frame / sample_rate)
        else:
            hf_wave = np.zeros(samples_per_frame)

        # 低周波（LF）モーターのサイン波合成
        if lf_f > 0 and lf_a > 0:
            lf_wave = lf_a * np.sin(2 * np.pi * lf_f * t + lf_phase)
            lf_phase += 2 * np.pi * lf_f * (samples_per_frame / sample_rate)
        else:
            lf_wave = np.zeros(samples_per_frame)

        # 2つのモーターの波形をミックスして格納
        audio_data[start_idx:end_idx] = hf_wave + lf_wave

    # PCのスピーカーが音割れ（クリッピング）を起こさないように音量を正規化
    max_val = np.max(np.abs(audio_data))
    if max_val > 0:
        audio_data = (audio_data / max_val) * 0.5  # 音量50%に設定

    return audio_data, sample_rate

if __name__ == '__main__':
    script_dir = Path(__file__).parent
    # 先ほどF0推定とダイナミックEQをかけて生成したCSVを指定
    csv_path = script_dir / "shunkan_commands.csv" 

    if not csv_path.exists():
        print(f"エラー: {csv_path} が見つかりません。")
        exit()

    try:
        # 音声の合成
        audio, sr = synthesize_joycon_audio(str(csv_path))
        
        # PCスピーカーで再生
        print("\nPCスピーカーでの再生を開始します。")
        print("（終了するには Ctrl+C を押してください）")
        sd.play(audio, sr)
        sd.wait() # 再生が終わるまで待機
        
    except KeyboardInterrupt:
        print("\n再生を強制停止しました。")
        sd.stop()
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")