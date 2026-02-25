import time
import math
import csv
from pathlib import Path
from pyjoycon import JoyCon
from pyjoycon.device import get_L_id, get_R_id

# ==========================================
# 1. データ変換ロジック
# ==========================================
def encode_joycon_rumble(hf_freq: float, hf_amp: float, lf_freq: float, lf_amp: float) -> bytes:
    if hf_amp == 0.0 and lf_amp == 0.0:
        return b'\x00\x01\x40\x40'
 
    hf_freq = max(0.0, min(1252.0, hf_freq))
    lf_freq = max(0.0, min(1252.0, lf_freq))
    hf_amp  = max(0.0, min(1.0, hf_amp))
    lf_amp  = max(0.0, min(1.0, lf_amp))

    encoded_hex_freq_hf = int(round(math.log2(hf_freq / 10.0) * 32.0)) if hf_freq >= 10.0 else 0
    hf = (encoded_hex_freq_hf - 0x60) * 4

    encoded_hex_amp_hf = _encode_amplitude(hf_amp)
    hf_amp_byte = encoded_hex_amp_hf * 2

    encoded_hex_freq_lf = int(round(math.log2(lf_freq / 10.0) * 32.0)) if lf_freq >= 10.0 else 0
    lf = encoded_hex_freq_lf - 0x40

    encoded_hex_amp_lf = _encode_amplitude(lf_amp)
    lf_amp_byte = (encoded_hex_amp_lf // 2) + 64
    byte0 = hf & 0xFF
    byte1 = (hf_amp_byte + ((hf >> 8) & 0xFF)) & 0xFF
    byte2 = (lf + ((lf_amp_byte >> 8) & 0xFF)) & 0xFF
    byte3 = lf_amp_byte & 0xFF

    return bytes([byte0, byte1, byte2, byte3])

def _encode_amplitude(amp: float) -> int:
    if amp == 0.0:
        return 0
    elif amp > 0.23:
        return int(round(math.log2(amp * 8.7) * 32.0))
    elif amp > 0.12:
        return int(round(math.log2(amp * 17.0) * 16.0))
    else:
        val = int(round(math.log2(amp * 120.0) * 8.0))
        return max(0, val)

# ==========================================
# 2. カスタムJoyConクラス
# ==========================================
class AudioJoyCon(JoyCon):
    def send_rumble_data(self, rumble_bytes: bytes):
        if len(rumble_bytes) != 8:
            raise ValueError("not valid rumble data")
        
        self._RUMBLE_DATA = rumble_bytes
        self._write_output_report(b'\x10', b'', b'')

def load_commands_from_csv(csv_path: str) -> list:
    commands = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # ヘッダー行をスキップ
        for row in reader:
            if len(row) == 4:
                commands.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
    print(f"CSV読み込み完了: {len(commands)} フレーム")
    return commands

def play_audio_on_joycon(joycon: AudioJoyCon, commands: list, fps: int = 66):
    frame_duration = 1.0 / fps
    
    print("再生を開始します...")
    start_time = time.perf_counter() 

    for i, cmd in enumerate(commands):
        hf_f, hf_a, lf_f, lf_a = cmd
        single_motor_data = encode_joycon_rumble(hf_f, hf_a, lf_f, lf_a)
        joycon.send_rumble_data(single_motor_data + single_motor_data)

        # 2. 次のフレームの開始予定時刻を計算
        next_frame_time = start_time + (i + 1) * frame_duration
        
        # 3. 予定時刻が来るまで待機
        while time.perf_counter() < next_frame_time:
            pass

    print("再生完了。振動を停止します。")
    stop_data = encode_joycon_rumble(0.0, 0.0, 0.0, 0.0)
    joycon.send_rumble_data(stop_data + stop_data)

def play_audio_on_joycons(joycons: list, commands: list, fps: int = 66):
    frame_duration = 1.0 / fps
    
    print(f"再生を開始します... (同期デバイス数: {len(joycons)}台)")
    start_time = time.perf_counter() 

    for i, cmd in enumerate(commands):
        hf_f, hf_a, lf_f, lf_a = cmd
        single_motor_data = encode_joycon_rumble(hf_f, hf_a, lf_f, lf_a)
        full_data = single_motor_data + single_motor_data

        # 接続されているすべてのJoy-Conに、タイムラグを最小限に抑えて連続送信
        for jc in joycons:
            jc.send_rumble_data(full_data)

        # 次のフレームの開始予定時刻を計算
        next_frame_time = start_time + (i + 1) * frame_duration
        
        # 予定時刻が来るまでビジーウェイト
        while time.perf_counter() < next_frame_time:
            pass

    print("再生完了。すべての振動を停止します。")
    stop_data = encode_joycon_rumble(0.0, 0.0, 0.0, 0.0)
    for jc in joycons:
        jc.send_rumble_data(stop_data + stop_data)

if __name__ == '__main__':
    script_dir = Path(__file__).parent
    csv_path = script_dir / "beyond_commands.csv"

    if not csv_path.exists():
        print(f"エラー: {csv_path} が見つかりません。")
        exit()

    audio_commands = load_commands_from_csv(str(csv_path))

    # --- 論理的なデバイス検出と初期化 ---
    active_joycons = []
    
    l_id = get_L_id()
    if None not in l_id:
        active_joycons.append(AudioJoyCon(*l_id))
        print("左Joy-Con(L) の接続を確立しました。")
        
    r_id = get_R_id()
    if None not in r_id:
        active_joycons.append(AudioJoyCon(*r_id))
        print("右Joy-Con(R) の接続を確立しました。")

    if not active_joycons:
        print("エラー: 制御可能なJoy-Conが見つかりません。Bluetoothのペアリング状態を確認してください。")
        exit()

    try:
        # 検出されたすべてのJoy-Conをリストとして渡す
        play_audio_on_joycons(active_joycons, audio_commands, fps=66)
    except KeyboardInterrupt:
        print("\nユーザーによって中断されました。すべての振動を強制停止します。")
        stop_data = encode_joycon_rumble(0.0, 0.0, 0.0, 0.0)
        for jc in active_joycons:
            jc.send_rumble_data(stop_data + stop_data)