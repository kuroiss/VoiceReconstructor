import numpy as np
import sounddevice as sd
from scipy.fft import fft
import time

# --- 設定 ---
FS = 44100  # サンプリングレート
DURATION_REC = 5  # 録音時間 (秒)
TARGET_TIME = 3  # 解析対象の時点 (秒)
ANALYSIS_WINDOW = 0.05  # 解析する時間幅 (秒)
NUM_PEAKS = 100  # 抽出するピーク数
STEP_DURATION = 0.5  # 各ステップの再生時間 (秒)

print("録音を開始します...")
# --- 1. 録音 ---
recording = sd.rec(int(DURATION_REC * FS), samplerate=FS, channels=1, dtype='float32')
sd.wait()  # 録音終了まで待機
print("録音完了。")

# データを1次元に変換
audio_data = recording.flatten()

# --- 2. 特定時点の解析 ---
start_idx = int(TARGET_TIME * FS)
end_idx = start_idx + int(ANALYSIS_WINDOW * FS)
window_data = audio_data[start_idx:end_idx]

# FFTを実行
n = len(window_data)
fft_data = fft(window_data)
freqs = np.fft.fftfreq(n, 1/FS)

# 振幅と位相を計算
amplitudes = np.abs(fft_data) * 2 / n
phases = np.angle(fft_data)

# 正の周波数成分のみに限定
pos_mask = freqs > 0
freqs = freqs[pos_mask]
amplitudes = amplitudes[pos_mask]
phases = phases[pos_mask]

# --- 3. ピークの抽出 (上位100個) ---
# 振幅が大きい順にインデックスをソート
sorted_indices = np.argsort(amplitudes)[::-1]
top_indices = sorted_indices[:NUM_PEAKS]

top_freqs = freqs[top_indices]
top_amps = amplitudes[top_indices]
top_phases = phases[top_indices]

# --- 4. 音声の段階的な合成と再生 ---
print(f"振幅が大きい順に {NUM_PEAKS} 個のサイン波を追加して再生します。")
t = np.linspace(0, STEP_DURATION, int(FS * STEP_DURATION), endpoint=False)
current_wave = np.zeros_like(t)

for i in range(NUM_PEAKS):
    f = top_freqs[i]
    a = top_amps[i]
    p = top_phases[i]
    
    # サイン波を追加
    current_wave += a * np.sin(2 * np.pi * f * t + p)
    
    # 振幅を正規化（音が割れないように）
    max_amp = np.max(np.abs(current_wave))
    if max_amp > 0:
        normalized_wave = current_wave / max_amp
    else:
        normalized_wave = current_wave
    
    print(f"Step {i+1}/100: 周波数 {f:.1f}Hz を追加中...")
    
    # 再生
    sd.play(normalized_wave, FS)
    sd.wait()
    
    # 再生時間分待機
    time.sleep(0.1) 

print("全ステップ終了。")