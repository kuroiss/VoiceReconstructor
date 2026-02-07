import sys
import numpy as np
import sounddevice as sd
from scipy.fft import fft
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

# --- 設定 ---
FS = 44100  # サンプリングレート
DURATION_REC = 5  # 録音時間 (秒)
TARGET_TIME = 3  # 解析対象の時点 (秒)
ANALYSIS_WINDOW = 0.05  # 解析する時間幅 (秒)
NUM_PEAKS = 100  # 抽出するピーク数
STEP_DURATION = 0.5  # 各ステップの再生時間 (秒)

class VoiceSpectrumVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Spectrum Synthesis Visualizer")
        self.resize(800, 600)

        # PyQTGraphの設定
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.plot_widget = pg.PlotWidget()
        self.layout.addWidget(self.plot_widget)
        
        # グラフ軸の設定
        self.plot_widget.setLabel('left', 'Amplitude')
        self.plot_widget.setLabel('bottom', 'Frequency [Hz]')
        self.plot_widget.setTitle("Adding Frequencies to Spectrum")
        self.plot_widget.setXRange(0, 4000) # 人の声の主要範囲を表示
        self.plot_widget.setYRange(0, 0.05) # 振幅の最大値に合わせて調整

        # 解析データの準備
        self.top_freqs = None
        self.top_amps = None
        self.top_phases = None
        self.current_step = 0
        
        # 表示用のバーグラフオブジェクトを初期化
        self.bar_graph = pg.BarGraphItem(x=[], height=[], width=20, brush='y')
        self.plot_widget.addItem(self.bar_graph)

        # 録音・解析の開始
        self.prepare_data()

        # タイマー設定 (0.5秒ごとにデータを更新)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_spectrum_and_sound)
        self.timer.start(int(STEP_DURATION * 1000))

    def prepare_data(self):
        print("録音を開始します...")
        recording = sd.rec(int(DURATION_REC * FS), samplerate=FS, channels=1, dtype='float32')
        sd.wait()
        print("録音完了。解析中...")
        audio_data = recording.flatten()

        # 解析対象の取得
        start_idx = int(TARGET_TIME * FS)
        end_idx = start_idx + int(ANALYSIS_WINDOW * FS)
        window_data = audio_data[start_idx:end_idx]

        # FFT
        n = len(window_data)
        fft_data = fft(window_data)
        freqs = np.fft.fftfreq(n, 1/FS)
        amplitudes = np.abs(fft_data) * 2 / n
        phases = np.angle(fft_data)

        # 正の周波数成分
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        amplitudes = amplitudes[pos_mask]
        phases = phases[pos_mask]

        # ピークの抽出
        sorted_indices = np.argsort(amplitudes)[::-1]
        top_indices = sorted_indices[:NUM_PEAKS]

        self.top_freqs = freqs[top_indices]
        self.top_amps = amplitudes[top_indices]
        self.top_phases = phases[top_indices]
        
        # 音合成用の時間軸
        self.t_sound = np.linspace(0, STEP_DURATION, int(FS * STEP_DURATION), endpoint=False)
        self.current_wave = np.zeros_like(self.t_sound)
        
        print("合成を開始します。")

    def update_spectrum_and_sound(self):
        if self.current_step < NUM_PEAKS:
            # 1. グラフの更新 (スペクトル)
            # 現在のステップまでの周波数と振幅を取得
            freqs_to_plot = self.top_freqs[:self.current_step + 1]
            amps_to_plot = self.top_amps[:self.current_step + 1]
            
            # バーグラフのデータを更新
            self.bar_graph.setOpts(x=freqs_to_plot, height=amps_to_plot)
            
            # 2. 音声合成と再生
            f = self.top_freqs[self.current_step]
            a = self.top_amps[self.current_step]
            p = self.top_phases[self.current_step]
            
            # 波形は全合成波の累積ではなく、このステップの音だけを鳴らすか、
            # 前回のプログラムのように全合成波を鳴らすか選べますが、
            # 「追加される様子」を聞くため、ここでも全合成波を流します
            self.current_wave += a * np.sin(2 * np.pi * f * self.t_sound + p)
            
            # 正規化
            max_amp = np.max(np.abs(self.current_wave))
            normalized_wave = self.current_wave / max_amp if max_amp > 0 else self.current_wave
            
            # 音声再生
            sd.play(normalized_wave, FS)
            
            print(f"Step {self.current_step+1}/{NUM_PEAKS}: Freq {f:.1f}Hz added.")
            self.current_step += 1
        else:
            self.timer.stop()
            print("完了。")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = VoiceSpectrumVisualizer()
    window.show()
    sys.exit(app.exec_())