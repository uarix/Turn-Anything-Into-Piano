import librosa
import numpy as np
from midiutil import MIDIFile
from scipy.signal import find_peaks

def audio_to_midi(
    input_file: str,
    output_file: str,
    n_fft: int = 8192,
    hop_length: int = 256,
    threshold_percentile: float = 95,
    max_peaks_per_frame: int = 20,
    tempo: int = 120,
    base_velocity: int = 50,
    velocity_scale: float = 0.5
) -> None:
    """
    将音频文件转换为钢琴MIDI文件

    参数:
    input_file: 输入音频文件路径
    output_file: 输出MIDI文件路径
    n_fft: FFT窗口大小
    hop_length: STFT帧移
    threshold_percentile: 峰值检测阈值百分比
    max_peaks_per_frame: 每帧最大音符数
    tempo: MIDI速度(BPM)
    base_velocity: 基础力度值
    velocity_scale: 力度缩放因子
    """

    # 加载音频文件
    y, sr = librosa.load(input_file, sr=None, mono=True)

    # 计算STFT
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
    magnitude = np.abs(stft)

    # 获取频率和时间信息
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.times_like(stft, sr=sr, hop_length=hop_length, n_fft=n_fft)

    # 计算全局幅度阈值
    threshold = np.percentile(magnitude.flatten(), threshold_percentile)

    # 创建MIDI文件（使用16个音轨）
    midi = MIDIFile(16, adjust_origin=False, ticks_per_quarternote=960)
    beat_duration = 60.0 / tempo  # 每beat秒数

    # 初始化音轨
    for track in range(16):
        midi.addTrackName(track, 0, f"Piano Track {track+1}")
        midi.addTempo(track, 0, tempo)

    # 跟踪音轨使用情况
    current_track = 0

    # 处理每个时间帧
    for i, time_sec in enumerate(times):
        frame = magnitude[:, i]
        peaks, properties = find_peaks(frame, height=threshold, distance=10)
        
        if len(peaks) == 0:
            continue

        # 按幅度降序排序
        sorted_indices = np.argsort(properties['peak_heights'])[::-1]
        peak_indices = peaks[sorted_indices[:max_peaks_per_frame]]

        # 处理每个峰值
        for idx in peak_indices:
            freq = freqs[idx]
            amp = frame[idx]

            # 跳过无效频率
            if freq < 20 or freq > 4186:  # 钢琴频率范围(A0=27.5Hz到C8≈4186Hz)
                continue

            # 计算MIDI音符编号
            midi_num = int(round(69 + 12 * np.log2(freq / 440.0)))
            midi_num = np.clip(midi_num, 21, 108)  # 限制在钢琴范围内

            # 计算力度（使用对数缩放更符合人耳感知）
            velocity = int(base_velocity + velocity_scale * 
                         np.interp(np.log(amp), 
                                 [np.log(threshold), np.log(np.max(magnitude))],
                                 [0, 127 - base_velocity]))

            # 转换时间
            time_beat = time_sec / beat_duration
            duration_beat = (hop_length / sr) / beat_duration

            # 分配到不同音轨实现复音
            midi.addNote(current_track, 0, midi_num, time_beat, duration_beat, velocity)
            current_track = (current_track + 1) % 16  # 循环使用16个音轨

    # 保存MIDI文件
    with open(output_file, "wb") as f:
        midi.writeFile(f)

if __name__ == "__main__":
    audio_to_midi("input.mp3", "output.mid")