import librosa
import numpy as np
from midiutil import MIDIFile
from scipy.signal import find_peaks
import concurrent.futures
import multiprocessing as mp
import os

# 全局变量用于存放共享内存数据
g_magnitude = None
g_times = None
g_freqs = None

def init_shared_arrays(mag_shared, mag_shape, times_shared, times_shape, freqs_shared, freqs_shape):
    """
    initializer 函数：在子进程中初始化全局共享内存变量
    """
    global g_magnitude, g_times, g_freqs
    g_magnitude = np.frombuffer(mag_shared, dtype=np.float32).reshape(mag_shape)
    g_times = np.frombuffer(times_shared, dtype=np.float64)  # times 一维数组
    g_freqs = np.frombuffer(freqs_shared, dtype=np.float64)

def process_chunk_optimized(args):
    """
    子进程中执行的块处理函数，此时直接使用全局变量 g_magnitude, g_times, g_freqs
    """
    (start, end, threshold, sr, hop_length,
     max_peaks_per_frame, base_velocity, velocity_scale, tempo,
     mag_log_threshold, mag_log_max) = args

    events = []
    beat_duration = 60.0 / tempo
    
    # 预计算有效频率索引
    # valid_freq_idx = np.where((g_freqs >= 20) & (g_freqs <= 4186))[0]
    
    for i in range(start, end):
        time_sec = g_times[i]
        frame = g_magnitude[:, i]
        
        # 峰值检测
        peaks, properties = find_peaks(frame, height=threshold, distance=10)
        if len(peaks) == 0:
            continue
            
        # 选择当前帧内的 top_k 峰值
        top_k = min(max_peaks_per_frame, len(peaks))
        if top_k < len(peaks):
            peak_heights = properties['peak_heights']
            partition_idx = np.argpartition(peak_heights, -top_k)[-top_k:]
            peak_indices = peaks[partition_idx]
        else:
            peak_indices = peaks

        for local_index, idx in enumerate(peak_indices):
            # 获取频率和幅值
            freq = g_freqs[idx]
            amp = frame[idx]
            
            # 计算 MIDI 音符号
            midi_num = int(round(69 + 12 * np.log2(freq / 440.0)))
            midi_num = np.clip(midi_num, 21, 108)
            
            # 根据幅值计算力度
            log_amp = np.log(amp)
            velocity = base_velocity + int(velocity_scale * 
                np.interp(log_amp, [mag_log_threshold, mag_log_max], [0, 127 - base_velocity]))
            
            # 时间转换
            time_beat = time_sec / beat_duration
            duration_beat = (hop_length / sr) / beat_duration
            
            events.append((time_beat, duration_beat, midi_num, velocity, i, local_index))
    
    return events

def audio_to_midi_optimized(
    input_file: str,
    output_file: str,
    n_fft: int = 4096,
    hop_length: int = 512,
    threshold_percentile: float = 95,
    max_peaks_per_frame: int = 160,
    tempo: int = 120,
    base_velocity: int = 50,
    velocity_scale: float = 0.5
) -> None:
    """
    音频转 MIDI 函数，使用共享内存和并行处理
    """
    # 加载音频
    y, sr = librosa.load(input_file, sr=None, mono=True)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
    magnitude = np.abs(stft).astype(np.float32)  # float32 降低内存占用
    
    # 频率计算和过滤
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    valid_mask = (freqs >= 20) & (freqs <= 4186)
    magnitude = magnitude[valid_mask]
    freqs = freqs[valid_mask]
    times = librosa.times_like(stft, sr=sr, hop_length=hop_length, n_fft=n_fft)
    
    # 创建共享内存
    mag_shared = mp.RawArray('f', magnitude.size)
    mag_np = np.frombuffer(mag_shared, dtype=np.float32).reshape(magnitude.shape)
    np.copyto(mag_np, magnitude)
    
    times_shared = mp.RawArray('d', times.size)
    times_np = np.frombuffer(times_shared, dtype=np.float64)
    np.copyto(times_np, times)
    
    freqs_shared = mp.RawArray('d', freqs.size)
    freqs_np = np.frombuffer(freqs_shared, dtype=np.float64)
    np.copyto(freqs_np, freqs)
    
    # 阈值计算
    threshold = np.percentile(mag_np, threshold_percentile)
    mag_max = np.max(mag_np)
    mag_log_threshold = np.log(threshold)
    mag_log_max = np.log(mag_max)
    
    # 分块参数
    total_frames = mag_np.shape[1]
    num_cores = os.cpu_count() or 4
    chunk_size = max(500, total_frames // (num_cores * 4))
    
    # 准备任务参数（注意：不再传递共享内存对象，而是由子进程的 initializer 初始化）
    args_list = []
    for start in range(0, total_frames, chunk_size):
        end = min(start + chunk_size, total_frames)
        args = (start, end, threshold, sr, hop_length,
                max_peaks_per_frame, base_velocity, velocity_scale, tempo,
                mag_log_threshold, mag_log_max)
        args_list.append(args)
    
    # 使用 ProcessPoolExecutor，并在每个子进程中初始化共享内存
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_cores,
            initializer=init_shared_arrays,
            initargs=(mag_shared, mag_np.shape, times_shared, times_np.shape, freqs_shared, freqs_np.shape)
        ) as executor:
        all_events = []
        for events in executor.map(process_chunk_optimized, args_list):
            all_events.extend(events)
    
    # 对事件进行排序，并生成 MIDI 文件
    all_events.sort(key=lambda x: (x[0], x[4], x[5]))
    
    midi = MIDIFile(16, adjust_origin=False, ticks_per_quarternote=960)
    for track in range(16):
        midi.addTrackName(track, 0, f"Track {track+1}")
        midi.addTempo(track, 0, tempo)
    
    for i, (time_beat, duration_beat, midi_num, velocity, _, _) in enumerate(all_events):
        track = i % 16
        midi.addNote(track, 0, midi_num, time_beat, duration_beat, velocity)
    
    with open(output_file, "wb") as f:
        midi.writeFile(f)

if __name__ == "__main__":
    audio_to_midi_optimized("【循环向】跟着雷总摇起来！Are you OK！.m4a", "【循环向】跟着雷总摇起来！Are you OK！.m4a.mid")
