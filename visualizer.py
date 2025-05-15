import librosa
import numpy as np
import cv2
import moviepy.editor as mpy
import os
from tqdm import tqdm


# ===== config:
# tên file = example.mp3
# background = background.jpg


# ===== install:
# pip install librosa numpy opencv-python tqdm
# pip install moviepy==1.0.3


# ===== run:
# python visualizer.py


CONFIG = {
    "audio_file": "example.mp3",
    "output_video": "output.mp4",
    "background_image_path": "background.jpg",
    "fps": 30,
    "video_width": 1280,
    "video_height": 720,
    "visualizer": {
        "type": "spectrum",
        "waveform": {
            "color": (255, 165, 0),
            "thickness": 2,
            "height_scale": 0.35,
            "style": "line",
            "smoothing_window": 5
        },
        "spectrum": {
            "bins": 64,
            "color_map": "viridis",
            "bar_width_factor": 0.9,
            "smoothing_factor": 0.6,
            "min_freq_hz": 50,
            "max_freq_hz": 16000,
            "log_freq_scale": True,
            "power_scale": 0.8,
            "gravity_effect": 0.05,
        }
    },
    "ffmpeg_params": {
        "codec": "libx264",
        "audio_codec": "aac",
        "preset": "medium",
        "threads": os.cpu_count() or 1,
        "logger": "bar"
    }
}

g_audio_data = None
g_sample_rate = None
g_samples_per_frame = 0
g_background_frame = None
g_prev_spectrum_magnitudes = None
g_spectrum_peak_magnitudes = None

def load_and_prepare_background(config):
    global g_background_frame
    bg_path = config.get("background_image_path")
    bg_color = config.get("background_color", (0,0,0))

    if bg_path and os.path.exists(bg_path):
        try:
            img = cv2.imread(bg_path)
            if img is None:
                raise ValueError("Cannot read background image.")
            h, w = img.shape[:2]
            target_w, target_h = config["video_width"], config["video_height"]
            target_aspect = target_w / target_h
            img_aspect = w / h

            if img_aspect > target_aspect:
                new_w = int(target_aspect * h)
                offset_w = (w - new_w) // 2
                img_cropped = img[:, offset_w : offset_w + new_w]
            else:
                new_h = int(w / target_aspect)
                offset_h = (h - new_h) // 2
                img_cropped = img[offset_h : offset_h + new_h, :]
            g_background_frame = cv2.resize(img_cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
            return
        except Exception:
            pass
    g_background_frame = np.full((config["video_height"], config["video_width"], 3), bg_color, dtype=np.uint8)

def get_colormap_gradient(name_or_tuple, num_steps):
    if isinstance(name_or_tuple, str):
        try:
            lut = np.arange(256, dtype=np.uint8).reshape(1, -1)
            colored_lut = cv2.applyColorMap(lut, getattr(cv2, f"COLORMAP_{name_or_tuple.upper()}"))
            indices = np.linspace(0, 255, num_steps, dtype=int)
            return [tuple(map(int, colored_lut[0, idx])) for idx in indices]
        except AttributeError:
            start_color = (0, 255, 0)
            end_color = (255, 0, 0)
            return [tuple(int(s + (e - s) * i / (num_steps -1 if num_steps > 1 else 1)) for s, e in zip(start_color, end_color))
                    for i in range(num_steps)]
    elif isinstance(name_or_tuple, tuple) and len(name_or_tuple) == 2:
        start_color, end_color = name_or_tuple
        return [tuple(int(s + (e - s) * i / (num_steps -1 if num_steps > 1 else 1)) for s, e in zip(start_color, end_color))
                for i in range(num_steps)]
    else:
        start_color = (0, 255, 0)
        end_color = (255, 0, 0)
        return [tuple(int(s + (e - s) * i / (num_steps -1 if num_steps > 1 else 1)) for s, e in zip(start_color, end_color))
                for i in range(num_steps)]

def draw_waveform(frame, audio_segment, config_vis):
    cfg = config_vis["waveform"]
    color = cfg["color"]
    thickness = cfg["thickness"]
    height_scale = cfg["height_scale"]
    center_y = frame.shape[0] // 2
    max_amplitude_draw = (frame.shape[0] // 2) * height_scale
    num_samples_audio = len(audio_segment)
    num_points_video = frame.shape[1]

    if num_samples_audio == 0: return frame

    points = np.empty((num_points_video, 2), dtype=np.int32)
    x_audio = np.linspace(0, num_samples_audio - 1, num_samples_audio)
    x_video = np.linspace(0, num_samples_audio - 1, num_points_video)
    y_audio_interp = np.interp(x_video, x_audio, audio_segment)

    points[:, 0] = np.arange(num_points_video)
    points[:, 1] = np.clip(
        center_y - (y_audio_interp * max_amplitude_draw),
        0, frame.shape[0]-1
    ).astype(np.int32)

    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return frame

def draw_spectrum(frame, audio_segment, config_vis, sample_rate):
    global g_prev_spectrum_magnitudes
    cfg = config_vis["spectrum"]
    num_bins = cfg["bins"]
    color_source = cfg["color_map"]
    bar_width_factor = cfg["bar_width_factor"]
    smoothing_factor = cfg["smoothing_factor"]
    min_freq = cfg["min_freq_hz"]
    max_freq = cfg["max_freq_hz"]
    log_freq = cfg["log_freq_scale"]
    power_scale = cfg["power_scale"]

    if g_prev_spectrum_magnitudes is None:
        g_prev_spectrum_magnitudes = np.zeros(num_bins)

    if len(audio_segment) == 0:
        current_magnitudes = np.zeros(num_bins)
    else:
        n_fft = 2048
        hop_length = max(1, len(audio_segment) // (num_bins * 2))

        if len(audio_segment) < n_fft:
            audio_segment_padded = np.pad(audio_segment, (0, n_fft - len(audio_segment)), 'constant')
        else:
            audio_segment_padded = audio_segment

        D = librosa.stft(audio_segment_padded, n_fft=n_fft, hop_length=hop_length, window='hann', center=False)
        magnitudes_stft = np.abs(D)
        freqs_stft = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        valid_freq_indices = np.where((freqs_stft >= min_freq) & (freqs_stft <= max_freq))[0]

        if len(valid_freq_indices) == 0:
            current_magnitudes = np.zeros(num_bins)
        else:
            magnitudes_filtered = magnitudes_stft[valid_freq_indices, :]
            freqs_filtered = freqs_stft[valid_freq_indices]
            avg_magnitudes_over_time = np.mean(magnitudes_filtered, axis=1)

            if log_freq and len(freqs_filtered) > 1:
                log_min_freq = np.log10(max(1,freqs_filtered[0]))
                log_max_freq = np.log10(freqs_filtered[-1])
                target_log_freqs = np.logspace(log_min_freq, log_max_freq, num_bins + 1)
            else:
                target_log_freqs = np.linspace(freqs_filtered[0], freqs_filtered[-1], num_bins + 1)

            binned_magnitudes = np.zeros(num_bins)
            for i in range(num_bins):
                bin_start_freq, bin_end_freq = target_log_freqs[i], target_log_freqs[i+1]
                stft_indices_in_bin = np.where((freqs_filtered >= bin_start_freq) & (freqs_filtered < bin_end_freq))[0]
                if len(stft_indices_in_bin) > 0:
                    binned_magnitudes[i] = np.mean(avg_magnitudes_over_time[stft_indices_in_bin])
                else:
                    binned_magnitudes[i] = 0

            if np.any(binned_magnitudes):
                total_energy = np.sum(binned_magnitudes**2)
                if total_energy > 1e-6 :
                    norm_factor = np.sqrt(total_energy) * 0.1 + np.mean(binned_magnitudes) * 0.5 + 1e-6
                    normalized_magnitudes = binned_magnitudes / norm_factor
                else:
                    normalized_magnitudes = binned_magnitudes
                normalized_magnitudes = np.clip(normalized_magnitudes, 0, 1.5)
                normalized_magnitudes = np.power(normalized_magnitudes, power_scale)
                normalized_magnitudes = np.clip(normalized_magnitudes, 0, 1)
            else:
                normalized_magnitudes = binned_magnitudes
            current_magnitudes = normalized_magnitudes

    smoothed_magnitudes = (1 - smoothing_factor) * current_magnitudes + smoothing_factor * g_prev_spectrum_magnitudes
    g_prev_spectrum_magnitudes = smoothed_magnitudes

    video_h, video_w = frame.shape[:2]
    colors = get_colormap_gradient(color_source, num_bins)
    bar_total_width_pixels = video_w / num_bins
    bar_actual_width = int(bar_total_width_pixels * bar_width_factor)
    bar_spacing = bar_total_width_pixels - bar_actual_width

    for i in range(num_bins):
        magnitude = smoothed_magnitudes[i]
        bar_height = int(magnitude * (video_h * 0.9))
        x1 = int(i * bar_total_width_pixels + bar_spacing / 2)
        y1 = video_h - bar_height
        x2 = int(x1 + bar_actual_width)
        y2 = video_h
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[i], -1)
    return frame

def make_video_frame(t):
    global g_audio_data, g_sample_rate, g_samples_per_frame, g_background_frame, CONFIG
    frame_idx = int(t * CONFIG["fps"])
    start_sample = frame_idx * g_samples_per_frame
    end_sample = min(start_sample + g_samples_per_frame, len(g_audio_data))
    audio_segment = g_audio_data[start_sample:end_sample]
    current_frame_bg = g_background_frame.copy()
    vis_type = CONFIG["visualizer"]["type"]
    if vis_type == "waveform":
        draw_waveform(current_frame_bg, audio_segment, CONFIG["visualizer"])
    elif vis_type == "spectrum":
        draw_spectrum(current_frame_bg, audio_segment, CONFIG["visualizer"], g_sample_rate)
    return cv2.cvtColor(current_frame_bg, cv2.COLOR_BGR2RGB)

def generate_video(config):
    global g_audio_data, g_sample_rate, g_samples_per_frame
    try:
        g_audio_data, g_sample_rate = librosa.load(config['audio_file'], sr=None)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return

    duration = librosa.get_duration(y=g_audio_data, sr=g_sample_rate)
    if duration == 0:
        print("Error: Audio file is empty or invalid.")
        return

    total_frames = int(duration * config['fps'])
    if total_frames == 0 : # Audio quá ngắn hoặc FPS=0
        print("Error: Audio too short or FPS too low resulting in zero frames.")
        return

    g_samples_per_frame = len(g_audio_data) // total_frames
    if g_samples_per_frame == 0 and len(g_audio_data) > 0 : # Vẫn có audio nhưng không đủ cho 1 frame
         print("Warning: Audio samples per frame is zero, but audio data exists. Adjusting FPS or audio might be needed.")
         g_samples_per_frame = 1
    elif len(g_audio_data) == 0 and g_samples_per_frame == 0:
        print("Error: No audio data and zero samples per frame.")
        return

    load_and_prepare_background(config)
    video_clip = mpy.VideoClip(make_video_frame, duration=duration)
    audio_mpy_clip = None
    final_clip = None
    try:
        audio_mpy_clip = mpy.AudioFileClip(config['audio_file'])
        final_clip = video_clip.set_audio(audio_mpy_clip)
        final_clip.write_videofile(
            config['output_video'],
            fps=config['fps'],
            codec=config["ffmpeg_params"]["codec"],
            audio_codec=config["ffmpeg_params"]["audio_codec"],
            preset=config["ffmpeg_params"]["preset"],
            threads=config["ffmpeg_params"]["threads"],
            logger=config["ffmpeg_params"]["logger"]
        )
    except Exception as e:
        print(f"Critical error during video writing: {e}")
    finally:
        if audio_mpy_clip: audio_mpy_clip.close()
        if hasattr(video_clip, 'close') and callable(video_clip.close): video_clip.close() # Moviepy 1.0.3+
        if final_clip and hasattr(final_clip, 'close') and callable(final_clip.close): final_clip.close()


if __name__ == "__main__":
    if not os.path.exists(CONFIG["audio_file"]):
        print(f"ERROR: Audio file '{CONFIG['audio_file']}' not found!")
    else:
        generate_video(CONFIG)
        print("Processing finished.")
