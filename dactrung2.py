import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import lfilter
from scipy.signal.windows import hamming
from scipy.signal import find_peaks
import tempfile
import json
import math
import scipy.fftpack
import requests
from Predict import  dtw, dtw2, dtw_library
import json
import librosa

import io
def manual_spectral_centroid(file_path):
    y, sr = librosa.load(file_path)

    frame_length = int(0.025 * sr)  # 25ms
    hop_length = int(0.010 * sr)  # 10ms

    # Tách tín hiệu thành các frame với cửa sổ Hann
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    window = np.hanning(frame_length)

    centroids = []
    freqs = np.fft.rfftfreq(frame_length, 1.0 / sr)  # tần số tương ứng

    for frame in frames:
        windowed = frame * window
        spectrum = np.abs(np.fft.rfft(windowed))

        numerator = np.sum(freqs * spectrum)
        denominator = np.sum(spectrum) + 1e-10  # tránh chia 0

        centroid = numerator / denominator
        centroids.append(centroid)

    centroids = np.array(centroids).reshape(-1, 1)
    arr_list = centroids.tolist()
    arr_str = json.dumps(arr_list)
    return arr_str


def manual_spectral_bandwidth(file_path):
    y, sr = librosa.load(file_path)

    frame_length = int(0.025 * sr)  # 25 ms
    hop_length = int(0.010 * sr)    # 10 ms

    if len(y) < frame_length:
        return json.dumps([])

    # Tách tín hiệu thành các frame
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    window = np.hanning(frame_length)
    freqs = np.fft.rfftfreq(frame_length, 1.0 / sr)

    bandwidths = []

    for frame in frames:
        windowed = frame * window
        spectrum = np.abs(np.fft.rfft(windowed))  # magnitude spectrum

        # Normalize phổ để thành phân phối xác suất
        spectrum_sum = np.sum(spectrum) + 1e-10
        norm_spectrum = spectrum / spectrum_sum

        # Tính centroid trước
        centroid = np.sum(freqs * norm_spectrum)

        # Tính bandwidth
        deviation = freqs - centroid
        bw = np.sqrt(np.sum((deviation**2) * norm_spectrum))
        bandwidths.append(bw)

    # Đưa về dạng JSON
    bandwidths = np.array(bandwidths).reshape(-1, 1)
    arr_list = bandwidths.tolist()
    arr_str = json.dumps(arr_list)
    return arr_str


def manual_spectral_rolloff(file_path):
    y, sr = librosa.load(file_path)

    frame_length = int(0.025 * sr)  # 25ms
    hop_length = int(0.010 * sr)    # 10ms
    roll_percent = 0.85             # giống mặc định librosa

    if len(y) < frame_length:
        return json.dumps([])

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    window = np.hanning(frame_length)
    freqs = np.fft.rfftfreq(frame_length, 1.0 / sr)

    rolloffs = []

    for frame in frames:
        windowed = frame * window
        spectrum = np.abs(np.fft.rfft(windowed))

        total_energy = np.sum(spectrum)
        threshold = roll_percent * total_energy
        cumulative = np.cumsum(spectrum)

        # Tìm chỉ số f_r đầu tiên sao cho tích lũy ≥ ngưỡng
        rolloff_idx = np.where(cumulative >= threshold)[0]
        if len(rolloff_idx) > 0:
            fr = freqs[rolloff_idx[0]]  # chuyển chỉ số sang tần số
        else:
            fr = 0.0  # nếu không tìm được thì gán 0
        rolloffs.append(fr)

    rolloffs = np.array(rolloffs).reshape(-1, 1)
    arr_list = rolloffs.tolist()
    arr_str = json.dumps(arr_list)
    return arr_str

def manual_mfcc(file_path):
    y, sr = librosa.load(file_path)

    # 1. Pre-emphasis
    pre_emphasis = 0.97
    emphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # 2. Framing
    frame_size = 0.025
    frame_stride = 0.01
    frame_length = int(frame_size * sr)
    frame_step = int(frame_stride * sr)

    signal_length = len(emphasized)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros(pad_signal_length - signal_length)
    pad_signal = np.append(emphasized, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # 3. Windowing
    frames *= np.hamming(frame_length)

    # 4. FFT và power spectrum
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2)         # Power Spectrum

    # 5. Mel filterbank
    nfilt = 26
    low_freq = 0
    high_freq = sr / 2
    mel_low = 2595 * np.log10(1 + low_freq / 700)
    mel_high = 2595 * np.log10(1 + high_freq / 700)
    mel_points = np.linspace(mel_low, mel_high, nfilt + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin = np.floor((NFFT + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))
    for m in range(1, nfilt + 1):
        f_m_minus = bin[m - 1]
        f_m = bin[m]
        f_m_plus = bin[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1] + 1e-6)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m] + 1e-6)

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # tránh log(0)
    log_fbanks = np.log(filter_banks)

    # 6. DCT
    num_ceps = 13
    mfcc = scipy.fftpack.dct(log_fbanks, type=2, axis=1, norm='ortho')[:, :num_ceps]

    # Xuất dạng JSON
    arr_list = mfcc.tolist()
    arr_str = json.dumps(arr_list)
    return arr_str

if __name__ == "__main__":
    file_path = "https://www.dropbox.com/scl/fi/pw01c2aodvapzi7dt6z5h/video_34_chunk_2.wav?rlkey=13x4t6l48vv3mxxc5sd2r3rza&dl=0"
    file_path2 =  "https://www.dropbox.com/scl/fi/lixym1uhgaw2j8uvlnax6/video_34_chunk_3.wav?rlkey=0gjpdh5voffagpaf6dx1otz34&dl=0"
    # Chuyển link preview thành link tải file nhị phân trực tiếp
    direct_url = file_path.replace("dl=0", "dl=1")
    direct_url2 = file_path2.replace("dl=0", "dl=1")

    response = requests.get(direct_url)
    response2 = requests.get(direct_url2)

    audio_bytes = io.BytesIO(response.content)
    audio_bytes2 = io.BytesIO(response2.content)

    x= manual_mfcc(audio_bytes)
    x2 = manual_mfcc(audio_bytes2)

    d1 = json.loads(x)
    d2 = json.loads(x2)
    print(x)
    print(len(x))
    print(dtw_library(d1,d2))
