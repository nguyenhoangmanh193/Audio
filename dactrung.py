import requests
import librosa
import io
import os
import numpy as np
import matplotlib.pyplot as plt
import parselmouth
import tempfile
import json
from api import api_key
import assemblyai as aai
from Predict import  dtw, dtw2, dtw_library
import scipy.fftpack
# Link bạn cung cấp
# dropbox_preview_link = "https://www.dropbox.com/scl/fi/phsclrp8qmqxkpps9jguu/video_59_chunk_3.wav?rlkey=95pt94yklx7a0dxd6m9shm7cz&st=4s20ubcd&dl=0"
#
# # Chuyển link preview thành link tải file nhị phân trực tiếp
# direct_url = dropbox_preview_link.replace("dl=0", "dl=1")
# response = requests.get(direct_url)
# if response.status_code != 200:
#     print("Lỗi tải file:", response.status_code)
# audio_bytes = io.BytesIO(response.content)
API_KEY = api_key()
def text_convert(audio_bytes):
    aai.settings.api_key = API_KEY

    transcriber = aai.Transcriber()

    trans = transcriber.transcribe(audio_bytes)
    return  trans.text

def mfcc(file_path):
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

# def mfcc(preview_url):
#     #try:
#         y, sr = librosa.load(preview_url)  # sr=None giữ nguyên sample rate gốc
#         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#         delta1 = librosa.feature.delta(mfccs, order=1)
#         delta2 = librosa.feature.delta(mfccs, order=2)
#         combined_mfcc = np.concatenate((mfccs, delta1, delta2), axis=0)
#         # Chuyển về (T, 39) để tính DTW
#         vec1 = combined_mfcc.T  # shape: (T1, 39)
#
#         arr_list = vec1.tolist()
#         arr_str = json.dumps(arr_list)
#         #print(arr_str)
#         return arr_str
#     # except Exception as e:
#     #     print("Lỗi khi đọc âm thanh hoặc tính MFCC:", e)
#     #     return None

def energy_contour(file_path):
    # Tham số
    frame_length = 2048
    hop_length = 512

    y, sr = librosa.load(file_path)
    # Trích xuất năng lượng từng khung
    energy = np.array([
        np.sum(np.abs(y[i:i + frame_length]) ** 2)
        for i in range(0, len(y) - frame_length, hop_length)
    ])
    # Loại bỏ NaN và 0
    energy = energy[~np.isnan(energy)]
    energy = energy[energy != 0]

    energy1_2d = energy.reshape(1, -1)
    arr_list = energy1_2d.tolist()
    arr_str = json.dumps(arr_list)
    # print(arr_str)
    return arr_str

def formant_f1_f2(file_path,response):
    try:
    # Ghi vào file tạm thời
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
          tmp.write(response.content)
          tmp_path = tmp.name

        # ✅ Trích xuất formants (F1 và F2) từ file1
        snd = parselmouth.Sound(tmp_path)
        formant = snd.to_formant_burg()
        f1_1, f2_1 = [], []
        for t in np.arange(0, snd.duration, 0.01):
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            if not np.isnan(f1) and not np.isnan(f2):
                f1_1.append(f1)
                f2_1.append(f2)
        f1_1 = np.array(f1_1)
        f2_1 = np.array(f2_1)
        # ✅ Ghép F1 và F2 thành vector đặc trưng
        min_len1 = min(len(f1_1), len(f2_1))

        vec = np.column_stack((f1_1[:min_len1], f2_1[:min_len1]))
        # ✅ Tính DTW
        vec_T = vec.T
        arr_list = vec_T.tolist()
        arr_str = json.dumps(arr_list)
        # print(arr_str)
        return arr_str

    except Exception as e:
         print("Lỗi khi đọc âm thanh hoặc tính:", e)
         return None
    # finally:
    #     os.remove(tmp_path)

def spactral_centroid(file_path):
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

# def spactral_centroid(file_path):
#
#     y, sr = librosa.load(file_path)
#     centroid = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()  # Tính spectral centroid
#     # Xử lý NaN (nếu có)
#     centroid = centroid[~np.isnan(centroid)]
#     # Đảm bảo centroid là một mảng 1D
#     centroid = centroid.reshape(-1, 1)
#     arr_list = centroid.tolist()
#     arr_str = json.dumps(arr_list)
#     # print(arr_str)
#     return arr_str

def bandwind(file_path):
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

# def bandwind(file_path):
#     y, sr = librosa.load(file_path)
#     # Tính Spectral Bandwidth
#     bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).flatten()  # Chuyển thành vector 1-D
#     # Xử lý NaN (nếu có)
#     bandwidth = bandwidth[~np.isnan(bandwidth)]
#     # Đảm bảo centroid là một mảng 1D
#     bandwidth = bandwidth.reshape(-1, 1)  # Chuyển đổi thành mảng 2D nếu cần, sau đó flatten nó
#     arr_list = bandwidth.tolist()
#     arr_str = json.dumps(arr_list)
#     # print(arr_str)
#     return arr_str


def roof_off(file_path):
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

# def roof_off(file_path):
#     y, sr = librosa.load(file_path)
#     # Tính Spectral Roll-offa
#     spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
#     # Đảm bảo dữ liệu là 1D trước khi tính DTW
#     spectral_rolloff_flat = spectral_rolloff.flatten()
#     # Đảm bảo centroid là một mảng 1D
#     spectral_rolloff_flat = spectral_rolloff_flat.reshape(-1, 1)
#     arr_list = spectral_rolloff_flat.tolist()
#     arr_str = json.dumps(arr_list)
#     # print(arr_str)
#     return arr_str


# x = mfcc(audio_bytes)
# print(x)
# print(type(x))
# print('sss')
if __name__ == "__main__":
    file_path = "https://www.dropbox.com/scl/fi/nzc7kgx45vln8ibroxm8e/video_30_chunk_3.wav?rlkey=8f1ke2ak2dsjczxw9q1a0tctb&dl=0"
    file_path2 = "https://www.dropbox.com/scl/fi/bvyaqqtx3uiegqauselvd/video_30_chunk_2.wav?rlkey=deikeuo3mxj0k7hxcn1cuacv2&dl=0"
    # Chuyển link preview thành link tải file nhị phân trực tiếp
    direct_url = file_path.replace("dl=0", "dl=1")
    direct_url2 = file_path2.replace("dl=0", "dl=1")

    response = requests.get(direct_url)
    response2 = requests.get(direct_url2)

    audio_bytes = io.BytesIO(response.content)
    audio_bytes2 = io.BytesIO(response2.content)

    x = mfcc(audio_bytes)
    x2 = mfcc(audio_bytes2)

    d1 = json.loads(x)
    d2 = json.loads(x2)
    print(x)
    print(len(x))



