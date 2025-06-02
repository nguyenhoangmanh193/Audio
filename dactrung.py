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
def mfcc(preview_url):
    #try:
        y, sr = librosa.load(preview_url)  # sr=None giữ nguyên sample rate gốc
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta1 = librosa.feature.delta(mfccs, order=1)
        delta2 = librosa.feature.delta(mfccs, order=2)
        combined_mfcc = np.concatenate((mfccs, delta1, delta2), axis=0)
        # Chuyển về (T, 39) để tính DTW
        vec1 = combined_mfcc.T  # shape: (T1, 39)

        arr_list = vec1.tolist()
        arr_str = json.dumps(arr_list)
        #print(arr_str)
        return arr_str
    # except Exception as e:
    #     print("Lỗi khi đọc âm thanh hoặc tính MFCC:", e)
    #     return None

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
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).flatten()  # Tính spectral centroid
    # Xử lý NaN (nếu có)
    centroid = centroid[~np.isnan(centroid)]
    # Đảm bảo centroid là một mảng 1D
    centroid = centroid.reshape(-1, 1)
    arr_list = centroid.tolist()
    arr_str = json.dumps(arr_list)
    # print(arr_str)
    return arr_str


def bandwind(file_path):
    y, sr = librosa.load(file_path)
    # Tính Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).flatten()  # Chuyển thành vector 1-D
    # Xử lý NaN (nếu có)
    bandwidth = bandwidth[~np.isnan(bandwidth)]
    # Đảm bảo centroid là một mảng 1D
    bandwidth = bandwidth.reshape(-1, 1)  # Chuyển đổi thành mảng 2D nếu cần, sau đó flatten nó
    arr_list = bandwidth.tolist()
    arr_str = json.dumps(arr_list)
    # print(arr_str)
    return arr_str


def roof_off(file_path):
    y, sr = librosa.load(file_path)
    # Tính Spectral Roll-offa
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    # Đảm bảo dữ liệu là 1D trước khi tính DTW
    spectral_rolloff_flat = spectral_rolloff.flatten()
    # Đảm bảo centroid là một mảng 1D
    spectral_rolloff_flat = spectral_rolloff_flat.reshape(-1, 1)
    arr_list = spectral_rolloff_flat.tolist()
    arr_str = json.dumps(arr_list)
    # print(arr_str)
    return arr_str


# x = mfcc(audio_bytes)
# print(x)
# print(type(x))
# print('sss')
if __name__ == "__main__":
    # chỉ chạy khi module này được chạy trực tiếp
    print("Tôi đang ở module_a.py")


