import librosa.sequence
import pandas as pd
import numpy as np
import  json
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from librosa.sequence import dtw

df = pd.read_csv("Data/audio_data.csv")

# Lọc bỏ các file_name bị lỗi ( quá ngắn )
patterns = "|".join([
    "video_2_chunk_3.wav",
    "video_3_chunk_3.wav",
    "video_4_chunk_3.wav",
    "video_5_chunk_3.wav",
    "video_6_chunk_3.wav"
])

df = df[~df['file_name'].str.contains(patterns)]
df = df.reset_index(drop=True)
def dtw(d1,d2):
    distance, path = fastdtw(d1, d2, dist=euclidean)
    return  distance


# def dtw(x, y, window=50):
#     x = np.array(x).reshape(-1, 1)
#     y = np.array(y).reshape(-1, 1)
#     N = len(x)
#     M = len(y)
#     window = max(window, abs(N - M))  # đảm bảo không bị vượt chiều dài
#
#     D = np.full((N, M), np.inf)
#     for i in range(N):
#         for j in range(max(0, i - window), min(M, i + window)):
#             D[i, j] = np.linalg.norm(x[i] - y[j])
#
#     cost = np.full((N, M), np.inf)
#     cost[0, 0] = D[0, 0]
#
#     for i in range(1, N):
#         cost[i, 0] = D[i, 0] + cost[i - 1, 0]
#     for j in range(1, M):
#         cost[0, j] = D[0, j] + cost[0, j - 1]
#
#     for i in range(1, N):
#         for j in range(max(1, i - window), min(M, i + window)):
#             cost[i, j] = D[i, j] + min(
#                 cost[i - 1, j],      # insertion
#                 cost[i, j - 1],      # deletion
#                 cost[i - 1, j - 1]   # match
#             )
#
#     return float(cost[-1, -1])

def dtw_library(d1,d2):
    distance, path = fastdtw(d1, d2, dist=euclidean)
    return  distance

def dtw2(d1,d2):
    D, path = librosa.sequence.dtw(d1, d2, metric='euclidean')
    distance = D[-1, -1]
    return  float(distance)

def dtw_ngudieu(mfccs, energy):
    # mfcc, rmse
    results = {}
    list_mfccs = df['mfccs'].tolist();
    list_energy = df['energy'].tolist();
    list_data = json.loads(mfccs) # chọn 1 file
    list_data2 = json.loads(energy)  # chọn 1 file
    # Chuyển list thành numpy array
    arr = np.array(list_data)
    arr2 = np.array(list_data2)
    for i in range(len(list_mfccs)):
        d1 = json.loads(list_mfccs[i])
        d1_b = json.loads(list_energy[i])

        d1 = np.array(d1)
        d1_b = np.array(d1_b)
        ds = dtw_library(d1, arr)
        ds1 = dtw2(d1_b, arr2)
        ds_sum = (ds + ds1)/2
        print(f"{df['file_name'][i]}: mfcc:{ds:.4f}, energy:{ds1:.4f}, sum:{ds_sum:.4f} ")
        score = ds_sum
        file_name = df['file_name'][i]

        results[file_name] = round(score, 4)

    # Sắp xếp theo giá trị tăng dần
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1]))

    list_audio = []
    top_3 = list(sorted_results.items())[:3]
    for idx, (key, value) in enumerate(top_3, 1):
        list_audio.append(key)
        key = key.split("/")[-1].split("?")[0]
        print(f"{idx}. {key}: {value}")
    return list_audio

def dtw_phatam(mfccs, formant):
    # mfcc, rmse
    results = {}
    list_mfccs = df['mfccs'].tolist();
    list_formant = df['formant'].tolist();

    list_data = json.loads(mfccs) # chọn 1 file
    list_data2 = json.loads(formant)  # chọn 1 file
    # Chuyển list thành numpy array
    arr = np.array(list_data)
    arr2 = np.array(list_data2)
    for i in range(len(list_mfccs)):
        d1 = json.loads(list_mfccs[i])
        d1_b = json.loads(list_formant[i])

        d1 = np.array(d1)
        d1_b = np.array(d1_b)
        ds = dtw_library(d1, arr)
        ds1 = dtw2(d1_b, arr2)
        ds_sum = (ds + ds1/100)
        print(f"{df['file_name'][i]}: mfcc:{ds:.4f}, formant:{ds1:.4f}, sum:{ds_sum:.4f} ")
        score = ds_sum
        file_name = df['file_name'][i]

        results[file_name] = round(score, 4)

    # Sắp xếp theo giá trị tăng dần
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1]))

    list_audio = []
    top_3 = list(sorted_results.items())[:3]
    for idx, (key, value) in enumerate(top_3, 1):
        list_audio.append(key)
        key = key.split("/")[-1].split("?")[0]
        print(f"{idx}. {key}: {value}")
    return list_audio


def dtw_giongnoi(spec, bandw, roof):
    # mfcc, rmse
    results = {}
    list_spec = df['spec'].tolist()
    list_bandw = df['bandw'].tolist()
    list_roof = df['roof'].tolist()
    list_data = json.loads(spec) # chọn 1 file
    list_data1 = json.loads(bandw)  # chọn 1 file
    list_data2 = json.loads(roof)
    # Chuyển list thành numpy array
    arr = np.array(list_data)
    arr1 = np.array(list_data1)
    arr2 = np.array(list_data2)
    for i in range(len(list_spec)):
        d1 = json.loads(list_spec[i])
        d1_a = json.loads(list_bandw[i])
        d1_b = json.loads(list_roof[i])

        d1 = np.array(d1)
        d1_a = np.array(d1_a)
        d1_b = np.array(d1_b)
        ds = dtw(d1, arr)
        ds1 = dtw(d1_a, arr1)
        ds2 = dtw(d1_b, arr2)

        ds_sum = (ds/10 + ds1+ds2/10)/2
        print(f"{df['file_name'][i]}: spec:{ds:.4f}, bandw:{ds1:.4f}, roof:{ds2:.4f} ,sum:{ds_sum:.4f} ")
        score = ds_sum
        file_name = df['file_name'][i]

        results[file_name] = round(score, 4)

    # Sắp xếp theo giá trị tăng dần
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1]))

    list_audio = []
    top_3 = list(sorted_results.items())[:3]
    for idx, (key, value) in enumerate(top_3, 1):
        list_audio.append(key)
        key = key.split("/")[-1].split("?")[0]
        print(f"{idx}. {key}: {value}")
    return list_audio



if __name__ == "__main__":
    df = pd.read_csv("Data/audio_data.csv")

    # list = dtw_ngudieu()
    # print(list)

    list_mfccs = df['formant'].tolist()

    print(df['file_name'][0])
    # Chuyển string về list bình thường
    list_data = json.loads(list_mfccs[76])

    # Chuyển list thành numpy array
    arr = np.array(list_data)

    results = {}

    for i in range(len(list_mfccs)):
        d1 = json.loads(list_mfccs[i])
        d1 = np.array(d1)
        ds = dtw2(d1, arr)
        print(f"{df['file_name'][i]}: {ds:.4f}")
        score = ds
        file_name = df['file_name'][i]

        results[file_name] = round(score, 4)
    # recommened.sort()
    # for i in range(0,10):
    #     print(recommened[i])

    # Sắp xếp theo giá trị tăng dần
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1]))

    top_10 = list(sorted_results.items())[:10]
    for idx, (key, value) in enumerate(top_10, 1):
        key = key.split("/")[-1].split("?")[0]
        print(f"{idx}. {key}: {value}")