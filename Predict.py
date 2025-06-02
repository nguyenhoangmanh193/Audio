import librosa.sequence
import pandas as pd
import numpy as np
import  json
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from librosa.sequence import dtw
def dtw(d1,d2):
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
        ds = dtw(d1, arr)
        ds1 = dtw2(d1_b, arr2)
        ds_sum = (ds/25 + ds1)/2
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



df = pd.read_csv("Data/audio_data.csv")

# list = dtw_ngudieu()
# print(list)

#list_mfccs = df['mfccs'].tolist()

# print(df['file_name'][0])
# # Chuyển string về list bình thường
# list_data = json.loads(list_mfccs[76])
#
#
# # Chuyển list thành numpy array
# arr = np.array(list_data)
#
# results = {}

# for i in range(len(list_mfccs)):
#     d1 = json.loads(list_mfccs[i])
#     d1 = np.array(d1)
#     ds = dtw(d1, arr)
#     print(f"{df['file_name'][i]}: {ds:.4f}")
#     score = ds
#     file_name = df['file_name'][i]
#
#     results[file_name] = round(score, 4)
# # recommened.sort()
# # for i in range(0,10):
# #     print(recommened[i])
#
#
# # Sắp xếp theo giá trị tăng dần
# sorted_results = dict(sorted(results.items(), key=lambda item: item[1]))
#
# top_10 = list(sorted_results.items())[:10]
# for idx, (key, value) in enumerate(top_10, 1):
#     key = key.split("/")[-1].split("?")[0]
#     print(f"{idx}. {key}: {value}")