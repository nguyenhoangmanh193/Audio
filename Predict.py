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


df = pd.read_csv("Data/audio_data.csv")

list_mfccs = df['formant'].tolist()
# print(df['file_name'][0])
# # Chuyển string về list bình thường
list_data = json.loads(list_mfccs[0])


# Chuyển list thành numpy array
#arr = np.array(list_data)

# for i in range(0,len(list_mfccs)):
#     d1 = json.loads(list_mfccs[i])
#     d1 = np.array((d1))
#
#     print(f"{df['file_name'][i]}: {dtw2(d1, arr):.4f}")
# recommened.sort()
# for i in range(0,10):
#     print(recommened[i])


print(len(df))