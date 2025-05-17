import mongoengine as me
import requests
import librosa
import io
import os
import numpy as np
import matplotlib.pyplot as plt
import parselmouth
import tempfile
import json
from  dactrung import  mfcc, energy_contour, formant_f1_f2, spactral_centroid, bandwind, roof_off, text_convert

# Kết nối MongoDB với URI
me.connect(
    db="audio",
    host="mongodb+srv://onmontoan:biahoibuncha8862@cluster0.mhbsw.mongodb.net/audio?retryWrites=true&w=majority"
)

# Định nghĩa schema/model
class AudioFeature(me.Document):
    file_name = me.StringField(required=True)
    text = me.StringField()# Tên file audio
    mfccs = me.StringField()                        # Chuỗi dữ liệu mô tả hoặc JSON string (tùy bạn lưu gì)
    energy = me.StringField()
    formant = me.StringField()
    spec = me.StringField()
    bandw = me.StringField()
    roof = me.StringField()
    meta = {
        'collection': 'audio_data'  # Tên collection trong MongoDB
    }

# Ví dụ tạo document mới
if __name__ == "__main__":
    dropbox_preview_link = "https://www.dropbox.com/scl/fi/phsclrp8qmqxkpps9jguu/video_59_chunk_3.wav?rlkey=95pt94yklx7a0dxd6m9shm7cz&st=4s20ubcd&dl=0"

    # Chuyển link preview thành link tải file nhị phân trực tiếp
    direct_url = dropbox_preview_link.replace("dl=0", "dl=1")
    response = requests.get(direct_url)
    if response.status_code != 200:
        print("Lỗi tải file:", response.status_code)
    audio_bytes = io.BytesIO(response.content)
    t = text_convert(audio_bytes)
    audio_bytes = io.BytesIO(response.content)
    m = mfcc(audio_bytes)
    audio_bytes = io.BytesIO(response.content)
    e = energy_contour(audio_bytes)
    audio_bytes = io.BytesIO(response.content)
    formant = formant_f1_f2(audio_bytes,response)

    audio_bytes = io.BytesIO(response.content)

    s = spactral_centroid(audio_bytes)
    audio_bytes = io.BytesIO(response.content)
    band = bandwind(audio_bytes)
    audio_bytes = io.BytesIO(response.content)
    r = roof_off(audio_bytes)


    feature = AudioFeature(file_name=dropbox_preview_link,text= t, mfccs=m, energy = e, formant=formant,spec= s, bandw= band, roof =r)
    feature.save()
    print("Đã lưu document với id:", feature.id)
