import json
import io
import mongoengine as me
import requests
from  dactrung import  mfcc, energy_contour, formant_f1_f2, spactral_centroid, bandwind, roof_off,text_convert
import time
# Kết nối MongoDB với URI
me.connect(
    db="audio",
    host="mongodb+srv://onmontoan:biahoibuncha8862@cluster0.mhbsw.mongodb.net/audio?retryWrites=true&w=majority"
)

# Định nghĩa schema/model
class AudioFeature(me.Document):
    file_name = me.StringField(required=True, unique=True)
    text = me.StringField(required=True)# Tên file audio
    mfccs = me.StringField(required=True)                        # Chuỗi dữ liệu mô tả hoặc JSON string (tùy bạn lưu gì)
    energy = me.StringField(required=True)
    formant = me.StringField(required=True)
    spec = me.StringField(required=True)
    bandw = me.StringField(required=True)
    roof = me.StringField(required=True)
    meta = {
        'collection': 'audio_data'  # Tên collection trong MongoDB
}
# Đọc lại từ file
with open("Data/my_list.json", "r") as f:
    loaded_list = json.load(f)


q=0
for i in range(0,len(loaded_list)):
       try:
           if AudioFeature.objects(file_name=loaded_list[i]).first():
               print(f"File '{loaded_list[i]}' đã tồn tại, bỏ qua.")
               continue
           dropbox_preview_link = loaded_list[i]

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
           audio_bytes = io.BytesIO(response.content)

           feature = AudioFeature(file_name=dropbox_preview_link,text= t, mfccs=m, energy=e, formant=formant, spec=s,
                                  bandw=band, roof=r)
           feature.save()
           print("Đã lưu document với id:", feature.file_name)
           print(q)



       except Exception as e:
           print(e)
           continue

       q += 1
