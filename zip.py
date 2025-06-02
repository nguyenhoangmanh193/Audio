import requests
import io
import json
from api import api_key
import assemblyai as aai

API_KEY = api_key()
# Đọc lại từ file
with open("Data/my_list.json", "r") as f:
    loaded_list = json.load(f)

# Chuyển link preview thành link tải file nhị phân trực tiếp
direct_url = loaded_list[0].replace("dl=0", "dl=1")
response = requests.get(direct_url)
if response.status_code != 200:
        print("Lỗi tải file:", response.status_code)
audio_bytes = io.BytesIO(response.content)




aai.settings.api_key= API_KEY

transcriber = aai.Transcriber()

trans = transcriber.transcribe(audio_bytes)

text = trans.text
max_chars = 50

for i in range(0, len(text), max_chars):
    print(text[i:i+max_chars])