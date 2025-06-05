import  pandas as pd
df = pd.read_csv("Data/audio_data.csv")
# Lọc bỏ các hàng mà file_name chứa bất kỳ chuỗi nào trong danh sách
patterns = "|".join([
    "video_2_chunk_3.wav",
    "video_3_chunk_3.wav",
    "video_4_chunk_3.wav",
    "video_5_chunk_3.wav",
    "video_6_chunk_3.wav"
])

df = df[~df['file_name'].str.contains(patterns)]
df = df.reset_index(drop=True)
print(df.head(45))