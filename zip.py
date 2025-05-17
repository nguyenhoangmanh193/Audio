import zipfile
import pandas as pd
# csv_path = "Data/audio_data.csv"
zip_path = "Data/audio_data.zip"
#
# with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
#     zipf.write(csv_path, arcname='audio_data.csv')
#df = pd.read_csv("Data/audio_data.csv")

# zip_path = "Data/audio_data.zip"
#
# Mở file zip
with zipfile.ZipFile(zip_path) as z:
    # Danh sách file bên trong zip (lấy tên file .csv)
    csv_filename = [f for f in z.namelist() if f.endswith('.csv')][0]

    # Đọc trực tiếp file CSV bên trong zip thành DataFrame
    with z.open(csv_filename) as f:
        df = pd.read_csv(f)

df.to_csv('Data/audio_data.csv.gz', index=False, compression='gzip')
# Kiểm tra kết quả
print(df.head())