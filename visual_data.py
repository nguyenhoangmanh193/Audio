import  pandas as pd
import numpy as np
import  json
df = pd.read_csv("Data/audio_data.csv")
arr= df.columns.tolist()
print("Danh sách thuộc tính: "+ str(arr))
for i in range(2,7):
    value =df.iloc[0,i]
    print("Độ dài mảng của thuộc tính "+ arr[i] + " là: " + str(len(value)) )
#         a
