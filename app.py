import  pandas as pd

link = 'https://www.dropbox.com/scl/fi/f86hv2f0akalkht1ylkc0/video_52_chunk_2.wav?rlkey=95egtsr54578987pz27haj4yc&dl=0';

df = pd.read_csv("Data/audio_data.csv")
mfcc_value = df.loc[df['file_name'] == link, 'mfccs'].values[0]

print(mfcc_value)