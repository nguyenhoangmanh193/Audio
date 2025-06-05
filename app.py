import streamlit as st
from sqlalchemy.sql.functions import min
import io
import requests
import numpy as np
import parselmouth
import librosa.sequence
import matplotlib.pyplot as plt
import  json
from  Predict import dtw_ngudieu, dtw_phatam
import  pandas as pd
from  dactrung import  mfcc, energy_contour, formant_f1_f2, spactral_centroid, bandwind, roof_off,text_convert

def convert_dropbox_link_to_direct(url):
    if "dropbox.com" in url and "dl=0" in url:
        url = url.replace("www.dropbox.com", "www.dl.dropboxusercontent.com")
        url = url.replace("?dl=0", "")
    return url

def plot_mfcc_streamlit(arr_str):
    # Chuyển JSON string thành numpy array
    arr_list = json.loads(arr_str)
    mfcc_array = np.array(arr_list).T  # chuyển về (39, T)

    # Vẽ biểu đồ MFCC
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfcc_array, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC')
    plt.tight_layout()

    # Hiển thị lên Streamlit
    st.pyplot(fig)

def plot_energy_streamlit(arr_str):
    # Giải mã chuỗi JSON -> mảng numpy
    arr_list = json.loads(arr_str)
    energy_array = np.array(arr_list).flatten()

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(energy_array, color='orange')
    ax.set_title('Biểu đồ Năng lượng (Energy Contour)')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Energy')

    # Hiển thị trên Streamlit
    st.pyplot(fig)

def plot_formants_streamlit(arr_str, step=0.01):
    """
    Vẽ đồ thị formant F1 và F2 theo thời gian từ chuỗi JSON arr_str.

    Parameters:
    - arr_str (str): Chuỗi JSON chứa mảng [[F1, F2], [F1, F2], ...]
    - step (float): Khoảng thời gian giữa các mẫu (mặc định 0.01s)

    Hiển thị đồ thị trực tiếp bằng Streamlit.
    """
    try:
        # Giải mã JSON về mảng numpy shape (N, 2)
        arr1 = np.array(json.loads(arr_str)).T  # shape (N,2)


        f1_1, f2_1 = arr1[:, 0], arr1[:, 1]


        # Vẽ scatter
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(f1_1, f2_1, alpha=0.5, label="Audio 1", color="blue", s=10)


        ax.set_xlabel("Formant 1 (F1) [Hz]")
        ax.set_ylabel("Formant 2 (F2) [Hz]")
        ax.set_title("Formant F1 và F2 ")
        ax.legend()
        ax.grid(True)

        # Hiển thị trên Streamlit
        st.subheader("Biểu đồ scatter so sánh F1–F2 giữa hai audio")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Lỗi khi vẽ scatter formant: {e}")



st.set_page_config(page_title="Phân tích giọng nói", layout="wide")

st.title("🎙️ Hệ thống phân tích giọng nói")

# Sidebar menu
page = st.sidebar.radio(
    "📌 Chọn chức năng",
    ["📝 Nội dung ngữ nghĩa", "📈 Ngữ điệu", "🗣️ Kiểu phát âm", "🎤 Giọng nói"]
)

df = pd.read_csv("Data/audio_data.csv")
if page == "📝 Nội dung ngữ nghĩa":
        st.subheader("📝 Phân tích Nội dung Ngữ nghĩa")
        if st.button("Phân tích ngữ nghĩa"):
            # Giả định xử lý chuyển giọng nói thành văn bản
            st.success("✅ Kết quả phân tích nội dung:")
            st.markdown("""
            - **Văn bản trích xuất**: "Xin chào, tôi là trợ lý ảo."
            - **Chủ đề**: Giao tiếp
            - **Cảm xúc**: Trung lập
            - **Mức độ trang trọng**: Trung bình
            """)



elif page == "📈 Ngữ điệu":
    st.subheader("📈 Phân tích Ngữ điệu")
    uploaded_file = st.file_uploader("🎵 Tải lên file âm thanh", type=["wav", "mp3", "m4a"])

    if uploaded_file is not None:
        uploaded_file.seek(0)
        mfccs = mfcc(uploaded_file)

        uploaded_file.seek(0)
        energy = energy_contour(uploaded_file)

        # Lấy link từ hàm dtw và chuyển về dạng direct
        raw_links = dtw_ngudieu(mfccs, energy)
        audio_links = [convert_dropbox_link_to_direct(link) for link in raw_links]

        cols = st.columns(4)

        with cols[0]:
            st.markdown("**🔊 File của bạn**")
            st.audio(uploaded_file, format="audio/wav")
            plot_mfcc_streamlit(mfccs)
            plot_energy_streamlit(energy)

        for i in range(3):
            with cols[i + 1]:
                #mfcc_value = df.loc[df['file_name'] == audio_links[i], 'mfccs'].values[0]
                #energy_value = df.loc[df['file_name'] == audio_links[i], 'energy'].values[0]
                key = audio_links[i].split("/")[-1].split("?")[0]
                mfcc_value = df.loc[df['file_name'].str.contains(key), 'mfccs'].values[0]
                energy_value = df.loc[df['file_name'].str.contains(key), 'energy'].values[0]
                st.markdown(f"**🔁File giống thứ {i+1}:  {key}**")
                st.audio(audio_links[i], format="audio/wav")
                plot_mfcc_streamlit(mfcc_value)
                plot_energy_streamlit(energy_value)


    else:
        st.info("📂 Vui lòng tải lên một file âm thanh.")


elif page == "🗣️ Kiểu phát âm":
    st.subheader("📈 Phân tích kiểu phát âm")
    uploaded_file = st.file_uploader("🎵 Tải lên file âm thanh", type=["wav", "mp3", "m4a"])

    if uploaded_file is not None:
        uploaded_file.seek(0)
        mfccs = mfcc(uploaded_file)

        # Tính formant
        uploaded_file.seek(0)
        # ✅ Trích xuất formants (F1 và F2) từ file1
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.read())

        snd = parselmouth.Sound("temp_audio.wav")
        formant = snd.to_formant_burg()
        f1_1, f2_1 = [], []
        for t in np.arange(0, snd.duration, 0.01):
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            if not np.isnan(f1) and not np.isnan(f2):
                f1_1.append(f1)
                f2_1.append(f2)

        # ✅ Ghép F1 và F2 thành vector đặc trưng
        if len(f1_1) < len(f2_1):
            min_len1 = len(f1_1)
        else:
            min_len1 = len(f2_1)

        f1_1 = np.array(f1_1)
        f2_1 = np.array(f2_1)

        vec = np.column_stack((f1_1[:min_len1], f2_1[:min_len1]))
        # # ✅ Tính DTW
        vec_T = vec.T
        arr_list = vec_T.tolist()
        arr_str = json.dumps(arr_list)

        # End
        formant = arr_str

        # Lấy link từ hàm dtw và chuyển về dạng direct
        raw_links = dtw_phatam(mfccs, formant)
        audio_links = [convert_dropbox_link_to_direct(link) for link in raw_links]

        cols = st.columns(4)

        with cols[0]:
            st.markdown("**🔊 File của bạn**")
            st.audio(uploaded_file, format="audio/wav")
            plot_mfcc_streamlit(mfccs)
            plot_formants_streamlit(formant)

        for i in range(3):
            with cols[i + 1]:
                # mfcc_value = df.loc[df['file_name'] == audio_links[i], 'mfccs'].values[0]
                # energy_value = df.loc[df['file_name'] == audio_links[i], 'energy'].values[0]
                key = audio_links[i].split("/")[-1].split("?")[0]
                mfcc_value = df.loc[df['file_name'].str.contains(key), 'mfccs'].values[0]
                formant_value = df.loc[df['file_name'].str.contains(key), 'formant'].values[0]
                st.markdown(f"**🔁File giống thứ {i + 1}:  {key}**")
                st.audio(audio_links[i], format="audio/wav")
                plot_mfcc_streamlit(mfcc_value)
                plot_formants_streamlit(formant_value)


    else:
        st.info("📂 Vui lòng tải lên một file âm thanh.")

elif page == "🎤 Giọng nói":
        st.subheader("🎤 Nhận dạng Giọng nói")
        if st.button("Nhận dạng giọng nói"):
            st.success("✅ Kết quả giọng nói:")
            st.markdown("""
            - **Giới tính**: Nữ
            - **Vùng miền**: Miền Trung
            - **Ước lượng tuổi**: 20-30 tuổi
            """)

