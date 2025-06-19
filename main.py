import streamlit as st
from sqlalchemy.sql.functions import min
import io
import requests
import numpy as np
import parselmouth
import librosa.sequence
import matplotlib.pyplot as plt
import  json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from  Predict import dtw_ngudieu, dtw_phatam, dtw_giongnoi
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

def plot_spectral_centroid(arr_str):

    try:
        # Tham số mặc định
        sr = 22050
        hop_length = 512

        # Giải mã chuỗi JSON
        centroid = np.array(json.loads(arr_str)).flatten()

        # Tính trục thời gian
        frames = np.arange(len(centroid))
        times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

        # Vẽ biểu đồ
        fig, ax = plt.subplots()
        ax.plot(times, centroid, color='purple', label='Spectral Centroid')
        ax.set_xlabel("Thời gian (s)")
        ax.set_ylabel("Tần số (Hz)")
        ax.set_title("Spectral Centroid theo thời gian")
        ax.grid(True)
        ax.legend()

        # Hiển thị trên Streamlit
        st.subheader("🎵 Spectral Centroid theo thời gian")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Lỗi khi vẽ spectral centroid: {e}")


def plot_spectral_bandwidth(arr_str):

    try:
        # Tham số mặc định
        sr = 22050
        hop_length = 512

        # Giải mã JSON
        bandwidth = np.array(json.loads(arr_str)).flatten()

        # Tạo trục thời gian tương ứng
        frames = np.arange(len(bandwidth))
        times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

        # Vẽ biểu đồ
        fig, ax = plt.subplots()
        ax.plot(times, bandwidth, color='green', label='Spectral Bandwidth')
        ax.set_xlabel("Thời gian (s)")
        ax.set_ylabel("Tần số (Hz)")
        ax.set_title("Spectral Bandwidth theo thời gian")
        ax.grid(True)
        ax.legend()

        # Hiển thị bằng Streamlit
        st.subheader("🎶 Spectral Bandwidth theo thời gian")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Lỗi khi vẽ spectral bandwidth: {e}")


def plot_spectral_rolloff(arr_str):

    try:
        # Mặc định librosa
        sr = 22050
        hop_length = 512

        # Giải mã JSON
        rolloff = np.array(json.loads(arr_str)).flatten()

        # Tính trục thời gian
        frames = np.arange(len(rolloff))
        times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

        # Vẽ biểu đồ
        fig, ax = plt.subplots()
        ax.plot(times, rolloff, color='orange', label='Spectral Roll-off')
        ax.set_xlabel("Thời gian (s)")
        ax.set_ylabel("Tần số (Hz)")
        ax.set_title("Spectral Roll-off theo thời gian")
        ax.grid(True)
        ax.legend()

        # Hiển thị trên Streamlit
        st.subheader("📉 Spectral Roll-off theo thời gian")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Lỗi khi vẽ spectral roll-off: {e}")

def cosine_similarity_manual(a, b):
    a = np.array(a)
    b = np.array(b)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b + 1e-10)  # thêm epsilon tránh chia 0

# def compute_similarities(text_main, text_list):
#     model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#     embeddings = model.encode([text_main] + text_list)
#     main_vec = embeddings[0].reshape(1, -1)
#     list_vecs = embeddings[1:]
#     similarities = cosine_similarity(main_vec, list_vecs).flatten()
#     return similarities

def compute_similarities(text_main, text_list):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode([text_main] + text_list)
    main_vec = embeddings[0]
    similarities = []

    for vec in embeddings[1:]:
        sim = cosine_similarity_manual(main_vec, vec)
        similarities.append(sim)

    return np.array(similarities)

st.set_page_config(page_title="Phân tích giọng nói", layout="wide")

st.title("🎙️ Hệ thống phân tích giọng nói")

# Sidebar menu
page = st.sidebar.radio(
    "📌 Chọn chức năng tìm kiếm",
    ["📝 Nội dung ngữ nghĩa", "📈 Ngữ điệu", "🗣️ Kiểu phát âm", "🎤 Giọng nói"]
)

df = pd.read_csv("Data/audio_data.csv")
# Lọc bỏ các file_name bị lỗi ( quá ngắn )
patterns = "|".join([
    "video_2_chunk_3.wav",
    "video_3_chunk_3.wav",
    "video_4_chunk_3.wav",
    "video_5_chunk_3.wav",
    "video_6_chunk_3.wav"
])

df = df[~df['file_name'].str.contains(patterns)]
df = df.reset_index(drop=True)

if page == "📝 Nội dung ngữ nghĩa":
        st.subheader("📝 Tìm kiếm với nội dung ngữ nghĩa")
        uploaded_file = st.file_uploader("🎵 Tải lên file âm thanh", type=["wav", "mp3", "m4a"])
        if uploaded_file is not None:
            text_main = text_convert(uploaded_file)

            # Lấy danh sách text và tính độ tương đồng
            text_list = df['text'].tolist()
            file_names = df['file_name'].tolist()
            similarities = compute_similarities(text_main, text_list)

            # Gắn vào DataFrame tạm để sắp xếp
            result_df = df.copy()
            result_df['similarity'] = similarities
            top_3 = result_df.sort_values(by='similarity', ascending=False).head(3)

            st.markdown("### 🔍 Top 3 kết quả giống nhất:")

            for idx, row in top_3.iterrows():
                # Chuyển link Dropbox sang dạng trực tiếp
                dropbox_url = row['file_name']
                audio_url= convert_dropbox_link_to_direct(dropbox_url)
                key = audio_url.split("/")[-1].split("?")[0]
                # Hiển thị thông tin
                st.write(f"🎧 **File:** `{key}`")
                st.audio(audio_url, format="audio/wav")  # bạn có thể thay format nếu file là mp3, m4a...
                st.write(f"📝 **Nội dung:** {row['text']}")
                st.write(f"🔍 **Mức độ tương đồng:** `{row['similarity']:.2f}`")
                st.markdown("---")



elif page == "📈 Ngữ điệu":
    st.subheader("📈 Tìm kiếm với ngữ điệu")
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
    st.subheader("📈 Tìm kiếm với kiểu phát âm")
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
    st.subheader("📈 Tìm kiếm với giọng nói")
    uploaded_file = st.file_uploader("🎵 Tải lên file âm thanh", type=["wav", "mp3", "m4a"])

    if uploaded_file is not None:
        uploaded_file.seek(0)
        spec = spactral_centroid(uploaded_file)

        uploaded_file.seek(0)
        banw = bandwind(uploaded_file)

        uploaded_file.seek(0)
        roof = roof_off(uploaded_file)

        # Lấy link từ hàm dtw và chuyển về dạng direct
        raw_links = dtw_giongnoi(spec, banw,roof)
        audio_links = [convert_dropbox_link_to_direct(link) for link in raw_links]

        cols = st.columns(4)

        with cols[0]:
            st.markdown("**🔊 File của bạn**")
            st.audio(uploaded_file, format="audio/wav")
            plot_spectral_centroid(spec)
            plot_spectral_bandwidth(banw)
            plot_spectral_rolloff(roof)

        for i in range(3):
            with cols[i + 1]:
                # mfcc_value = df.loc[df['file_name'] == audio_links[i], 'mfccs'].values[0]
                # energy_value = df.loc[df['file_name'] == audio_links[i], 'energy'].values[0]
                key = audio_links[i].split("/")[-1].split("?")[0]
                spec_value = df.loc[df['file_name'].str.contains(key), 'spec'].values[0]
                bandw_value = df.loc[df['file_name'].str.contains(key), 'bandw'].values[0]
                roof_value = df.loc[df['file_name'].str.contains(key), 'roof'].values[0]
                st.markdown(f"**🔁File giống thứ {i + 1}:  {key}**")
                st.audio(audio_links[i], format="audio/wav")
                plot_spectral_centroid(spec_value)
                plot_spectral_bandwidth(bandw_value)
                plot_spectral_rolloff(roof_value)


    else:
        st.info("📂 Vui lòng tải lên một file âm thanh.")

