import dropbox
import io
import  os
import IPython.display as ipd
from api import api_create, folder_path
import json
# Thay token bằng access token bạn vừa tạo
ACCESS_TOKEN = api_create()

# Đường dẫn đến folder Dropbox của bạn
FOLDER_PATH = folder_path()
# Khởi tạo kết nối Dropbox
dbx = dropbox.Dropbox(ACCESS_TOKEN)


def list_and_print_preview_links(folder_path):
    try:
        result = dbx.files_list_folder(folder_path)
        wav_files = [entry for entry in result.entries if entry.name.endswith(".wav")]

        if not wav_files:
            print("❌ Không có file .wav nào trong folder.")
            return

        # Chỉ lấy 2 file đầu
        #wav_files = wav_files[:2]

        print("📃 Danh sách 2 file .wav đầu tiên và link preview:")
        arr = []
        a=0
        for file in wav_files:
            try:
                # Tạo hoặc lấy shared link
                shared_links = dbx.sharing_list_shared_links(path=file.path_lower, direct_only=True).links
                if shared_links:
                    preview_link = shared_links[0].url
                else:
                    shared_link_metadata = dbx.sharing_create_shared_link_with_settings(file.path_lower)
                    preview_link = shared_link_metadata.url
                arr.append(str(preview_link))
                #print(f"- {file.name}: {preview_link}")
                print(a)
                a+=1

            except dropbox.exceptions.ApiError as e:
                print(f"Lỗi lấy shared link cho {file.name}: {e}")

        # Lấy link preview của file đầu tiên trong 2 file này
        # first_file = wav_files[0]
        # first_shared_links = dbx.sharing_list_shared_links(path=first_file.path_lower, direct_only=True).links
        # if first_shared_links:
        #     first_link = first_shared_links[0].url
        # else:
        #     first_link = dbx.sharing_create_shared_link_with_settings(first_file.path_lower).url
        #
        # print(f"\n🔗 Link preview file đầu tiên: {first_link}")
        # return first_link
        return arr
    except dropbox.exceptions.ApiError as e:
        print("Lỗi API Dropbox:", e)


url = list_and_print_preview_links(FOLDER_PATH)
# Lưu vào file
with open("Data/my_list.json", "w") as f:
    json.dump(url, f)
os.chmod("Data/my_list.json", 0o444)
print(url)


