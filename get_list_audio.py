import dropbox
import os
import json
from  api import API_TOKEN
# ======= CẤU HÌNH =======
ACCESS_TOKEN = API_TOKEN  # Thay bằng token thật
folder_path = "/Wav_main"           # Thư mục Dropbox cần quét
output_dir = "Data"                 # Thư mục chứa file json
output_file = "dropbox_links.json" # Tên file json đầu ra

os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_file)

dbx = dropbox.Dropbox(ACCESS_TOKEN)

def list_links_only(path, save_path):
    links = []
    try:
        result = dbx.files_list_folder(path)
        for entry in result.entries:
            if isinstance(entry, dropbox.files.FileMetadata):
                try:
                    try:
                        shared_link = dbx.sharing_create_shared_link_with_settings(entry.path_lower).url
                    except dropbox.exceptions.ApiError as e:
                        if isinstance(e.error, dropbox.sharing.CreateSharedLinkWithSettingsError) and e.error.is_shared_link_already_exists():
                            existing_links = dbx.sharing_list_shared_links(path=entry.path_lower, direct_only=True)
                            shared_link = existing_links.links[0].url if existing_links.links else "N/A"
                        else:
                            raise e
                    links.append(shared_link)
                    print(f"✔ {entry.name}: {shared_link}")
                except Exception as e:
                    print(f"❌ Lỗi tạo link cho {entry.name}: {e}")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(links, f, indent=2)
        print(f"\n✅ Đã lưu JSON dạng list tại: {save_path}")
    except Exception as e:
        print("❌ Lỗi liệt kê thư mục:", e)

list_links_only(folder_path, output_path)