import os
import pandas as pd
from pathlib import Path

# Đường dẫn tới thư mục vntc_tmp
VNTC_DIR = os.path.join(os.path.dirname(__file__), "vntc_tmp")
TRAINING_FILE = os.path.join(os.path.dirname(__file__), "training_data.csv")

# Danh sách các thư mục danh mục
CATEGORIES = ["Chính trị", "Công nghệ", "Kinh tế", "Thể thao"]


def extract_text_from_txt(file_path: str) -> str:
    """Đọc nội dung từ file .txt với xử lý nhiều loại encoding"""
    # Thử các encoding phổ biến
    encodings = ['utf-16', 'utf-8', 'cp1252', 'latin-1', 'utf-16-le', 'utf-16-be']
    
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read().strip()
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            print(f"Lỗi khác khi đọc {file_path} với encoding {encoding}: {e}")
            continue
    
    # Nếu tất cả đều thất bại, trả về chuỗi rỗng
    return ""


def load_vntc_data() -> pd.DataFrame:
    """Lấy dữ liệu từ các thư mục vntc_tmp"""
    data = []

    for category in CATEGORIES:
        category_path = os.path.join(VNTC_DIR, category)

        if not os.path.exists(category_path):
            print(f"⚠️  Không tìm thấy thư mục: {category_path}")
            continue

        # Lấy danh sách các file .txt trong thư mục
        txt_files = list(Path(category_path).glob("*.txt"))
        print(f"📁 {category}: Tìm thấy {len(txt_files)} file")

        for txt_file in txt_files:
            content = extract_text_from_txt(str(txt_file))
            if content:  # Chỉ thêm nếu có nội dung
                data.append({"label": category, "content": content})

    return pd.DataFrame(data)


def merge_with_existing_training_data():
    """Hợp nhất dữ liệu mới với dữ liệu training hiện tại"""
    
    # Tải dữ liệu hiện tại (nếu có)
    existing_data = None
    if os.path.exists(TRAINING_FILE):
        try:
            existing_data = pd.read_csv(
                TRAINING_FILE,
                sep=r"\|\|",
                engine="python",
                encoding="utf-8",
                skipinitialspace=True,
            )
            existing_data.columns = [col.strip() for col in existing_data.columns]
            print(f"✅ Tải dữ liệu hiện tại: {len(existing_data)} dòng")
        except Exception as e:
            print(f"⚠️  Không thể tải dữ liệu hiện tại: {e}")
            existing_data = None

    # Lấy dữ liệu từ vntc_tmp
    new_data = load_vntc_data()
    print(f"\n📊 Dữ liệu mới từ vntc_tmp: {len(new_data)} dòng")

    # Hợp nhất
    if existing_data is not None:
        # Loại bỏ dữ liệu trùng lặp dựa trên nội dung
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        combined_data = combined_data.drop_duplicates(subset=["content"], keep="first")
        print(f"✅ Hợp nhất hoàn tất: {len(combined_data)} dòng (sau khi loại bỏ trùng lặp)")
    else:
        combined_data = new_data
        print(f"✅ Sử dụng dữ liệu mới: {len(combined_data)} dòng")

    # Lưu vào file với định dạng label||content
    combined_data = combined_data[["label", "content"]].copy()
    
    with open(TRAINING_FILE, "w", encoding="utf-8") as f:
        f.write("label||content\n")
        for idx, row in combined_data.iterrows():
            f.write(f"{row['label']}||{row['content']}\n")
    
    print(f"\n💾 Lưu vào file: {TRAINING_FILE}")

    # Hiển thị thống kê
    print("\n📈 Thống kê dữ liệu theo danh mục:")
    stats = combined_data["label"].value_counts()
    for label, count in stats.items():
        print(f"  {label}: {count} dòng")


if __name__ == "__main__":
    print("🚀 Bắt đầu load dữ liệu từ vntc_tmp...\n")
    merge_with_existing_training_data()
    print("\n✨ Hoàn tất!")
