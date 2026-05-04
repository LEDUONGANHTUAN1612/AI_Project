import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os
import argparse


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_training_data(file_path: str) -> pd.DataFrame:
    if os.path.exists(file_path):
        df = pd.read_csv(
            file_path,
            sep=r"\|\|",
            engine="python",
            encoding="utf-8",
            skipinitialspace=True,
        )

        df.columns = [col.strip() for col in df.columns]

        if "label" not in df.columns or "content" not in df.columns:
            raise ValueError("File train phải có 2 cột: label||content")

        df = df[["label", "content"]].copy()
        df["label"] = df["label"].astype(str).str.strip()
        df["content"] = df["content"].astype(str).str.strip()
        df = df.dropna(subset=["label", "content"])
        df = df[(df["label"] != "") & (df["content"] != "")]
        return df

    print(f"Không tìm thấy file '{file_path}', dùng dữ liệu mẫu mặc định.")
    return pd.DataFrame(
        {
            "content": [
                "Bóng đá Việt Nam giành chiến thắng thuyết phục trước đối thủ mạnh",
                "Chính sách kinh tế mới ở Việt Nam giúp giảm lạm phát, ổn định lãi suất",
                "Điện thoại iPhone 16 mới ra mắt có camera siêu nét và chip AI",
                "Đội tuyển thể thao quốc gia đang tập huấn cho kỳ Olympic sắp tới",
                "Thị trường chứng khoán giảm điểm mạnh hôm nay, VN-Index bốc hơi",
                "Trí tuệ nhân tạo đang thay đổi ngành công nghệ phần mềm",
                "Chuyên gia nhận định về tình hình tài chính quý 3 có dấu hiệu khả quan",
                "Trận chung kết cúp C1 châu Âu diễn ra vào rạng sáng mai khép lại mùa giải",
                "Vi xử lý, thuật toán mới của máy tính mạnh hơn 20%, tiết kiệm điện năng",
                "Giá vàng trong nước tăng kỷ lục do ảnh hưởng từ thị trường quốc tế",
            ],
            "label": [
                "Thể thao",
                "Kinh tế",
                "Công nghệ",
                "Thể thao",
                "Kinh tế",
                "Công nghệ",
                "Kinh tế",
                "Thể thao",
                "Công nghệ",
                "Kinh tế",
            ],
        }
    )


parser = argparse.ArgumentParser(description="Huấn luyện mô hình phân loại văn bản từ file label||content.")
parser.add_argument(
    "--data-file",
    default=os.path.join(SCRIPT_DIR, "training_data.csv"),
    help="Đường dẫn tới file train có định dạng label||content",
)
args = parser.parse_args()

print("Đang tải dữ liệu huấn luyện...")
df = load_training_data(args.data_file)

if len(df) == 0:
    raise ValueError("Không có dữ liệu hợp lệ để huấn luyện.")

print("Đang huấn luyện mô hình...")
# Trích xuất đặc trưng văn bản bằng TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['content'])
y = df['label']

# Huấn luyện mô hình Naive Bayes (rất phù hợp với phân loại văn bản)
model = MultinomialNB()
model.fit(X, y)

# Khởi tạo thư mục nếu chưa có
models_dir = os.path.join(SCRIPT_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

# Lưu mô hình đã huấn luyện
joblib.dump(model, os.path.join(models_dir, "model.pkl"))
joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.pkl"))

print("✅ Đã huấn luyện và lưu mô hình thành công vào thư mục models/!")
