import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

print("Đang khởi tạo dữ liệu mẫu...")
# Dữ liệu mẫu (Bạn có thể thay thế bằng dataset lớn hơn đọc từ file CSV)
data = {
    'text': [
        "Bóng đá Việt Nam giành chiến thắng thuyết phục trước đối thủ mạnh",
        "Chính sách kinh tế mới ở Việt Nam giúp giảm lạm phát, ổn định lãi suất",
        "Điện thoại iPhone 16 mới ra mắt có camera siêu nét và chip AI",
        "Đội tuyển thể thao quốc gia đang tập huấn cho kỳ Olympic sắp tới",
        "Thị trường chứng khoán giảm điểm mạnh hôm nay, VN-Index bốc hơi",
        "Trí tuệ nhân tạo đang thay đổi ngành công nghệ phần mềm",
        "Chuyên gia nhận định về tình hình tài chính quý 3 có dấu hiệu khả quan",
        "Trận chung kết cúp C1 châu Âu diễn ra vào rạng sáng mai khép lại mùa giải",
        "Vi xử lý, thuật toán mới của máy tính mạnh hơn 20%, tiết kiệm điện năng",
        "Giá vàng trong nước tăng kỷ lục do ảnh hưởng từ thị trường quốc tế"
    ],
    'label': [
        "Thể thao",
        "Kinh tế",
        "Công nghệ",
        "Thể thao",
        "Kinh tế",
        "Công nghệ",
        "Kinh tế",
        "Thể thao",
        "Công nghệ",
        "Kinh tế"
    ]
}

df = pd.DataFrame(data)

print("Đang huấn luyện mô hình...")
# Trích xuất đặc trưng văn bản bằng TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Huấn luyện mô hình Naive Bayes (rất phù hợp với phân loại văn bản)
model = MultinomialNB()
model.fit(X, y)

# Khởi tạo thư mục nếu chưa có
os.makedirs("models", exist_ok=True)

# Lưu mô hình đã huấn luyện
joblib.dump(model, 'models/model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("✅ Đã huấn luyện và lưu mô hình thành công vào thư mục models/!")
