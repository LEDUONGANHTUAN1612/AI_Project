import streamlit as st
import joblib
import os

# Cấu hình giao diện trang web
st.set_page_config(page_title="AI Phân loại Văn bản", layout="centered", page_icon="🤖")

st.title("🤖 Ứng dụng AI Phân loại Văn bản")
st.markdown("Nhập một đoạn văn bản tiếng Việt và AI sẽ tự động nhận diện xem nó thuộc chủ đề: **Kinh tế**, **Thể thao**, hay **Công nghệ**.")

# Caching lại việc tải tải model để không load lại mỗi khi reload trang
@st.cache_resource
def load_model():
    model_path = os.path.join("models", "model.pkl")
    vector_path = os.path.join("models", "vectorizer.pkl")
    
    if os.path.exists(model_path) and os.path.exists(vector_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vector_path)
        return model, vectorizer
    else:
        return None, None

model, vectorizer = load_model()

if model is None:
    st.warning("⚠️ Không tìm thấy mô hình. Vui lòng chạy file `train_model.py` để huấn luyện ra dữ liệu mô hình trước.")
else:
    # Giao diện nhận dữ liệu
    user_input = st.text_area("Nhập đoạn văn bản của bạn tại đây:", height=150, placeholder="Ví dụ: Giá vàng hôm nay tăng mạnh, thị trường sôi động...")
    
    if st.button("Phân loại", type="primary"):
        if user_input.strip() == "":
            st.error("Vui lòng nhập văn bản trước khi phân loại!")
        else:
            with st.spinner("AI đang phân tích..."):
                # Tiền xử lý & Dự đoán
                X_input = vectorizer.transform([user_input])
                prediction = model.predict(X_input)[0]
                
                st.success(f"📌 Nhãn chủ đề dự đoán: **{prediction}**")
