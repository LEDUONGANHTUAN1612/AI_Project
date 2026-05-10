import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILE = os.path.join(SCRIPT_DIR, "test_data.csv")

def test_model():
    """Tải test file và đánh giá mô hình"""
    if not os.path.exists(TEST_FILE):
        print(f"⚠️ Không tìm thấy file {TEST_FILE}. Hãy chạy file create_test_data.py trước.")
        return

    print("Đang tải dữ liệu test...")
    df = pd.read_csv(
        TEST_FILE,
        sep=r"\|\|",
        engine="python",
        encoding="utf-8",
        skipinitialspace=True,
    )
    df.columns = [col.strip() for col in df.columns]
    
    df["label"] = df["label"].astype(str).str.strip()
    df["content"] = df["content"].astype(str).str.strip()
    df = df.dropna(subset=["label", "content"])
    df = df[(df["label"] != "") & (df["content"] != "")]
    df = df[(df["label"].str.lower() != "nan") & (df["content"].str.lower() != "nan")]
    
    if len(df) == 0:
        print("⚠️ Dữ liệu test rỗng!")
        return
        
    model_path = os.path.join(SCRIPT_DIR, "models", "model.pkl")
    vectorizer_path = os.path.join(SCRIPT_DIR, "models", "vectorizer.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print("⚠️ Không tìm thấy file mô hình!. Hãy chạy file train_model.py trước.")
        return
        
    print("Đang tải mô hình...")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    print("Đang đánh giá dữ liệu test...")
    X_test = vectorizer.transform(df["content"])
    y_test = df["label"]
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Kiểm thử mô hình hoàn tất!")
    print(f"Độ chính xác chung (Accuracy): {acc * 100:.2f}%\n")
    
    print("Chi tiết dự đoán theo từng chủ đề:")
    print("-" * 65)
    print(f"{'Chủ đề':<15} | {'Thực tế':<10} | {'AI Dự đoán':<12} | {'Nhận diện đúng':<12}")
    print("-" * 65)
    labels = sorted(df["label"].unique())
    for label in labels:
        actual_count = sum(y_test == label)
        pred_count = sum(y_pred == label)
        correct_count = sum((y_test == label) & (y_pred == label))
        print(f"{label:<15} | {actual_count:<10} | {pred_count:<12} | {correct_count:<12}")
    print("-" * 65)
    
    print("\nBáo cáo phân loại (Classification Report):")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    test_model()
