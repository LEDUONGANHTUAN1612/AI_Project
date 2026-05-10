import os
import pandas as pd
from pathlib import Path
from load_vntc_data import extract_text_from_txt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(SCRIPT_DIR, "vntc_tmp", "Test")
TEST_FILE = os.path.join(SCRIPT_DIR, "test_data.csv")

CATEGORIES = ["Chính trị", "Công nghệ", "Kinh tế", "Thể thao"]

def generate_test_data() -> pd.DataFrame:
    """Đọc dữ liệu từ thư mục Test và tạo test_data.csv"""
    data = []
    
    for category in CATEGORIES:
        category_path = os.path.join(TEST_DIR, category)
        
        if not os.path.exists(category_path):
            print(f"⚠️  Không tìm thấy thư mục Test: {category_path}")
            continue
            
        txt_files = list(Path(category_path).glob("*.txt"))[:500]
        print(f"📁 {category}: Tìm thấy {len(txt_files)} file test")
        
        for txt_file in txt_files:
            content = extract_text_from_txt(str(txt_file))
            if content:
                data.append({"label": category, "content": content})
                
    df = pd.DataFrame(data)
    
    if len(df) > 0:
        with open(TEST_FILE, "w", encoding="utf-8") as f:
            f.write("label||content\n")
            for idx, row in df.iterrows():
                f.write(f"{row['label']}||{row['content']}\n")
        print(f"✅ Đã lưu {len(df)} file test vào {TEST_FILE}")
    else:
        print("⚠️ Không có dữ liệu test!")
        
    return df

if __name__ == "__main__":
    generate_test_data()
