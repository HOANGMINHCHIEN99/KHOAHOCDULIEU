import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

# Load mô hình từ file model.sav
model = pickle.load(open('model.sav', 'rb'))

# Chuyển đổi dữ liệu đầu vào
cat_features = np.array(['Tên chung cư', 'Đường/Phố', 'Xã/Phường/Thị Trấn', 'Quận/Huyện'])

while True:
    # Nhập dữ liệu đầu vào từ người dùng
    tenchungcu = input("Nhập tên chung cư: ")
    duongpho = input("Nhập đường/phố: ")
    xaphuong = input("Nhập xã/phường/thị trấn: ")
    quanhuyen = input("Nhập quận/huyện: ")
    dientich = float(input("Nhập diện tích: "))
    sophongngu = int(input("Nhập số phòng ngủ: "))
    sotoilet = int(input("Nhập số toilet: "))

    # Tạo DataFrame từ dữ liệu đầu vào và sắp xếp lại thứ tự các cột
    new_sample = pd.DataFrame({
        'Tên chung cư': [tenchungcu],
        'Đường/Phố': [duongpho],
        'Xã/Phường/Thị Trấn': [xaphuong],
        'Quận/Huyện': [quanhuyen],
        'Diện tích': [dientich],
        'Số phòng ngủ': [sophongngu],
        'Số toilet': [sotoilet]
    }).reindex(columns=['Tên chung cư', 'Đường/Phố', 'Xã/Phường/Thị Trấn', 'Quận/Huyện', 'Diện tích', 'Số phòng ngủ', 'Số toilet'])

    # Chuyển đổi dữ liệu đầu vào
    for feature in cat_features:
        enc = LabelEncoder()
        new_sample[feature] = enc.fit_transform(new_sample[feature])

    # Dự đoán giá
    predicted_price = model.predict(new_sample)
    print("Giá dự đoán cho căn hộ là:", predicted_price)

    # Hỏi người dùng có muốn dự đoán tiếp không
    choice = input("Bạn có muốn dự đoán giá cho căn hộ khác không? (có/không): ")
    if choice.lower() != 'có':
        break
