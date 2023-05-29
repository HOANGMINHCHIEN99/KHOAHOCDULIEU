from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

# Load mô hình từ file model.sav
model = pickle.load(open('model.sav', 'rb'))

# Chuyển đổi dữ liệu đầu vào
cat_features = np.array(['Tên chung cư', 'Đường/Phố', 'Xã/Phường/Thị Trấn', 'Quận/Huyện'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    ten_chung_cu = data['ten_chung_cu']
    duong_pho = data['duong_pho']
    xa_phuong = data['xa_phuong']
    quan_huyen = data['quan_huyen']
    dien_tich = float(data['dien_tich'])
    so_phong_ngu = int(data['so_phong_ngu'])
    so_toilet = int(data['so_toilet'])

    new_sample = pd.DataFrame({
        'Tên chung cư': [ten_chung_cu],
        'Đường/Phố': [duong_pho],
        'Xã/Phường/Thị Trấn': [xa_phuong],
        'Quận/Huyện': [quan_huyen],
        'Diện tích': [dien_tich],
        'Số phòng ngủ': [so_phong_ngu],
        'Số toilet': [so_toilet]
    }).reindex(columns=['Tên chung cư', 'Đường/Phố', 'Xã/Phường/Thị Trấn', 'Quận/Huyện', 'Diện tích', 'Số phòng ngủ', 'Số toilet'])

    for feature in cat_features:
        enc = LabelEncoder()
        new_sample[feature] = enc.fit_transform(new_sample[feature])

    predicted_price = model.predict(new_sample)[0]

    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True)
