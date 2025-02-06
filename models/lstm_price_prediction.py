import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 데이터 경로 설정
data_dir = "data"
data_path = os.path.join(data_dir, "historical_data.csv")

# data 폴더가 없으면 생성
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# CSV 파일이 없으면 자동 생성
if not os.path.exists(data_path):
    print(f"{data_path} 파일이 없어서 새로 생성합니다.")
    dates = pd.date_range(start="2024-02-01", periods=100, freq="D")
    prices = np.linspace(50000, 55000, num=100) + np.random.normal(0, 500, 100)

    df = pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "Close": prices})
    df.to_csv(data_path, index=False)
    print(f"새로운 데이터 파일이 생성되었습니다: {data_path}")

# CSV 파일 읽기
data = pd.read_csv(data_path)
print("CSV 데이터 로드 완료:")
print(data.head())

# Close 가격 데이터만 가져오기
prices = data["Close"].values.reshape(-1, 1)

# 데이터 정규화 (MinMaxScaler)
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# 슬라이딩 윈도우 데이터 생성 함수
def create_dataset(data, time_step=10):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    X, Y = np.array(X), np.array(Y)

    # 데이터가 정상적으로 생성되었는지 확인
    if X.shape[0] == 0:
        raise ValueError("데이터셋이 비어 있습니다. 'time_step' 값을 조정하세요.")

    return X, Y

# LSTM 입력 데이터 생성
time_step = 10
X, Y = create_dataset(prices_scaled, time_step)

# LSTM 입력 데이터 차원 조정 (samples, time_steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 모델 구성
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

# 모델 컴파일
model.compile(optimizer="adam", loss="mean_squared_error")

# 모델 학습
model.fit(X, Y, epochs=20, batch_size=16, verbose=1)

# 모델 저장
model_save_path = "models/lstm_price_prediction.h5"
model.save(model_save_path)
print(f"모델이 저장되었습니다: {model_save_path}")

# 예측 수행
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)  # 스케일링 복원

# 예측 결과 시각화
plt.figure(figsize=(12, 6))

# 실제 가격
plt.plot(data["Date"][time_step+1:], data["Close"][time_step+1:], label="Actual Prices", color="blue")

# 예측 가격
predictions = predictions.flatten()  # 1차원 변환
plt.plot(data["Date"][time_step+1:], predictions, label="Predicted Prices", color="red")

plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.title("LSTM Predicted vs Actual Prices")
plt.xticks(rotation=45)
plt.show()
