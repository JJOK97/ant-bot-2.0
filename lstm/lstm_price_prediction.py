import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv("data/historical_data.csv")
prices = data["Close"].values.reshape(-1, 1)

# 데이터 스케일링 (0~1 범위로 변환)
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# LSTM 모델 학습을 위한 데이터 준비
def create_dataset(data, time_step=10):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i : (i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X, Y = create_dataset(prices_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)  # LSTM 입력 형식 변경

# LSTM 모델 생성
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
model.fit(X, Y, epochs=50, batch_size=16, verbose=1)

# 예측 수행
predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)  # 원래 가격 스케일로 변환

# 결과 시각화
plt.plot(data["Date"][time_step:], data["Close"][time_step:], label="Real Prices")
plt.plot(data["Date"][time_step:], predictions, label="Predicted Prices")
plt.legend()
plt.show()

# ✅ 모델 저장
model.save("models/lstm_price_prediction.h5")
print("✅ LSTM 가격 예측 모델 저장 완료!")
