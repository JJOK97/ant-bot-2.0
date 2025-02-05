import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# 데이터 불러오기
data = pd.read_csv("data/historical_data.csv")
prices = data["Close"].values.reshape(-1, 1)

# LSTM 모델 불러오기
model = tf.keras.models.load_model("models/lstm_price_prediction.h5")

# 데이터 스케일링 적용
scaler = MinMaxScaler()
prices_scaled = scaler.fit_transform(prices)

# 예측 데이터 준비
time_step = 10
def prepare_input(data, time_step):
    return np.array([data[-time_step:].reshape(time_step, 1)])

# 예측 수행
last_data = prepare_input(prices_scaled, time_step)
predicted_price = model.predict(last_data)
predicted_price = scaler.inverse_transform(predicted_price)[0][0]

print(f"LSTM 예측 가격: {predicted_price}")

# 단순 백테스팅 전략 (예측 가격이 현재 가격보다 높으면 매수)
current_price = prices[-1][0]
if predicted_price > current_price:
    print("매수 추천!")
else:
    print("매도 추천!")
