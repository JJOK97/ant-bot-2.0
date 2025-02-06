import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib

# 데이터 로드
data = pd.read_csv("data/historical_data.csv")
data["Return"] = data["Close"].pct_change()  # 수익률 계산
data.dropna(inplace=True)

# 데이터 스케일링
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[["Return"]])

# LSTM 학습 데이터 생성
X, y = [], []
time_step = 10  # 10일 동안의 데이터를 사용하여 예측
for i in range(len(data_scaled) - time_step):
    X.append(data_scaled[i:i+time_step])
    y.append(data_scaled[i+time_step])

X, y = np.array(X), np.array(y)

# LSTM 모델 생성
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(50, return_sequences=False),
    Dense(25, activation="relu"),
    Dense(1, activation="sigmoid")  # 0~1 사이의 위험도 예측
])

# 모델 컴파일
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=50, batch_size=16)

# 모델 저장
model.save("models/lstm_risk_model.h5")
joblib.dump(scaler, "models/scaler_risk.pkl")
print("LSTM 리스크 모델 저장 완료!")
