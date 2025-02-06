import os
import numpy as np
import pandas as pd
import requests
import joblib
import tensorflow as tf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_trading.trading_env import TradingEnv
from sklearn.preprocessing import MinMaxScaler

#  PPO 모델 저장 경로
ppo_model_path = "models/ppo_lstm_trading_model.zip"

#  LSTM 예측 모델 & 리스크 모델 로드
lstm_price_model_path = "models/lstm_price_prediction.h5"
lstm_risk_model_path = "models/lstm_risk_model.h5"
scaler_risk_path = "models/scaler_risk.pkl"

#  LSTM 모델 로드
price_model = tf.keras.models.load_model(lstm_price_model_path)
risk_model = tf.keras.models.load_model(
    lstm_risk_model_path,
    custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
)
scaler_risk = joblib.load(scaler_risk_path)

#  환경 설정
env = DummyVecEnv([lambda: TradingEnv(lstm_price_model_path)])

#  강화학습 보상 함수 조정
def custom_reward(action, state, future_price, position, trade_history):
    """
    PPO의 보상을 조정하여 손실 패널티를 강화하고 리스크 모델을 반영.
    """
    reward = 0

    #  LSTM 리스크 모델을 사용해 시장 위험 예측
    if len(trade_history) >= 10:
        risk_input = np.array(trade_history[-10:]).reshape(-1, 1)
        scaled_risk_input = scaler_risk.transform(risk_input)
        risk_score = risk_model.predict(np.array([scaled_risk_input]))[0, 0]
    else:
        risk_score = 0.5  # 기본값

    #  보상 계산
    price_change = state[-1] - state[-2] if len(state) > 1 else 0

    #  기본 보상: 가격 상승 시 매수(1) 보상, 하락 시 매도(2) 보상
    reward += price_change * (1 if action == 1 else -1)

    #  리스크 반영: 위험이 높을수록 보상 감소
    reward *= (1 - risk_score)

    #  손실 패널티 추가 (손실 발생 시 강한 패널티)
    if position is not None:
        profit = (future_price / position) - 1
        if profit < 0:
            reward += profit * 5  # 손실이 크면 패널티도 증가

    #  샤프 비율 기반 보상 조정
    if len(trade_history) > 5:
        avg_return = np.mean(trade_history)
        volatility = np.std(trade_history)
        sharpe_ratio = avg_return / (volatility + 1e-6)  # 0으로 나누는 것 방지
        reward += sharpe_ratio * 10

    return reward

#  PPO 모델 설정 & 학습
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)  # 학습 수행

#  PPO 모델 저장
model.save(ppo_model_path)
print(f" PPO 강화학습 모델이 저장되었습니다: {ppo_model_path}")
