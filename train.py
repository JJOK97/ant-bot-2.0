from stable_baselines3 import PPO
from gym_trading.trading_env import TradingEnv
import os

# 강화학습 환경 생성 (LSTM 가격 예측 모델 반영)
env = TradingEnv("models/lstm_price_prediction.h5")

# PPO 강화학습 모델 학습
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=20000)

# 학습된 모델 저장
os.makedirs("models", exist_ok=True)
model.save("models/ppo_lstm_trading_model")
print("PPO + LSTM 강화학습 모델 저장 완료!")
