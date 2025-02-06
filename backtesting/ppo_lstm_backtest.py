import sys
import os

# 현재 파일의 부모 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from gym_trading.trading_env import TradingEnv

# 학습된 모델 불러오기
env = TradingEnv("models/lstm_price_prediction.h5")
model = PPO.load("models/ppo_lstm_trading_model")

obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}")

print("PPO + LSTM 강화학습 백테스팅 완료!")
