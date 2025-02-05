from stable_baselines3 import PPO
from gym_trading.trading_env import TradingEnv
import os

# 1. 환경 생성
env = TradingEnv()

# 2. 강화학습 모델 학습
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# 3. 모델 저장 
os.makedirs("models", exist_ok=True)
model.save("models/ppo_trading_model")
print("모델이 models/ppo_trading_model.zip 에 저장되었습니다!")
