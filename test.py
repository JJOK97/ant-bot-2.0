from stable_baselines3 import PPO
from gym_trading.trading_env import TradingEnv
import os

# 모델 파일 확인 후 로드
model_path = "models/ppo_trading_model.zip"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

env = TradingEnv()
model = PPO.load(model_path)

obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}")

print("AI 매매 테스트 완료!")
