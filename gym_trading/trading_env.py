import gym
from gym import spaces
import numpy as np
import tensorflow as tf
import os

class TradingEnv(gym.Env):
    """LSTM 가격 예측을 활용한 강화학습 환경"""

    def __init__(self, lstm_model_path):
        super(TradingEnv, self).__init__()

        # 액션 공간 (매수=1, 매도=2, 유지=0)
        self.action_space = spaces.Discrete(3)

        # 관측 공간: 최근 10개 가격 데이터 + LSTM 예측값 추가
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)

        # LSTM 모델 로드
        if os.path.exists(lstm_model_path):
            self.lstm_model = tf.keras.models.load_model(lstm_model_path)
        else:
            raise FileNotFoundError(f"LSTM 모델 파일이 없습니다: {lstm_model_path}")

        # 초기 상태 (랜덤 가격 데이터 생성)
        self.state = np.random.rand(10)
        self.done = False
        self.current_step = 0

    def predict_future_price(self):
        """LSTM 모델을 사용해 미래 가격 예측"""
        input_data = np.array(self.state).reshape(1, 10, 1)
        predicted_price = self.lstm_model.predict(input_data)[0][0]
        return predicted_price

    def step(self, action):
        """환경의 한 스텝 진행"""
        future_price = self.predict_future_price()

        # 보상 계산 (예제: 예측 가격이 상승할 것으로 보이면 매수, 하락이면 매도)
        reward = (future_price - self.state[-1]) * (1 if action == 1 else -1)

        # 새로운 관측 상태 업데이트 (예측 가격 포함)
        self.state = np.append(self.state[1:], future_price)

        self.current_step += 1
        self.done = self.current_step > 100

        return np.append(self.state, future_price), reward, self.done, {}

    def reset(self):
        """환경 초기화"""
        self.state = np.random.rand(10)
        self.current_step = 0
        return np.append(self.state, self.predict_future_price())

# 환경 테스트
if __name__ == "__main__":
    env = TradingEnv("models/lstm_price_prediction.h5")
    print(env.step(1))  # 매수 실행 예제
