import gym
from gym import spaces
import numpy as np
import tensorflow as tf
import os

class TradingEnv(gym.Env):
    """LSTM 가격 예측을 활용한 강화학습 환경 + 샤프 비율 보상 시스템"""

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
        self.position = None  # 현재 포지션
        self.trade_history = []  # 매매 이력 (수익률 저장)
        self.rewards = []  # 보상 이력

    def predict_future_price(self):
        """LSTM 모델을 사용해 미래 가격 예측"""
        input_data = np.array(self.state).reshape(1, 10, 1)
        predicted_price = self.lstm_model.predict(input_data)[0][0]
        return predicted_price

    def calculate_sharpe_ratio(self):
        """샤프 비율 계산 (수익률의 평균 / 표준편차)"""
        if len(self.trade_history) < 5:
            return 0  # 샘플 부족 시 0 리턴
        avg_return = np.mean(self.trade_history)
        volatility = np.std(self.trade_history)
        sharpe_ratio = avg_return / (volatility + 1e-6)  # 0으로 나누는 것 방지
        return sharpe_ratio

    def step(self, action):
        """환경의 한 스텝 진행 (강화학습 진행)"""
        future_price = self.predict_future_price()

        reward = 0  # 기본 보상
        if action == 1:  # 매수
            if self.position is None:
                self.position = self.state[-1]
        elif action == 2:  # 매도
            if self.position is not None:
                profit = (self.state[-1] / self.position) - 1
                self.trade_history.append(profit)
                self.position = None

        # 샤프 비율 기반 보상 계산
        sharpe_ratio = self.calculate_sharpe_ratio()
        reward = sharpe_ratio * 10  # 보상을 스케일링하여 강화학습에 반영

        # 새로운 관측 상태 업데이트 (예측 가격 포함)
        self.state = np.append(self.state[1:], future_price)

        self.current_step += 1
        self.done = self.current_step > 100

        return np.append(self.state, future_price), reward, self.done, {}

    def reset(self):
        """환경 초기화"""
        self.state = np.random.rand(10)
        self.current_step = 0
        self.trade_history = []
        self.position = None
        return np.append(self.state, self.predict_future_price())

# 환경 테스트 코드
if __name__ == "__main__":
    env = TradingEnv("models/lstm_price_prediction.h5")
    print(env.step(1))  # 매수 실행 예제
