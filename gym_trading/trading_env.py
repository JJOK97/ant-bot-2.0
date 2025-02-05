import gym
from gym import spaces
import numpy as np

class TradingEnv(gym.Env):
    
    def __init__(self):
        super(TradingEnv, self).__init__()

        # 매수(1), 매도(2), 유지(0) 3가지 액션 가능
        self.action_space = spaces.Discrete(3)

        # 관측 공간: 최근 10개의 가격 데이터를 입력받음
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)

        # 초기 상태 (랜덤 가격 데이터 생성)
        self.state = np.random.rand(10)
        self.done = False
        self.current_step = 0

    def step(self, action):
        """환경의 한 스텝 진행 (매수/매도/유지 결정)"""
        reward = np.random.randn()  # 랜덤 보상 (초기 테스트용)
        self.current_step += 1
        self.done = self.current_step > 100  # 100 스텝 후 종료

        return np.random.rand(10), reward, self.done, {}

    def reset(self):
        """환경 초기화"""
        self.state = np.random.rand(10)
        self.current_step = 0
        return self.state

# 환경 테스트
if __name__ == "__main__":
    env = TradingEnv()
    print(env.step(1))  # 매수 실행 예제
