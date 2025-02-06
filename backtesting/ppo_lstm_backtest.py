import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from gym_trading.trading_env import TradingEnv
from trading.evaluate_risk import calculate_mdd, calculate_sharpe_ratio, calculate_win_rate, calculate_profit_factor

# 실제 BTC 백테스트 데이터 로드
backtest_data_path = "data/backtest_results.csv"
if not os.path.exists(backtest_data_path):
    raise FileNotFoundError(f"백테스트용 데이터 파일이 없습니다: {backtest_data_path}")

backtest_data = pd.read_csv(backtest_data_path)
print(f" 백테스트 데이터 로드 완료: {backtest_data_path}")

# PPO 모델 & 환경 로드
env = TradingEnv("models/lstm_price_prediction.h5")
model = PPO.load("models/ppo_lstm_trading_model.zip")

# 백테스트 시작
obs = env.reset()
done = False
trade_returns = []
equity_curve = [100000]  # 초기 자본
position = None  # 초기 포지션 설정

for index, row in backtest_data.iterrows():
    actual_price = row["Actual"]
    predicted_price = row["Predicted"]

    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)

    # 트레이딩 기록 추가
    if action == 1:  # 매수
        if position is None:  # 포지션이 없을 때만 매수
            position = actual_price  # 매수 시점 가격 기록
            print(f"매수 주문 실행 @ {position}")

    elif action == 2 and position is not None:  # 매도
        profit = (actual_price / position) - 1  # 손익 계산
        trade_returns.append(profit)
        equity_curve.append(equity_curve[-1] * (1 + profit))

        # 손실 발생 시 출력
        if profit < 0:
            print(f"손실 거래 발생: {round(profit * 100, 2)}%")
        else:
            print(f"매도 주문 실행 @ {actual_price}, 수익률: {round(profit * 100, 2)}%")

        position = None  # 포지션 초기화

# 리스크 평가
mdd = calculate_mdd(equity_curve)
sharpe = calculate_sharpe_ratio(trade_returns)
win_rate = calculate_win_rate(trade_returns)
profit_factor = calculate_profit_factor(trade_returns)

# 결과 저장
report = f"""
백테스트 성과 리포트
-----------------------------------
최대 드로다운 (MDD): {round(mdd, 2)}%
샤프 비율: {round(sharpe, 2)}
승률: {round(win_rate, 2)}%
손익비 (Profit Factor): {round(profit_factor, 2)}
"""

print(report)

# 파일로 저장
with open("backtesting_results.txt", "w", encoding="utf-8") as f:
    f.write(report)
