import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import PPO
from trading.trading_env import TradingEnv
from trading.evaluate_risk import calculate_mdd, calculate_sharpe_ratio, calculate_win_rate, calculate_profit_factor

# 데이터 로드
backtest_data_path = "data/historical_data.csv"
if not os.path.exists(backtest_data_path):
   raise FileNotFoundError(f"백테스트용 데이터 파일이 없습니다: {backtest_data_path}")

backtest_data = pd.read_csv(backtest_data_path)
print(f"백테스트 데이터 로드 완료: {backtest_data_path}")

# PPO 모델 & 환경 로드
env = TradingEnv("models/lstm_price_prediction.h5")
model = PPO.load("models/ppo_lstm_trading_model.zip")

# 백테스트 결과 저장용
backtest_results = []
obs = env.reset()
position = None
equity = 100000  # 초기 자본

for index, row in backtest_data.iterrows():
   current_price = row["price"]
   
   # PPO 모델로 행동 결정
   action, _states = model.predict(obs)
   obs, reward, done, info = env.step(action)
   
   # 매매 로직
   if action == 1 and position is None:  # 매수
       position = current_price
       trade_type = "매수"
       print(f"매수 @ {position:,.0f}")
       
   elif action == 2 and position is not None:  # 매도
       profit = (current_price / position) - 1
       equity *= (1 + profit)
       trade_type = "매도"
       print(f"매도 @ {current_price:,.0f}, 수익률: {profit*100:.2f}%")
       
       # 거래 기록 저장
       backtest_results.append({
           "date": row["date"],
           "price": current_price,
           "trade_type": trade_type,
           "profit": profit,
           "equity": equity
       })
       position = None

   if done:
       obs = env.reset()

# 결과를 DataFrame으로 변환
results_df = pd.DataFrame(backtest_results)
results_df.to_csv("data/backtest_results.csv", index=False)

# 성과 분석
if len(backtest_results) > 0:
   profits = [r["profit"] for r in backtest_results]
   equity_curve = [r["equity"] for r in backtest_results]
   
   mdd = calculate_mdd(equity_curve)
   sharpe = calculate_sharpe_ratio(profits)
   win_rate = calculate_win_rate(profits)
   profit_factor = calculate_profit_factor(profits)

   report = f"""
   백테스트 성과 리포트
   -----------------------------------
   총 거래 횟수: {len(backtest_results)}
   최종 자본: {equity:,.0f} KRW
   수익률: {((equity/100000)-1)*100:.2f}%
   최대 드로다운 (MDD): {mdd:.2f}%
   샤프 비율: {sharpe:.2f}
   승률: {win_rate:.2f}%
   손익비: {profit_factor:.2f}
   """

   print(report)

   with open("data/backtesting_results.txt", "w", encoding="utf-8") as f:
       f.write(report)