import sys
import os
import time
import numpy as np
import joblib
import tensorflow as tf
from stable_baselines3 import PPO
from trading_env import TradingEnv
from bithumb_trader import BithumbTrader
from evaluate_risk import calculate_mdd, calculate_sharpe_ratio, calculate_win_rate, calculate_profit_factor
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("BITHUMB_API_KEY")
SECRET_KEY = os.getenv("BITHUMB_SECRET_KEY")

LOG_FILE = "logs/live_trading.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

class LiveTrading:
    def __init__(self):
        self.env = TradingEnv("models/lstm_price_prediction.h5")
        self.model = PPO.load("models/ppo_lstm_trading_model.zip")
        self.risk_model = tf.keras.models.load_model(
            "models/lstm_risk_model.h5",
            custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
        )
        self.scaler = joblib.load("models/scaler_risk.pkl")
        self.trader = BithumbTrader(API_KEY, SECRET_KEY)

        self.price_history = []
        self.equity_curve = [100000]
        self.trade_returns = []
        self.window_size = 10
        self.position = None

    def log(self, message):
        """매매 발생 시 로그 기록"""
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        print(message)

    def calculate_volatility(self):
        """시장 변동성 계산"""
        if len(self.price_history) < self.window_size:
            return 0.02
        returns = np.diff(self.price_history) / self.price_history[:-1]
        return np.std(returns)

    def get_risk_level(self):
        """LSTM 모델 기반 시장 위험 예측"""
        if len(self.price_history) < self.window_size:
            return 0.5
        data = np.array(self.price_history[-self.window_size:]).reshape(-1, 1)
        scaled_data = self.scaler.transform(data).reshape(1, -1)
        return self.risk_model.predict(scaled_data)[0, 0]

    def run(self):
        """실시간 트레이딩 실행"""
        obs = self.env.reset()

        while True:
            price = self.trader.get_market_price()
            if not price:
                time.sleep(1)
                continue

            self.price_history.append(price)
            if len(self.price_history) > self.window_size:
                self.price_history.pop(0)

            volatility = self.calculate_volatility()
            stop_loss = 1 - (volatility * 2)
            take_profit = 1 + (volatility * 3)
            risk_score = self.get_risk_level()

            print(f"현재 BTC 가격: {price:,.0f} KRW")
            print(f"시장 변동성: {volatility*100:.2f}%, 손절: {(1-stop_loss)*100:.2f}%, 익절: {(take_profit-1)*100:.2f}%")
            print(f"AI 예측 위험도: {risk_score*100:.2f}%")

            action, _states = self.model.predict(obs)
            obs, reward, done, info = self.env.step(action)

            if action == 1 and risk_score < 0.5:
                if self.position is None:
                    result = self.trader.place_order("BTC", "bid", amount=0.001)
                    if result.get("status") == "0000":
                        self.position = price
                        self.log(f"매수 주문 실행 @ {price:,.0f} KRW")

            elif action == 2 or risk_score > 0.8:
                if self.position is not None:
                    result = self.trader.place_order("BTC", "ask", amount=0.001)
                    if result.get("status") == "0000":
                        profit = (price / self.position) - 1
                        self.trade_returns.append(profit)
                        self.log(f"매도 주문 실행 @ {price:,.0f} KRW, 수익률: {profit*100:.2f}%")
                        self.position = None
                        self.equity_curve.append(self.equity_curve[-1] * (1 + profit))

            if self.position is not None:
                if price <= self.position * stop_loss:
                    result = self.trader.place_order("BTC", "ask", amount=0.001)
                    if result.get("status") == "0000":
                        loss = (price / self.position) - 1
                        self.trade_returns.append(loss)
                        self.log(f"손절 실행! {price:,.0f} KRW (손실률: {loss*100:.2f}%)")
                        self.position = None
                        self.equity_curve.append(self.equity_curve[-1] * (1 + loss))
                elif price >= self.position * take_profit:
                    result = self.trader.place_order("BTC", "ask", amount=0.001)
                    if result.get("status") == "0000":
                        profit = (price / self.position) - 1
                        self.trade_returns.append(profit)
                        self.log(f"익절 실행! {price:,.0f} KRW (수익률: {profit*100:.2f}%)")
                        self.position = None
                        self.equity_curve.append(self.equity_curve[-1] * (1 + profit))

            if done:
                obs = self.env.reset()

            time.sleep(1)

if __name__ == "__main__":
    trader = LiveTrading()
    trader.run()
