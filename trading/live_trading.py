import sys
import os
import requests
import time
import numpy as np
import joblib
import tensorflow as tf
from stable_baselines3 import PPO
from gym_trading.trading_env import TradingEnv
from evaluate_risk import calculate_mdd, calculate_sharpe_ratio, calculate_win_rate, calculate_profit_factor

#  경로 설정 (필요한 경우)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#  PPO 모델 & 환경 로드
env = TradingEnv("models/lstm_price_prediction.h5")
model = PPO.load("models/ppo_lstm_trading_model.zip")

#  LSTM 리스크 모델 로드
risk_model = tf.keras.models.load_model(
    "models/lstm_risk_model.h5",
    custom_objects={"mse": tf.keras.losses.MeanSquaredError()}
)
scaler = joblib.load("models/scaler_risk.pkl")

API_URL = "https://api.bithumb.com/public/ticker/BTC_KRW"

#  가격 데이터 저장 (변동성 & 리스크 예측용)
price_history = []
equity_curve = [100000]  # 초기 자본
trade_returns = []
window_size = 10  # 최근 10개 가격 변동성 기준
position = None  # 초기 포지션 설정

def get_market_price():
    """실시간 BTC 가격을 가져옴"""
    try:
        response = requests.get(API_URL)
        data = response.json()
        return float(data["data"]["closing_price"])
    except Exception as e:
        print(f"실시간 가격 가져오기 실패: {e}")
        return None

def calculate_volatility():
    """최근 가격 변동성 계산 (표준편차 기반)"""
    if len(price_history) < window_size:
        return 0.02  # 기본값 (2% 변동성)
    
    returns = np.diff(price_history) / price_history[:-1]  # 수익률 계산
    return np.std(returns)  # 표준편차 = 변동성

def get_risk_level():
    """LSTM 모델을 사용하여 시장 위험 예측"""
    if len(price_history) < window_size:
        return 0.5  # 기본 위험도

    #  수익률 변환 후 LSTM 입력 데이터 생성
    scaled_data = scaler.transform(np.array(price_history[-window_size:]).reshape(-1, 1))
    risk_score = risk_model.predict(np.array([scaled_data]))[0, 0]
    return risk_score  # 0(낮은 위험) ~ 1(높은 위험)

if __name__ == "__main__":
    obs = env.reset()

    while True:
        price = get_market_price()
        if price:
            print(f"현재 BTC 가격: {price} KRW")

            #  가격 기록 저장
            price_history.append(price)
            if len(price_history) > window_size:
                price_history.pop(0)

            #  동적 변동성 계산
            volatility = calculate_volatility()
            stop_loss = 1 - (volatility * 2)  # 변동성이 높을수록 손절폭 확대
            take_profit = 1 + (volatility * 3)  # 변동성이 높을수록 익절폭 확대

            #  AI 기반 시장 위험 예측
            risk_score = get_risk_level()
            print(f"시장 변동성: {round(volatility * 100, 2)}%, 손절: {round((1 - stop_loss) * 100, 2)}%, 익절: {round((take_profit - 1) * 100, 2)}%")
            print(f"AI 예측 위험도: {round(risk_score * 100, 2)}%")

            #  PPO 모델이 행동 결정 (매수=1, 매도=2, 유지=0)
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)

            #  매수 / 매도 결정 로직 추가
            if action == 1 and risk_score < 0.5:  # 위험이 낮으면 매수
                if position is None:  # 새로운 매수
                    position = price
                    print(f"매수 주문 실행 @ {price} KRW")
            elif action == 2 or risk_score > 0.8:  # 위험이 높으면 매도
                if position is not None:  # 포지션 있을 때만 매도 가능
                    profit = (price / position) - 1
                    trade_returns.append(profit)
                    print(f"매도 주문 실행 @ {price} KRW, 수익률: {round(profit * 100, 2)}%")
                    position = None  # 포지션 정리

                    #  자본 곡선 업데이트
                    equity_curve.append(equity_curve[-1] * (1 + profit))

            #  동적 손절 / 익절 체크
            if position is not None:
                if price <= position * stop_loss:
                    print(f"손절 실행! {price} KRW (기준: {position * stop_loss} KRW)")
                    position = None  # 손절 후 포지션 종료
                elif price >= position * take_profit:
                    print(f"익절 실행! {price} KRW (기준: {position * take_profit} KRW)")
                    position = None  # 익절 후 포지션 종료

            if done:
                obs = env.reset()

            #  트레이딩 종료 후 리스크 평가
            if len(trade_returns) > 5:  # 최소 거래 5개 이상 필요
                mdd = calculate_mdd(equity_curve)
                sharpe = calculate_sharpe_ratio(trade_returns)
                win_rate = calculate_win_rate(trade_returns)
                profit_factor = calculate_profit_factor(trade_returns)

                print(f"최대 드로다운 (MDD): {round(mdd, 2)}%")
                print(f"샤프 비율: {round(sharpe, 2)}")
                print(f"승률: {round(win_rate, 2)}%")
                print(f"평균 손익비 (Profit Factor): {round(profit_factor, 2)}")

        time.sleep(1)  # 1초마다 갱신
