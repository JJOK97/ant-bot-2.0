import numpy as np

def calculate_mdd(equity_curve):
    """최대 드로다운 (MDD) 계산"""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return np.min(drawdown) * 100  # 퍼센트 변환

def calculate_sharpe_ratio(trade_returns, risk_free_rate=0.02):
    """샤프 비율 계산"""
    if len(trade_returns) < 2:
        return 0
    excess_returns = np.array(trade_returns) - risk_free_rate
    return np.mean(excess_returns) / (np.std(excess_returns) + 1e-6)

def calculate_win_rate(trade_returns):
    """승률 계산"""
    wins = sum(1 for r in trade_returns if r > 0)
    return (wins / len(trade_returns)) * 100 if trade_returns else 0

def calculate_profit_factor(trade_returns):
    """손익비 (Profit Factor) 계산"""
    gains = sum(r for r in trade_returns if r > 0)
    losses = abs(sum(r for r in trade_returns if r < 0))
    return gains / (losses + 1e-6)  # 0으로 나누는 것 방지
