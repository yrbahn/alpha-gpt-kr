"""
Backtesting Engine
알파 팩터의 성능을 평가하는 백테스팅 엔진
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime
from loguru import logger


@dataclass
class BacktestResult:
    """백테스트 결과"""
    alpha_expr: str
    returns: pd.Series
    cumulative_returns: pd.Series
    
    # 성능 지표
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    ic: float  # Information Coefficient
    ic_std: float
    ir: float  # Information Ratio
    turnover: float
    
    # 추가 통계
    win_rate: float
    num_trades: int
    
    # 세부 데이터
    positions: Optional[pd.DataFrame] = None
    trades: Optional[pd.DataFrame] = None
    
    def summary(self) -> str:
        """결과 요약"""
        summary = f"""
=== Backtest Result ===
Alpha Expression: {self.alpha_expr}

Performance Metrics:
  Total Return:    {self.total_return:>8.2%}
  Annual Return:   {self.annual_return:>8.2%}
  Sharpe Ratio:    {self.sharpe_ratio:>8.2f}
  Max Drawdown:    {self.max_drawdown:>8.2%}
  
Factor Metrics:
  IC (mean):       {self.ic:>8.4f}
  IC (std):        {self.ic_std:>8.4f}
  IR (IC/std):     {self.ir:>8.2f}
  
Trading Metrics:
  Turnover:        {self.turnover:>8.2%}
  Win Rate:        {self.win_rate:>8.2%}
  Num Trades:      {self.num_trades:>8d}
"""
        return summary.strip()


class BacktestEngine:
    """
    백테스팅 엔진
    
    Features:
    - IC (Information Coefficient) 계산
    - 롱-숏 포트폴리오 시뮬레이션
    - 성능 지표 계산 (Sharpe, MDD, Turnover 등)
    """
    
    def __init__(self,
                 universe: List[str],
                 price_data: pd.DataFrame,
                 return_data: Optional[pd.DataFrame] = None,
                 initial_capital: float = 100_000_000,  # 1억원
                 commission: float = 0.0015,  # 0.15% (편도)
                 slippage: float = 0.001):  # 0.1%
        """
        Args:
            universe: 종목 유니버스
            price_data: 가격 데이터 (날짜 x 종목)
            return_data: 수익률 데이터 (None이면 자동 계산)
            initial_capital: 초기 자본
            commission: 수수료율
            slippage: 슬리피지율
        """
        self.universe = universe
        self.price_data = price_data
        self.return_data = return_data if return_data is not None else price_data.pct_change()
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        logger.info(f"BacktestEngine initialized: {len(universe)} stocks, "
                   f"{len(price_data)} days")
    
    def calculate_ic(self,
                    alpha: pd.DataFrame,
                    forward_return: Optional[pd.DataFrame] = None,
                    periods: int = 1) -> Tuple[pd.Series, float, float]:
        """
        Information Coefficient 계산
        
        Args:
            alpha: 알파 팩터 값 (날짜 x 종목)
            forward_return: 미래 수익률 (None이면 자동 계산)
            periods: 수익률 기간
            
        Returns:
            (IC 시계열, IC 평균, IC 표준편차)
        """
        if forward_return is None:
            forward_return = self.return_data.shift(-periods)
        
        ic_series = []
        
        for date in alpha.index[:-periods]:
            alpha_values = alpha.loc[date]
            return_values = forward_return.loc[date]
            
            # 유효한 값만 추출
            valid_mask = (~alpha_values.isna()) & (~return_values.isna())
            
            if valid_mask.sum() < 5:  # 최소 5개 종목
                ic_series.append(np.nan)
                continue
            
            # Spearman 상관계수 계산
            ic = alpha_values[valid_mask].corr(return_values[valid_mask], method='spearman')
            ic_series.append(ic)
        
        ic_series = pd.Series(ic_series, index=alpha.index[:-periods])
        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        
        return ic_series, ic_mean, ic_std
    
    def create_long_short_portfolio(self,
                                    alpha: pd.DataFrame,
                                    quantiles: Tuple[float, float] = (0.2, 0.8),
                                    leverage: float = 1.0) -> pd.DataFrame:
        """
        롱-숏 포트폴리오 생성
        
        Args:
            alpha: 알파 팩터 값
            quantiles: (롱 비율, 숏 비율) - 상위/하위 몇 %
            leverage: 레버리지 배수
            
        Returns:
            포지션 DataFrame (-1: 숏, 0: 중립, 1: 롱)
        """
        positions = pd.DataFrame(0.0, index=alpha.index, columns=alpha.columns)
        
        for date in alpha.index:
            alpha_values = alpha.loc[date]
            valid_mask = ~alpha_values.isna()
            
            if valid_mask.sum() < 10:
                continue
            
            # 순위 계산
            ranks = alpha_values[valid_mask].rank(pct=True)
            
            # 롱 포지션: 상위 quantiles[1] 이상
            long_mask = ranks >= quantiles[1]
            long_tickers = long_mask[long_mask].index.tolist()  # 명시적으로 티커 리스트 추출
            if len(long_tickers) > 0:
                long_weight = leverage / len(long_tickers)
                positions.loc[date, long_tickers] = long_weight
            
            # 숏 포지션: 하위 quantiles[0] 이하
            short_mask = ranks <= quantiles[0]
            short_tickers = short_mask[short_mask].index.tolist()  # 명시적으로 티커 리스트 추출
            if len(short_tickers) > 0:
                short_weight = -leverage / len(short_tickers)
                positions.loc[date, short_tickers] = short_weight
        
        return positions
    
    def calculate_portfolio_returns(self,
                                    positions: pd.DataFrame,
                                    returns: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        포트폴리오 수익률 계산
        
        Args:
            positions: 포지션 (날짜 x 종목)
            returns: 수익률 데이터
            
        Returns:
            포트폴리오 일별 수익률
        """
        if returns is None:
            returns = self.return_data
        
        # 전일 포지션 x 당일 수익률
        portfolio_returns = (positions.shift(1) * returns).sum(axis=1)
        
        return portfolio_returns
    
    def calculate_turnover(self, positions: pd.DataFrame) -> pd.Series:
        """
        포트폴리오 회전율 계산
        
        Args:
            positions: 포지션
            
        Returns:
            일별 회전율
        """
        position_changes = positions.diff().abs()
        turnover = position_changes.sum(axis=1) / 2  # 롱+숏 양방향이므로 /2
        return turnover
    
    def calculate_sharpe_ratio(self,
                              returns: pd.Series,
                              periods_per_year: int = 252) -> float:
        """
        Sharpe Ratio 계산
        
        Args:
            returns: 수익률 시계열
            periods_per_year: 연간 거래일 수
            
        Returns:
            Sharpe Ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        mean_return = returns.mean() * periods_per_year
        std_return = returns.std() * np.sqrt(periods_per_year)
        
        return mean_return / std_return
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Maximum Drawdown 계산
        
        Args:
            returns: 수익률 시계열
            
        Returns:
            MDD (음수)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    def backtest(self,
                alpha: pd.DataFrame,
                alpha_expr: str = "",
                quantiles: Tuple[float, float] = (0.2, 0.8),
                leverage: float = 1.0,
                rebalance_freq: str = '1D') -> BacktestResult:
        """
        백테스트 실행
        
        Args:
            alpha: 알파 팩터 값
            alpha_expr: 알파 표현식 (기록용)
            quantiles: 롱-숏 quantile
            leverage: 레버리지
            rebalance_freq: 리밸런싱 주기 ('1D', '5D', '1W' 등)
            
        Returns:
            BacktestResult
        """
        logger.info(f"Running backtest: {alpha_expr if alpha_expr else 'unnamed alpha'}")
        
        # IC 계산
        ic_series, ic_mean, ic_std = self.calculate_ic(alpha)
        ir = ic_mean / ic_std if ic_std > 0 else 0.0
        
        # 포트폴리오 생성
        positions = self.create_long_short_portfolio(alpha, quantiles, leverage)
        
        # 리밸런싱 주기 적용
        if rebalance_freq != '1D':
            positions = positions.resample(rebalance_freq).ffill()
        
        # 수익률 계산
        portfolio_returns = self.calculate_portfolio_returns(positions)
        
        # 거래 비용 차감
        turnover = self.calculate_turnover(positions)
        transaction_costs = turnover * (self.commission + self.slippage)
        net_returns = portfolio_returns - transaction_costs
        
        # 누적 수익률
        cumulative_returns = (1 + net_returns).cumprod()
        
        # 성과 지표
        total_return = cumulative_returns.iloc[-1] - 1
        num_years = len(net_returns) / 252
        annual_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0.0
        
        sharpe = self.calculate_sharpe_ratio(net_returns)
        mdd = self.calculate_max_drawdown(net_returns)
        avg_turnover = turnover.mean()
        
        # 승률 계산
        win_rate = (net_returns > 0).sum() / len(net_returns) if len(net_returns) > 0 else 0.0
        num_trades = (positions != 0).sum().sum()
        
        result = BacktestResult(
            alpha_expr=alpha_expr,
            returns=net_returns,
            cumulative_returns=cumulative_returns,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            ic=ic_mean,
            ic_std=ic_std,
            ir=ir,
            turnover=avg_turnover,
            win_rate=win_rate,
            num_trades=int(num_trades),
            positions=positions
        )
        
        logger.info(f"Backtest complete - IC: {ic_mean:.4f}, Sharpe: {sharpe:.2f}, "
                   f"Annual Return: {annual_return:.2%}")
        
        return result
    
    def cross_validate(self,
                      alpha: pd.DataFrame,
                      alpha_expr: str = "",
                      n_splits: int = 5,
                      test_size: float = 0.2) -> List[BacktestResult]:
        """
        교차 검증
        
        Args:
            alpha: 알파 팩터
            alpha_expr: 알파 표현식
            n_splits: 분할 수
            test_size: 테스트 비율
            
        Returns:
            BacktestResult 리스트
        """
        results = []
        total_days = len(alpha)
        split_size = int(total_days / n_splits)
        
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = min((i + 1) * split_size, total_days)
            
            alpha_fold = alpha.iloc[start_idx:end_idx]
            
            result = self.backtest(alpha_fold, f"{alpha_expr}_fold{i+1}")
            results.append(result)
        
        logger.info(f"Cross-validation complete: {n_splits} folds")
        return results


def quick_backtest(alpha: pd.DataFrame,
                   price_data: pd.DataFrame,
                   alpha_expr: str = "",
                   universe: Optional[List[str]] = None) -> BacktestResult:
    """
    빠른 백테스트 함수
    
    Args:
        alpha: 알파 팩터
        price_data: 가격 데이터
        alpha_expr: 알파 표현식
        universe: 종목 유니버스
        
    Returns:
        BacktestResult
    """
    if universe is None:
        universe = alpha.columns.tolist()
    
    engine = BacktestEngine(universe, price_data)
    return engine.backtest(alpha, alpha_expr)
