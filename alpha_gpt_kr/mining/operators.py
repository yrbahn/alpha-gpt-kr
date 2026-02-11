"""
Alpha Operators Implementation
논문 Table 1의 모든 연산자 구현
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Optional
import warnings

warnings.filterwarnings('ignore')


class AlphaOperators:
    """
    Alpha-GPT에서 사용하는 모든 알파 연산자 구현
    논문의 Table 1 기반
    """
    
    # ============================================
    # Time-series Operators
    # ============================================
    
    @staticmethod
    def shift(x: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """시계열 데이터를 지정된 기간만큼 이동"""
        return x.shift(periods)
    
    @staticmethod
    def ts_delta(x: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """현재값 - N일 전 값"""
        return x - x.shift(periods)
    
    @staticmethod
    def ts_delta_ratio(x: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """(현재값 - N일 전 값) / N일 전 값"""
        prev = x.shift(periods)
        return (x - prev) / prev.replace(0, np.nan)
    
    @staticmethod
    def ts_mean(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """이동 평균"""
        return x.rolling(window=window, min_periods=1).mean()
    
    @staticmethod
    def ts_std(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """이동 표준편차"""
        return x.rolling(window=window, min_periods=1).std()
    
    @staticmethod
    def ts_sum(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """이동 합계"""
        return x.rolling(window=window, min_periods=1).sum()
    
    @staticmethod
    def ts_product(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """이동 곱셈"""
        return x.rolling(window=window, min_periods=1).apply(np.prod, raw=True)
    
    @staticmethod
    def ts_min(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """이동 최소값"""
        return x.rolling(window=window, min_periods=1).min()
    
    @staticmethod
    def ts_max(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """이동 최대값"""
        return x.rolling(window=window, min_periods=1).max()
    
    @staticmethod
    def ts_argmin(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """이동 윈도우에서 최소값의 인덱스 (0부터 시작)"""
        return x.rolling(window=window, min_periods=1).apply(
            lambda arr: arr.argmin() if len(arr) > 0 else np.nan, raw=True
        )
    
    @staticmethod
    def ts_argmax(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """이동 윈도우에서 최대값의 인덱스 (0부터 시작)"""
        return x.rolling(window=window, min_periods=1).apply(
            lambda arr: arr.argmax() if len(arr) > 0 else np.nan, raw=True
        )
    
    @staticmethod
    def ts_rank(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """이동 윈도우 내에서의 순위 (0~1)"""
        return x.rolling(window=window, min_periods=1).apply(
            lambda arr: stats.rankdata(arr)[-1] / len(arr) if len(arr) > 0 else np.nan,
            raw=True
        )
    
    @staticmethod
    def ts_corr(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """두 시계열 간의 이동 상관계수"""
        return x.rolling(window=window, min_periods=1).corr(y)
    
    @staticmethod
    def ts_cov(x: pd.DataFrame, y: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """두 시계열 간의 이동 공분산"""
        return x.rolling(window=window, min_periods=1).cov(y)
    
    @staticmethod
    def ts_ema(x: pd.DataFrame, window: int = 10, alpha: float = 0.5) -> pd.DataFrame:
        """지수 이동 평균 (Exponential Moving Average)"""
        return x.ewm(span=window, adjust=False, alpha=alpha).mean()
    
    @staticmethod
    def ts_zscore_scale(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """이동 윈도우 내에서 Z-score 정규화"""
        mean = x.rolling(window=window, min_periods=1).mean()
        std = x.rolling(window=window, min_periods=1).std()
        return (x - mean) / std.replace(0, np.nan)
    
    @staticmethod
    def ts_maxmin_scale(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """이동 윈도우 내에서 Min-Max 정규화"""
        min_val = x.rolling(window=window, min_periods=1).min()
        max_val = x.rolling(window=window, min_periods=1).max()
        return (x - min_val) / (max_val - min_val).replace(0, np.nan)
    
    @staticmethod
    def ts_skew(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """이동 윈도우 내의 왜도 (Skewness)"""
        return x.rolling(window=window, min_periods=1).skew()
    
    @staticmethod
    def ts_kurt(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """이동 윈도우 내의 첨도 (Kurtosis)"""
        return x.rolling(window=window, min_periods=1).kurt()
    
    @staticmethod
    def ts_ir(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """Information Ratio: mean / std"""
        mean = x.rolling(window=window, min_periods=1).mean()
        std = x.rolling(window=window, min_periods=1).std()
        return mean / std.replace(0, np.nan)
    
    @staticmethod
    def ts_decayed_linear(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """선형 감쇠 가중 평균"""
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()
        
        def weighted_mean(arr):
            if len(arr) < window:
                w = np.arange(1, len(arr) + 1)
                w = w / w.sum()
                return np.sum(arr * w)
            return np.sum(arr * weights)
        
        return x.rolling(window=window, min_periods=1).apply(weighted_mean, raw=True)
    
    @staticmethod
    def ts_percentile(x: pd.DataFrame, window: int = 10, q: float = 0.5) -> pd.DataFrame:
        """이동 윈도우 내의 백분위수"""
        return x.rolling(window=window, min_periods=1).quantile(q)
    
    @staticmethod
    def ts_linear_reg(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """선형 회귀 기울기"""
        def calc_slope(arr):
            if len(arr) < 2:
                return np.nan
            x_vals = np.arange(len(arr))
            slope, _ = np.polyfit(x_vals, arr, 1)
            return slope
        
        return x.rolling(window=window, min_periods=2).apply(calc_slope, raw=True)
    
    @staticmethod
    def ts_max_diff(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """현재값 - 이동 윈도우 최대값"""
        return x - x.rolling(window=window, min_periods=1).max()
    
    @staticmethod
    def ts_min_diff(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """현재값 - 이동 윈도우 최소값"""
        return x - x.rolling(window=window, min_periods=1).min()
    
    @staticmethod
    def ts_argmaxmin_diff(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """argmax - argmin"""
        argmax = AlphaOperators.ts_argmax(x, window)
        argmin = AlphaOperators.ts_argmin(x, window)
        return argmax - argmin
    
    @staticmethod
    def ts_median(x: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """이동 중앙값"""
        return x.rolling(window=window, min_periods=1).median()
    
    # ============================================
    # Cross-sectional Operators
    # ============================================
    
    @staticmethod
    def zscore_scale(x: pd.DataFrame) -> pd.DataFrame:
        """횡단면 Z-score 정규화 (각 시점별)"""
        return x.apply(lambda row: (row - row.mean()) / row.std(), axis=1)
    
    @staticmethod
    def winsorize_scale(x: pd.DataFrame, lower: float = 0.05, upper: float = 0.95) -> pd.DataFrame:
        """횡단면 Winsorization (이상치 제거)"""
        def winsorize_row(row):
            lower_val = row.quantile(lower)
            upper_val = row.quantile(upper)
            return row.clip(lower=lower_val, upper=upper_val)
        
        return x.apply(winsorize_row, axis=1)
    
    @staticmethod
    def normed_rank(x: pd.DataFrame) -> pd.DataFrame:
        """횡단면 정규화 순위 (0~1)"""
        return x.apply(lambda row: row.rank() / len(row), axis=1)
    
    @staticmethod
    def normed_rank_diff(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """두 변수의 정규화 순위 차이"""
        return AlphaOperators.normed_rank(x) - AlphaOperators.normed_rank(y)
    
    @staticmethod
    def cwise_max(x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        """Element-wise maximum"""
        return pd.DataFrame(np.maximum(x, y), index=x.index, columns=x.columns)
    
    @staticmethod
    def cwise_min(x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        """Element-wise minimum"""
        return pd.DataFrame(np.minimum(x, y), index=x.index, columns=x.columns)
    
    # ============================================
    # Group-wise Operators
    # ============================================
    
    @staticmethod
    def grouped_demean(x: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
        """그룹별 평균 제거 (산업 중립화 등)"""
        result = x.copy()
        for date in x.index:
            for group in groups.loc[date].unique():
                mask = groups.loc[date] == group
                if mask.sum() > 0:
                    result.loc[date, mask] = x.loc[date, mask] - x.loc[date, mask].mean()
        return result
    
    @staticmethod
    def grouped_zscore_scale(x: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
        """그룹별 Z-score 정규화"""
        result = x.copy()
        for date in x.index:
            for group in groups.loc[date].unique():
                mask = groups.loc[date] == group
                if mask.sum() > 1:
                    group_data = x.loc[date, mask]
                    result.loc[date, mask] = (group_data - group_data.mean()) / group_data.std()
        return result
    
    @staticmethod
    def grouped_winsorize_scale(x: pd.DataFrame, groups: pd.DataFrame, 
                                lower: float = 0.05, upper: float = 0.95) -> pd.DataFrame:
        """그룹별 Winsorization"""
        result = x.copy()
        for date in x.index:
            for group in groups.loc[date].unique():
                mask = groups.loc[date] == group
                if mask.sum() > 0:
                    group_data = x.loc[date, mask]
                    lower_val = group_data.quantile(lower)
                    upper_val = group_data.quantile(upper)
                    result.loc[date, mask] = group_data.clip(lower=lower_val, upper=upper_val)
        return result
    
    @staticmethod
    def grouped_max(x: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
        """그룹별 최대값"""
        result = x.copy()
        for date in x.index:
            for group in groups.loc[date].unique():
                mask = groups.loc[date] == group
                if mask.sum() > 0:
                    result.loc[date, mask] = x.loc[date, mask].max()
        return result
    
    @staticmethod
    def grouped_min(x: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
        """그룹별 최소값"""
        result = x.copy()
        for date in x.index:
            for group in groups.loc[date].unique():
                mask = groups.loc[date] == group
                if mask.sum() > 0:
                    result.loc[date, mask] = x.loc[date, mask].min()
        return result
    
    @staticmethod
    def grouped_sum(x: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
        """그룹별 합계"""
        result = x.copy()
        for date in x.index:
            for group in groups.loc[date].unique():
                mask = groups.loc[date] == group
                if mask.sum() > 0:
                    result.loc[date, mask] = x.loc[date, mask].sum()
        return result
    
    @staticmethod
    def grouped_mean(x: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
        """그룹별 평균"""
        result = x.copy()
        for date in x.index:
            for group in groups.loc[date].unique():
                mask = groups.loc[date] == group
                if mask.sum() > 0:
                    result.loc[date, mask] = x.loc[date, mask].mean()
        return result
    
    @staticmethod
    def grouped_std(x: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
        """그룹별 표준편차"""
        result = x.copy()
        for date in x.index:
            for group in groups.loc[date].unique():
                mask = groups.loc[date] == group
                if mask.sum() > 1:
                    result.loc[date, mask] = x.loc[date, mask].std()
        return result
    
    # ============================================
    # Element-wise Operators
    # ============================================
    
    @staticmethod
    def relu(x: pd.DataFrame) -> pd.DataFrame:
        """ReLU activation: max(0, x)"""
        return pd.DataFrame(np.maximum(0, x), index=x.index, columns=x.columns)
    
    @staticmethod
    def neg(x: pd.DataFrame) -> pd.DataFrame:
        """부호 반전: -x"""
        return -x
    
    @staticmethod
    def abs(x: pd.DataFrame) -> pd.DataFrame:
        """절대값"""
        return x.abs()
    
    @staticmethod
    def log(x: pd.DataFrame) -> pd.DataFrame:
        """자연로그 (0보다 큰 값만)"""
        return np.log(x.replace(0, np.nan).clip(lower=1e-10))
    
    @staticmethod
    def sign(x: pd.DataFrame) -> pd.DataFrame:
        """부호 함수: -1, 0, 1"""
        return np.sign(x)
    
    @staticmethod
    def pow(x: pd.DataFrame, exponent: float = 2) -> pd.DataFrame:
        """거듭제곱"""
        return x ** exponent
    
    @staticmethod
    def pow_sign(x: pd.DataFrame, exponent: float = 2) -> pd.DataFrame:
        """부호 보존 거듭제곱: sign(x) * |x|^exponent"""
        return np.sign(x) * (x.abs() ** exponent)
    
    @staticmethod
    def round(x: pd.DataFrame, decimals: int = 0) -> pd.DataFrame:
        """반올림"""
        return x.round(decimals)
    
    @staticmethod
    def add(x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        """덧셈"""
        return x + y
    
    @staticmethod
    def minus(x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        """뺄셈"""
        return x - y
    
    @staticmethod
    def cwise_mul(x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        """곱셈"""
        return x * y
    
    @staticmethod
    def div(x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        """나눗셈 (0으로 나누기 방지)"""
        if isinstance(y, pd.DataFrame):
            return x / y.replace(0, np.nan)
        else:
            return x / y if y != 0 else x * np.nan
    
    @staticmethod
    def greater(x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        """x > y인 경우 1, 아니면 0"""
        return (x > y).astype(int)
    
    @staticmethod
    def less(x: pd.DataFrame, y: Union[pd.DataFrame, float]) -> pd.DataFrame:
        """x < y인 경우 1, 아니면 0"""
        return (x < y).astype(int)
    
    # ============================================
    # Utility Methods
    # ============================================
    
    @classmethod
    def get_all_operators(cls) -> dict:
        """모든 연산자 딕셔너리 반환"""
        return {
            name: getattr(cls, name)
            for name in dir(cls)
            if not name.startswith('_') and callable(getattr(cls, name))
            and name not in ['get_all_operators', 'validate_operator']
        }
    
    @classmethod
    def validate_operator(cls, expr: str) -> bool:
        """연산자 표현식이 유효한지 검증"""
        operators = cls.get_all_operators()
        # 간단한 검증 (실제로는 더 정교한 파싱 필요)
        for op_name in operators.keys():
            if op_name in expr:
                return True
        return False


# 편의를 위한 별칭
ops = AlphaOperators
