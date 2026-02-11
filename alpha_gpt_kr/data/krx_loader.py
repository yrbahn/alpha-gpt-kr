"""
한국 증시 데이터 로더
FinanceDataReader와 pykrx를 사용하여 KRX 데이터 수집
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Union
import os
from pathlib import Path
import warnings

try:
    import FinanceDataReader as fdr
except ImportError:
    fdr = None
    warnings.warn("FinanceDataReader not installed. Install with: pip install FinanceDataReader")

try:
    from pykrx import stock as krx_stock
except ImportError:
    krx_stock = None
    warnings.warn("pykrx not installed. Install with: pip install pykrx")

from loguru import logger


class KRXDataLoader:
    """
    한국 증시 데이터 로더
    
    Features:
    - OHLCV 데이터
    - VWAP (Volume-Weighted Average Price)
    - 산업/섹터 분류
    - 시가총액, 거래대금 등
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Args:
            cache_dir: 데이터 캐시 디렉토리
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if fdr is None:
            raise ImportError("FinanceDataReader is required. Install with: pip install FinanceDataReader")
        
        logger.info(f"KRXDataLoader initialized with cache_dir: {self.cache_dir}")
    
    def get_universe(self, 
                     market: str = "KOSPI", 
                     date: Optional[str] = None) -> List[str]:
        """
        시장 유니버스 종목 리스트 가져오기
        
        Args:
            market: KOSPI, KOSDAQ, KONEX, ALL
            date: 조회 날짜 (YYYY-MM-DD), None이면 최신
            
        Returns:
            종목 코드 리스트
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        cache_file = self.cache_dir / f"universe_{market}_{date}.csv"
        
        if cache_file.exists():
            logger.debug(f"Loading universe from cache: {cache_file}")
            df = pd.read_csv(cache_file)
            return df['Code'].tolist()
        
        logger.info(f"Fetching {market} universe for {date}")
        
        if market == "ALL":
            markets = ["KOSPI", "KOSDAQ"]
            codes = []
            for m in markets:
                codes.extend(self.get_universe(m, date))
            return list(set(codes))  # 중복 제거
        
        # FinanceDataReader로 종목 리스트 가져오기
        df = fdr.StockListing(market)
        
        if df is not None and not df.empty:
            df.to_csv(cache_file, index=False)
            logger.info(f"Cached {len(df)} stocks to {cache_file}")
            return df['Code'].tolist()
        
        logger.warning(f"Failed to fetch universe for {market}")
        return []
    
    def get_kospi200(self, date: Optional[str] = None) -> List[str]:
        """KOSPI 200 구성 종목"""
        # KOSPI200은 별도로 관리
        # 실제로는 KRX API나 별도 데이터 소스 필요
        logger.warning("KOSPI200 component list requires additional data source")
        # 임시로 시가총액 상위 200개 반환
        universe = self.get_universe("KOSPI", date)
        return universe[:200] if len(universe) >= 200 else universe
    
    def get_ohlcv(self,
                  tickers: Union[str, List[str]],
                  start_date: str,
                  end_date: str,
                  interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        OHLCV 데이터 가져오기
        
        Args:
            tickers: 종목 코드 또는 리스트
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            interval: 데이터 주기 (1d, 1h 등)
            
        Returns:
            {ticker: DataFrame} 딕셔너리
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        
        result = {}
        
        for ticker in tickers:
            cache_file = self.cache_dir / f"ohlcv_{ticker}_{start_date}_{end_date}_{interval}.pkl"
            
            if cache_file.exists():
                logger.debug(f"Loading {ticker} from cache")
                result[ticker] = pd.read_pickle(cache_file)
                continue
            
            try:
                logger.info(f"Fetching OHLCV for {ticker}")
                df = fdr.DataReader(ticker, start_date, end_date)
                
                if df is not None and not df.empty:
                    # 컬럼명 표준화
                    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                    
                    # VWAP 계산 (없는 경우)
                    if 'vwap' not in df.columns and all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
                        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
                    
                    # 캐시 저장
                    df.to_pickle(cache_file)
                    result[ticker] = df
                else:
                    logger.warning(f"No data for {ticker}")
                    
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
        
        return result
    
    def get_panel_data(self,
                       tickers: List[str],
                       start_date: str,
                       end_date: str,
                       fields: List[str] = ['open', 'high', 'low', 'close', 'volume']) -> Dict[str, pd.DataFrame]:
        """
        패널 데이터 형식으로 가져오기 (날짜 x 종목)
        
        Args:
            tickers: 종목 코드 리스트
            start_date: 시작일
            end_date: 종료일
            fields: 데이터 필드 리스트
            
        Returns:
            {field: DataFrame(date x ticker)} 딕셔너리
        """
        logger.info(f"Fetching panel data for {len(tickers)} tickers, {len(fields)} fields")
        
        # 개별 종목 데이터 수집
        ohlcv_data = self.get_ohlcv(tickers, start_date, end_date)
        
        if not ohlcv_data:
            logger.error("No data fetched")
            return {}
        
        # 패널 데이터 구성
        panel = {}
        
        for field in fields:
            field_lower = field.lower()
            field_data = {}
            
            for ticker, df in ohlcv_data.items():
                if field_lower in df.columns:
                    field_data[ticker] = df[field_lower]
            
            if field_data:
                panel[field] = pd.DataFrame(field_data)
                logger.debug(f"Created panel for {field}: {panel[field].shape}")
        
        return panel
    
    def get_sector_classification(self, 
                                  tickers: List[str],
                                  date: Optional[str] = None) -> pd.Series:
        """
        종목별 섹터/산업 분류
        
        Args:
            tickers: 종목 코드 리스트
            date: 조회 날짜
            
        Returns:
            Series(ticker -> sector)
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        cache_file = self.cache_dir / f"sectors_{date}.pkl"
        
        if cache_file.exists():
            logger.debug("Loading sectors from cache")
            sector_map = pd.read_pickle(cache_file)
        else:
            logger.info("Fetching sector classifications")
            sector_map = {}
            
            # KOSPI와 KOSDAQ 종목 리스트에서 섹터 정보 추출
            for market in ["KOSPI", "KOSDAQ"]:
                try:
                    df = fdr.StockListing(market)
                    if df is not None and 'Sector' in df.columns:
                        for _, row in df.iterrows():
                            if 'Code' in row and 'Sector' in row:
                                sector_map[row['Code']] = row['Sector']
                except Exception as e:
                    logger.error(f"Error fetching sectors for {market}: {e}")
            
            # 캐시 저장
            pd.Series(sector_map).to_pickle(cache_file)
        
        # 요청된 종목만 필터링
        result = pd.Series({ticker: sector_map.get(ticker, 'Unknown') for ticker in tickers})
        return result
    
    def calculate_returns(self, 
                         price_data: pd.DataFrame,
                         periods: int = 1,
                         method: str = 'simple') -> pd.DataFrame:
        """
        수익률 계산
        
        Args:
            price_data: 가격 데이터 (DataFrame)
            periods: 수익률 계산 기간
            method: 'simple' 또는 'log'
            
        Returns:
            수익률 DataFrame
        """
        if method == 'simple':
            returns = price_data.pct_change(periods)
        elif method == 'log':
            returns = np.log(price_data / price_data.shift(periods))
        else:
            raise ValueError(f"Invalid method: {method}. Use 'simple' or 'log'")
        
        return returns
    
    def get_market_cap(self,
                       tickers: List[str],
                       date: str) -> pd.Series:
        """
        종목별 시가총액
        
        Args:
            tickers: 종목 코드 리스트
            date: 조회 날짜
            
        Returns:
            Series(ticker -> market_cap)
        """
        logger.info(f"Fetching market cap for {len(tickers)} tickers on {date}")
        
        market_caps = {}
        
        if krx_stock is not None:
            try:
                # pykrx를 사용하여 시가총액 조회
                for ticker in tickers:
                    try:
                        df = krx_stock.get_market_cap(date, date, ticker)
                        if not df.empty and '시가총액' in df.columns:
                            market_caps[ticker] = df['시가총액'].iloc[0]
                    except Exception as e:
                        logger.debug(f"Failed to get market cap for {ticker}: {e}")
            except Exception as e:
                logger.error(f"Error in get_market_cap: {e}")
        
        return pd.Series(market_caps)
    
    def clean_data(self, 
                   data: pd.DataFrame,
                   fill_method: str = 'ffill',
                   drop_na: bool = False) -> pd.DataFrame:
        """
        데이터 정제
        
        Args:
            data: 원본 데이터
            fill_method: 결측치 처리 방법 ('ffill', 'bfill', None)
            drop_na: True면 결측치 행 제거
            
        Returns:
            정제된 DataFrame
        """
        cleaned = data.copy()
        
        # 무한대 값 제거
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        
        # 결측치 처리
        if drop_na:
            cleaned = cleaned.dropna()
        elif fill_method:
            cleaned = cleaned.fillna(method=fill_method)
        
        return cleaned
    
    def get_trading_dates(self,
                         start_date: str,
                         end_date: str,
                         market: str = "KOSPI") -> List[str]:
        """
        거래일 목록 가져오기
        
        Args:
            start_date: 시작일
            end_date: 종료일
            market: 시장 (KOSPI, KOSDAQ)
            
        Returns:
            거래일 리스트 (YYYY-MM-DD)
        """
        # KRX 지수 데이터를 사용하여 거래일 추출
        index_code = "KS11" if market == "KOSPI" else "KQ11"
        
        try:
            df = fdr.DataReader(index_code, start_date, end_date)
            if df is not None and not df.empty:
                return df.index.strftime("%Y-%m-%d").tolist()
        except Exception as e:
            logger.error(f"Error fetching trading dates: {e}")
        
        # 실패 시 영업일 기준으로 생성
        logger.warning("Using business days as fallback")
        dates = pd.date_range(start_date, end_date, freq='B')
        return dates.strftime("%Y-%m-%d").tolist()


# 편의 함수들
def load_krx_data(tickers: List[str],
                  start_date: str,
                  end_date: str,
                  fields: List[str] = ['close', 'volume'],
                  cache_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    KRX 데이터 로드 편의 함수
    
    Args:
        tickers: 종목 리스트
        start_date: 시작일
        end_date: 종료일
        fields: 필드 리스트
        cache_dir: 캐시 디렉토리
        
    Returns:
        {field: DataFrame} 딕셔너리
    """
    loader = KRXDataLoader(cache_dir=cache_dir)
    return loader.get_panel_data(tickers, start_date, end_date, fields)


def get_kospi_universe(date: Optional[str] = None,
                      cache_dir: Optional[str] = None) -> List[str]:
    """KOSPI 전 종목 가져오기"""
    loader = KRXDataLoader(cache_dir=cache_dir)
    return loader.get_universe("KOSPI", date)


def get_kosdaq_universe(date: Optional[str] = None,
                       cache_dir: Optional[str] = None) -> List[str]:
    """KOSDAQ 전 종목 가져오기"""
    loader = KRXDataLoader(cache_dir=cache_dir)
    return loader.get_universe("KOSDAQ", date)
