"""
PostgreSQL 기반 한국 증시 데이터 로더
marketsense 데이터베이스 전용
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from loguru import logger
import psycopg2
from psycopg2.extras import RealDictCursor


class PostgresDataLoader:
    """
    PostgreSQL marketsense 데이터베이스에서 주식 데이터 로드
    
    주요 테이블:
    - stocks: 종목 정보
    - price_data: OHLCV 가격 데이터
    - technical_indicators: 기술적 지표
    - supply_demand_data: 수급 데이터
    """
    
    def __init__(self, 
                 host: str = "192.168.0.248",
                 port: int = 5432,
                 database: str = "marketsense",
                 user: str = "yrbahn",
                 password: str = "1234"):
        """
        Args:
            host: PostgreSQL 호스트
            port: 포트 번호
            database: 데이터베이스 이름
            user: 사용자명
            password: 비밀번호
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        
        logger.info(f"PostgreSQL 데이터 로더 초기화: {host}:{port}/{database}")
        
        # 연결 테스트
        self._test_connection()
    
    def _test_connection(self):
        """데이터베이스 연결 테스트"""
        try:
            conn = self._get_connection()
            conn.close()
            logger.info("✅ 데이터베이스 연결 성공")
        except Exception as e:
            logger.error(f"❌ 데이터베이스 연결 실패: {e}")
            raise
    
    def _get_connection(self):
        """PostgreSQL 연결 생성"""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password
        )
    
    def load_data(self,
                  start_date: str = None,
                  end_date: str = None,
                  universe: Optional[List[str]] = None,
                  include_technical: bool = False,
                  include_supply_demand: bool = False) -> Dict[str, pd.DataFrame]:
        """
        가격 데이터 로드
        
        Args:
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            universe: 종목 코드 리스트 (None이면 전체)
            include_technical: 기술적 지표 포함 여부
            include_supply_demand: 수급 데이터 포함 여부 (외국인/기관/개인 순매수)
            
        Returns:
            Dict with keys: 'close', 'open', 'high', 'low', 'volume', 'vwap'
            수급 데이터 포함 시: 'foreign_net', 'inst_net', 'indiv_net', 'foreign_ownership'
            각 value는 DataFrame (index=date, columns=ticker)
        """
        logger.info(f"데이터 로드 시작: {start_date} ~ {end_date}")
        
        # 기본 날짜 설정
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
        
        conn = self._get_connection()
        
        try:
            # 1. 종목 정보 로드
            stocks_df = self._load_stocks(conn, universe)
            logger.info(f"종목 수: {len(stocks_df)}")
            
            # 2. 가격 데이터 로드
            price_df = self._load_price_data(conn, stocks_df, start_date, end_date)
            logger.info(f"가격 데이터: {len(price_df)} 행")
            
            # 3. 패널 데이터로 변환
            panel_data = self._convert_to_panel(price_df, stocks_df)
            
            # 4. 기술적 지표 추가 (옵션)
            if include_technical:
                tech_df = self._load_technical_indicators(conn, stocks_df, start_date, end_date)
                tech_panel = self._convert_technical_to_panel(tech_df, stocks_df)
                panel_data.update(tech_panel)
            
            # 5. 수급 데이터 추가 (옵션)
            if include_supply_demand:
                sd_df = self._load_supply_demand_data(conn, stocks_df, start_date, end_date)
                sd_panel = self._convert_supply_demand_to_panel(sd_df, stocks_df)
                panel_data.update(sd_panel)
                logger.info(f"수급 데이터 추가: {list(sd_panel.keys())}")
            
            logger.info(f"✅ 데이터 로드 완료: {list(panel_data.keys())}")
            return panel_data
            
        finally:
            conn.close()
    
    def _load_stocks(self, conn, tickers: Optional[List[str]] = None) -> pd.DataFrame:
        """종목 정보 로드"""
        query = """
            SELECT id, ticker, name, sector, industry, market_cap, index_membership
            FROM stocks
            WHERE is_active = true
        """
        
        if tickers:
            # 티커 리스트를 6자리 포맷으로 변환
            formatted_tickers = [f"{t:0>6}" if len(str(t)) < 6 else str(t) for t in tickers]
            ticker_list = "', '".join(formatted_tickers)
            query += f" AND ticker IN ('{ticker_list}')"
        
        query += " ORDER BY ticker;"
        
        df = pd.read_sql(query, conn)
        logger.info(f"종목 로드: {len(df)}개")
        return df
    
    def _load_price_data(self, conn, stocks_df: pd.DataFrame, 
                        start_date: str, end_date: str) -> pd.DataFrame:
        """가격 데이터 로드"""
        stock_ids = stocks_df['id'].tolist()
        stock_id_list = ', '.join(map(str, stock_ids))
        
        query = f"""
            SELECT 
                p.stock_id,
                p.date,
                p.open,
                p.high,
                p.low,
                p.close,
                p.adj_close,
                p.volume
            FROM price_data p
            WHERE p.stock_id IN ({stock_id_list})
                AND p.date >= '{start_date}'
                AND p.date <= '{end_date}'
            ORDER BY p.date, p.stock_id;
        """
        
        df = pd.read_sql(query, conn)
        
        # stock_id를 ticker로 매핑
        id_to_ticker = dict(zip(stocks_df['id'], stocks_df['ticker']))
        df['ticker'] = df['stock_id'].map(id_to_ticker)
        
        logger.info(f"가격 데이터: {len(df)} 행, {df['date'].nunique()} 일, {df['ticker'].nunique()} 종목")
        return df
    
    def _load_technical_indicators(self, conn, stocks_df: pd.DataFrame,
                                   start_date: str, end_date: str) -> pd.DataFrame:
        """기술적 지표 로드"""
        stock_ids = stocks_df['id'].tolist()
        stock_id_list = ', '.join(map(str, stock_ids))
        
        query = f"""
            SELECT 
                t.stock_id,
                t.date,
                t.sma_20,
                t.sma_50,
                t.sma_200,
                t.rsi_14,
                t.macd,
                t.macd_signal,
                t.bb_upper,
                t.bb_middle,
                t.bb_lower,
                t.atr_14,
                t.volume_sma_20,
                t.volatility_20d
            FROM technical_indicators t
            WHERE t.stock_id IN ({stock_id_list})
                AND t.date >= '{start_date}'
                AND t.date <= '{end_date}'
            ORDER BY t.date, t.stock_id;
        """
        
        df = pd.read_sql(query, conn)
        
        # stock_id를 ticker로 매핑
        id_to_ticker = dict(zip(stocks_df['id'], stocks_df['ticker']))
        df['ticker'] = df['stock_id'].map(id_to_ticker)
        
        logger.info(f"기술적 지표: {len(df)} 행")
        return df
    
    def _convert_to_panel(self, price_df: pd.DataFrame, 
                         stocks_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Long 형태를 Panel 형태로 변환
        
        Returns:
            {'close': DataFrame, 'open': DataFrame, ...}
            각 DataFrame은 index=date, columns=ticker
        """
        panel_data = {}
        
        # 기본 가격 필드들
        price_fields = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
        
        for field in price_fields:
            if field in price_df.columns:
                pivot = price_df.pivot(index='date', columns='ticker', values=field)
                pivot.index = pd.to_datetime(pivot.index)
                panel_data[field] = pivot
        
        # VWAP 계산 (근사: (high + low + close) / 3)
        if all(k in panel_data for k in ['high', 'low', 'close']):
            panel_data['vwap'] = (
                panel_data['high'] + 
                panel_data['low'] + 
                panel_data['close']
            ) / 3
        
        logger.info(f"Panel 데이터 생성: {list(panel_data.keys())}")
        return panel_data
    
    def _convert_technical_to_panel(self, tech_df: pd.DataFrame,
                                    stocks_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """기술적 지표를 Panel 형태로 변환"""
        panel_data = {}
        
        tech_fields = [
            'sma_20', 'sma_50', 'sma_200', 'rsi_14',
            'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
            'atr_14', 'volume_sma_20', 'volatility_20d'
        ]
        
        for field in tech_fields:
            if field in tech_df.columns:
                pivot = tech_df.pivot(index='date', columns='ticker', values=field)
                pivot.index = pd.to_datetime(pivot.index)
                panel_data[field] = pivot
        
        return panel_data
    
    def _load_supply_demand_data(self, conn, stocks_df: pd.DataFrame,
                                  start_date: str, end_date: str) -> pd.DataFrame:
        """
        수급 데이터 로드 (외국인/기관/개인 순매수, 외국인 보유비율)
        
        Args:
            conn: DB 연결
            stocks_df: 종목 정보 DataFrame
            start_date: 시작일
            end_date: 종료일
            
        Returns:
            수급 데이터 DataFrame
        """
        stock_ids = stocks_df['id'].tolist()
        stock_id_list = ', '.join(map(str, stock_ids))
        
        query = f"""
            SELECT 
                sd.stock_id,
                sd.date,
                sd.foreign_net_buy,
                sd.institution_net_buy,
                sd.individual_net_buy,
                sd.foreign_ownership
            FROM supply_demand_data sd
            WHERE sd.stock_id IN ({stock_id_list})
                AND sd.date >= '{start_date}'
                AND sd.date <= '{end_date}'
            ORDER BY sd.date, sd.stock_id;
        """
        
        df = pd.read_sql(query, conn)
        
        # stock_id를 ticker로 매핑
        id_to_ticker = dict(zip(stocks_df['id'], stocks_df['ticker']))
        df['ticker'] = df['stock_id'].map(id_to_ticker)
        
        logger.info(f"수급 데이터: {len(df)} 행, {df['date'].nunique()} 일")
        return df
    
    def _convert_supply_demand_to_panel(self, sd_df: pd.DataFrame,
                                         stocks_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        수급 데이터를 Panel 형태로 변환
        
        Returns:
            Dict with keys:
            - 'foreign_net': 외국인 순매수
            - 'inst_net': 기관 순매수
            - 'indiv_net': 개인 순매수
            - 'foreign_ownership': 외국인 보유비율
        """
        panel_data = {}
        
        # 컬럼 매핑 (DB 컬럼명 -> 짧은 이름)
        field_mapping = {
            'foreign_net_buy': 'foreign_net',
            'institution_net_buy': 'inst_net',
            'individual_net_buy': 'indiv_net',
            'foreign_ownership': 'foreign_ownership'
        }
        
        for db_field, panel_field in field_mapping.items():
            if db_field in sd_df.columns:
                pivot = sd_df.pivot(index='date', columns='ticker', values=db_field)
                pivot.index = pd.to_datetime(pivot.index)
                panel_data[panel_field] = pivot
        
        return panel_data
    
    def get_universe_by_index(self, index_name: str = "KOSPI200") -> List[str]:
        """
        인덱스별 종목 리스트 조회
        
        Args:
            index_name: KOSPI200, KOSDAQ150 등
            
        Returns:
            종목 코드 리스트
        """
        conn = self._get_connection()
        
        try:
            query = f"""
                SELECT ticker
                FROM stocks
                WHERE is_active = true
                    AND index_membership LIKE '%{index_name}%'
                ORDER BY ticker;
            """
            
            df = pd.read_sql(query, conn)
            tickers = df['ticker'].tolist()
            
            logger.info(f"{index_name} 종목 수: {len(tickers)}")
            return tickers
            
        finally:
            conn.close()
    
    def get_stock_info(self, ticker: str) -> Dict:
        """종목 상세 정보 조회"""
        conn = self._get_connection()
        
        try:
            query = f"""
                SELECT *
                FROM stocks
                WHERE ticker = '{ticker}'
                LIMIT 1;
            """
            
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(query)
            result = cur.fetchone()
            
            return dict(result) if result else None
            
        finally:
            conn.close()
    
    def get_latest_date(self) -> str:
        """가장 최근 데이터 날짜 조회"""
        conn = self._get_connection()
        
        try:
            query = "SELECT MAX(date) FROM price_data;"
            cur = conn.cursor()
            cur.execute(query)
            latest_date = cur.fetchone()[0]
            
            return latest_date.strftime('%Y-%m-%d') if latest_date else None
            
        finally:
            conn.close()


# 편의 함수
def load_krx_data_from_postgres(
    start_date: str = None,
    end_date: str = None,
    universe: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, pd.DataFrame]:
    """
    PostgreSQL에서 KRX 데이터 로드 (편의 함수)
    
    Args:
        start_date: 시작일
        end_date: 종료일
        universe: 종목 리스트 또는 'KOSPI200', 'KOSDAQ150' 같은 인덱스명
        
    Returns:
        Panel 데이터 딕셔너리
    """
    loader = PostgresDataLoader(**kwargs)
    
    # universe가 인덱스명인 경우
    if isinstance(universe, str):
        universe = loader.get_universe_by_index(universe)
    
    return loader.load_data(
        start_date=start_date,
        end_date=end_date,
        universe=universe
    )


if __name__ == "__main__":
    # 테스트
    loader = PostgresDataLoader()
    
    # 최근 날짜 확인
    latest = loader.get_latest_date()
    print(f"최근 데이터: {latest}")
    
    # KOSPI200 종목 조회
    kospi200 = loader.get_universe_by_index("KOSPI200")
    print(f"KOSPI200 종목 수: {len(kospi200)}")
    
    # 샘플 데이터 로드
    data = loader.load_data(
        start_date="2025-01-01",
        end_date="2025-02-11",
        universe=kospi200[:10],  # 처음 10개만
        include_technical=True
    )
    
    print("\n로드된 데이터:")
    for key, df in data.items():
        print(f"  {key}: {df.shape}")
    
    print("\n종가 샘플:")
    print(data['close'].tail())
