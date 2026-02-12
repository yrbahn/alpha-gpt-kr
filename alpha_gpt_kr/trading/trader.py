"""
Alpha-GPT 기반 자동 매매 시스템
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger

from .kis_api import KISApi
from ..core import AlphaGPT
from ..data.postgres_loader import PostgresDataLoader


class AlphaTrader:
    """Alpha-GPT 기반 자동 트레이더"""
    
    def __init__(
        self,
        kis_api: KISApi,
        alpha_gpt: AlphaGPT,
        max_stocks: int = 10,
        rebalance_days: int = 5,
        stop_loss_pct: float = -0.05,
        take_profit_pct: float = 0.10
    ):
        """
        Args:
            kis_api: KIS API 클라이언트
            alpha_gpt: AlphaGPT 인스턴스
            max_stocks: 최대 보유 종목 수
            rebalance_days: 리밸런싱 주기 (영업일)
            stop_loss_pct: 손절매 비율 (예: -0.05 = -5%)
            take_profit_pct: 익절 비율 (예: 0.10 = 10%)
        """
        self.api = kis_api
        self.alpha_gpt = alpha_gpt
        self.max_stocks = max_stocks
        self.rebalance_days = rebalance_days
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        self.last_rebalance_date = None
        
        logger.info(f"AlphaTrader 초기화: max_stocks={max_stocks}, rebalance_days={rebalance_days}")
    
    def get_current_portfolio(self) -> pd.DataFrame:
        """현재 포트폴리오 조회"""
        holdings = self.api.get_holdings()
        
        if not holdings:
            logger.info("보유 종목 없음")
            return pd.DataFrame()
        
        portfolio = []
        for h in holdings:
            portfolio.append({
                'ticker': h['pdno'],  # 종목코드
                'name': h['prdt_name'],  # 종목명
                'qty': int(h['hldg_qty']),  # 보유수량
                'avg_price': float(h['pchs_avg_pric']),  # 매입평균가
                'current_price': float(h['prpr']),  # 현재가
                'eval_amt': int(h['evlu_amt']),  # 평가금액
                'profit_loss': int(h['evlu_pfls_amt']),  # 평가손익
                'profit_rate': float(h['evlu_pfls_rt'])  # 수익률
            })
        
        df = pd.DataFrame(portfolio)
        logger.info(f"현재 포트폴리오: {len(df)}개 종목")
        return df
    
    def get_account_balance(self) -> Dict:
        """계좌 잔고 조회"""
        balance = self.api.get_balance()
        
        info = {
            'total_assets': int(balance.get('tot_evlu_amt', balance.get('dnca_tot_amt', 0))),  # 총평가금액
            'total_profit': int(balance.get('evlu_pfls_smtl_amt', 0)),  # 평가손익합계
            'cash': int(balance.get('dnca_tot_amt', 0)),  # 예수금총액
            'buyable_cash': int(balance.get('ord_psbl_cash', balance.get('dnca_tot_amt', 0)))  # 주문가능현금
        }
        
        logger.info(f"총자산: {info['total_assets']:,}원, 예수금: {info['cash']:,}원")
        return info
    
    def generate_alpha_signals(self, top_n: int = None) -> pd.DataFrame:
        """알파 팩터 기반 매매 신호 생성"""
        if top_n is None:
            top_n = self.max_stocks
        
        # 최근 데이터로 알파 계산
        logger.info("알파 팩터 계산 중...")
        
        # AlphaGPT의 마지막 알파 사용
        if not hasattr(self.alpha_gpt, 'last_alpha_values'):
            raise Exception("알파 팩터가 생성되지 않았습니다. 먼저 run_evolution()을 실행하세요.")
        
        alpha_values = self.alpha_gpt.last_alpha_values
        
        # 최신 날짜의 알파 값
        latest_date = alpha_values.index[-1]
        signals = alpha_values.loc[latest_date].sort_values(ascending=False)
        
        # 상위 종목 선택
        top_stocks = signals.head(top_n)
        
        result = pd.DataFrame({
            'ticker': top_stocks.index,
            'alpha_score': top_stocks.values,
            'target_weight': 1.0 / top_n  # 동일 비중
        })
        
        logger.info(f"✅ 매수 신호: {len(result)}개 종목")
        return result
    
    def check_risk_management(self, portfolio: pd.DataFrame) -> List[str]:
        """리스크 관리 - 손절/익절 체크"""
        to_sell = []
        
        for _, row in portfolio.iterrows():
            profit_rate = row['profit_rate'] / 100  # % -> 비율
            
            # 손절매
            if profit_rate <= self.stop_loss_pct:
                logger.warning(f"🔴 손절매: {row['ticker']} ({profit_rate:.2%})")
                to_sell.append(row['ticker'])
            
            # 익절
            elif profit_rate >= self.take_profit_pct:
                logger.info(f"🟢 익절: {row['ticker']} ({profit_rate:.2%})")
                to_sell.append(row['ticker'])
        
        return to_sell
    
    def rebalance_portfolio(self, force: bool = False):
        """포트폴리오 리밸런싱"""
        today = datetime.now().date()
        
        # 리밸런싱 주기 체크
        if not force and self.last_rebalance_date:
            days_since = (today - self.last_rebalance_date).days
            if days_since < self.rebalance_days:
                logger.info(f"리밸런싱 주기 아님 ({days_since}/{self.rebalance_days}일)")
                return
        
        logger.info("=" * 60)
        logger.info("🔄 포트폴리오 리밸런싱 시작")
        logger.info("=" * 60)
        
        # 1. 현재 포트폴리오 & 잔고
        current_portfolio = self.get_current_portfolio()
        balance = self.get_account_balance()
        
        # 2. 리스크 관리 - 손절/익절
        if not current_portfolio.empty:
            to_sell_risk = self.check_risk_management(current_portfolio)
            
            for ticker in to_sell_risk:
                row = current_portfolio[current_portfolio['ticker'] == ticker].iloc[0]
                logger.info(f"매도 (리스크): {ticker} {row['qty']}주")
                self.api.sell_stock(ticker, row['qty'])
        
        # 3. 알파 신호 생성
        signals = self.generate_alpha_signals()
        target_tickers = set(signals['ticker'].tolist())
        current_tickers = set(current_portfolio['ticker'].tolist()) if not current_portfolio.empty else set()
        
        # 4. 매도 대상: 알파 신호에 없는 종목
        to_sell = current_tickers - target_tickers
        
        for ticker in to_sell:
            row = current_portfolio[current_portfolio['ticker'] == ticker].iloc[0]
            logger.info(f"매도 (리밸런싱): {ticker} {row['qty']}주")
            self.api.sell_stock(ticker, row['qty'])
        
        # 5. 매수 대상: 새로운 종목
        to_buy = target_tickers - current_tickers
        
        if to_buy:
            # 가용 자금 계산
            available_cash = balance['buyable_cash']
            cash_per_stock = available_cash / len(to_buy)
            
            for ticker in to_buy:
                # 현재가 조회
                price_info = self.api.get_current_price(ticker)
                current_price = int(price_info['stck_prpr'])
                
                # 매수 수량 계산
                qty = int(cash_per_stock / current_price)
                
                if qty > 0:
                    logger.info(f"매수: {ticker} {qty}주 @ {current_price:,}원")
                    self.api.buy_stock(ticker, qty)
        
        # 6. 리밸런싱 완료
        self.last_rebalance_date = today
        logger.info("=" * 60)
        logger.info("✅ 리밸런싱 완료")
        logger.info("=" * 60)
    
    def run_daily_check(self):
        """일일 체크 - 손절/익절만"""
        logger.info("📊 일일 체크 시작")
        
        portfolio = self.get_current_portfolio()
        
        if portfolio.empty:
            logger.info("보유 종목 없음")
            return
        
        # 리스크 관리
        to_sell = self.check_risk_management(portfolio)
        
        for ticker in to_sell:
            row = portfolio[portfolio['ticker'] == ticker].iloc[0]
            logger.info(f"매도: {ticker} {row['qty']}주")
            self.api.sell_stock(ticker, row['qty'])
        
        logger.info("✅ 일일 체크 완료")
