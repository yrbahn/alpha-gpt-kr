#!/usr/bin/env python3
"""
한국투자증권 KIS API 실전 매매 테스트
"""

import os
from dotenv import load_dotenv
from alpha_gpt_kr.trading.kis_api import KISApi
from alpha_gpt_kr.trading.trader import AlphaTrader
from alpha_gpt_kr.core import AlphaGPT
from alpha_gpt_kr.data.postgres_loader import PostgresDataLoader
from loguru import logger

# 환경변수 로드
load_dotenv()

def test_kis_api():
    """KIS API 기본 테스트"""
    logger.info("=" * 60)
    logger.info("한국투자증권 KIS API 테스트")
    logger.info("=" * 60)
    
    # API 클라이언트 초기화
    api = KISApi(
        app_key=os.getenv("KIS_APP_KEY"),
        app_secret=os.getenv("KIS_APP_SECRET"),
        account_no=os.getenv("KIS_ACCOUNT_NO"),
        is_real=False  # 모의투자로 테스트
    )
    
    # 1. 계좌 잔고 조회
    logger.info("\n[1] 계좌 잔고 조회")
    balance = api.get_balance()
    logger.info(f"총평가금액: {int(balance['tot_evlu_amt']):,}원")
    logger.info(f"예수금: {int(balance['dnca_tot_amt']):,}원")
    logger.info(f"평가손익: {int(balance['evlu_pfls_smtl_amt']):,}원")
    
    # 2. 보유 종목 조회
    logger.info("\n[2] 보유 종목 조회")
    holdings = api.get_holdings()
    
    if holdings:
        for h in holdings[:5]:  # 상위 5개만
            logger.info(
                f"  {h['prdt_name']} ({h['pdno']}): "
                f"{h['hldg_qty']}주, "
                f"수익률 {h['evlu_pfls_rt']}%"
            )
    else:
        logger.info("  보유 종목 없음")
    
    # 3. 현재가 조회 테스트
    logger.info("\n[3] 현재가 조회 (삼성전자)")
    price_info = api.get_current_price("005930")
    logger.info(f"  현재가: {int(price_info['stck_prpr']):,}원")
    logger.info(f"  전일대비: {price_info['prdy_vrss']} ({price_info['prdy_ctrt']}%)")
    
    logger.info("\n✅ KIS API 테스트 완료")


def test_alpha_trading():
    """Alpha-GPT 기반 자동 매매 테스트"""
    logger.info("=" * 60)
    logger.info("Alpha-GPT 자동 매매 시스템 테스트")
    logger.info("=" * 60)
    
    # 1. KIS API 초기화
    api = KISApi(
        app_key=os.getenv("KIS_APP_KEY"),
        app_secret=os.getenv("KIS_APP_SECRET"),
        account_no=os.getenv("KIS_ACCOUNT_NO"),
        is_real=False  # 모의투자
    )
    
    # 2. 데이터 로더 초기화
    db_url = os.getenv("DATABASE_URL", "postgresql://yrbahn:1234@192.168.0.248:5432/marketsense")
    loader = PostgresDataLoader(db_url)
    
    # 3. AlphaGPT 초기화
    logger.info("\n데이터 로딩 중...")
    data = loader.load_data(
        start_date="2024-01-01",
        end_date="2025-02-11",
        top_n=500
    )
    
    alpha_gpt = AlphaGPT(
        price_data=data['close'],
        volume_data=data['volume'],
        model_name="gpt-4o-mini"
    )
    
    # 간단한 알파 생성 (테스트용)
    logger.info("\n알파 팩터 생성 중...")
    alpha_expression = "ops.ts_delta(close, 20)"  # 20일 모멘텀
    alpha_gpt.last_alpha_values = alpha_gpt.alpha_miner.calculate_alpha(alpha_expression)
    
    # 4. AlphaTrader 초기화
    trader = AlphaTrader(
        kis_api=api,
        alpha_gpt=alpha_gpt,
        max_stocks=10,  # 최대 10종목
        rebalance_days=5,  # 5영업일마다 리밸런싱
        stop_loss_pct=-0.05,  # -5% 손절
        take_profit_pct=0.10  # +10% 익절
    )
    
    # 5. 현재 포트폴리오 확인
    logger.info("\n현재 포트폴리오:")
    portfolio = trader.get_current_portfolio()
    if not portfolio.empty:
        print(portfolio[['ticker', 'name', 'qty', 'profit_rate']])
    else:
        logger.info("  보유 종목 없음")
    
    # 6. 알파 신호 생성
    logger.info("\n알파 매수 신호:")
    signals = trader.generate_alpha_signals()
    print(signals.head(10))
    
    # 7. 리밸런싱 실행 (주의: 실제 주문 발생!)
    # trader.rebalance_portfolio(force=True)
    logger.info("\n⚠️ 리밸런싱은 주석 처리됨 (테스트 모드)")
    logger.info("실제 실행하려면 test_kis_trading.py에서 주석 해제")
    
    logger.info("\n✅ Alpha-GPT 매매 시스템 테스트 완료")


def main():
    print("\n" + "=" * 60)
    print("한국투자증권 KIS API + Alpha-GPT 실전 매매 시스템")
    print("=" * 60)
    print("\n선택:")
    print("1. KIS API 기본 테스트")
    print("2. Alpha-GPT 자동 매매 테스트 (모의투자)")
    print("3. 전체 테스트")
    
    choice = input("\n선택 (1/2/3): ").strip()
    
    if choice == "1":
        test_kis_api()
    elif choice == "2":
        test_alpha_trading()
    elif choice == "3":
        test_kis_api()
        print("\n")
        test_alpha_trading()
    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()
