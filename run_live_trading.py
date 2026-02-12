#!/usr/bin/env python3
"""
Alpha-GPT 실전 매매 시스템
⚠️ 주의: 실제 돈으로 거래됩니다!
"""

import os
from dotenv import load_dotenv
from alpha_gpt_kr.trading.kis_api import KISApi
from alpha_gpt_kr.trading.trader import AlphaTrader
from alpha_gpt_kr.core import AlphaGPT
from alpha_gpt_kr.data.postgres_loader import PostgresDataLoader
from loguru import logger

load_dotenv()

# ==========================================
# 🔧 설정 (여기를 수정하세요)
# ==========================================

# 리스크 관리
MAX_STOCKS = 5           # 최대 보유 종목 수 (작게 시작!)
REBALANCE_DAYS = 5       # 리밸런싱 주기 (영업일)
STOP_LOSS_PCT = -0.05    # 손절매 -5%
TAKE_PROFIT_PCT = 0.10   # 익절 +10%

# 최대 투자 금액 제한 (선택)
MAX_INVESTMENT = None    # 예: 1000000 = 100만원 제한, None = 제한없음

# 알파 설정
ALPHA_EXPRESSION = "ops.ts_delta(close, 26)"  # GP 진화 최고 알파

# ==========================================

def get_alpha_gpt():
    """AlphaGPT 인스턴스 생성"""
    logger.info("데이터 로딩 중...")
    
    db_url = os.getenv("DATABASE_URL", "postgresql://yrbahn:1234@192.168.0.248:5432/marketsense")
    loader = PostgresDataLoader(db_url)
    
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
    
    logger.info(f"알파 팩터 계산 중: {ALPHA_EXPRESSION}")
    alpha_gpt.last_alpha_values = alpha_gpt.alpha_miner.calculate_alpha(ALPHA_EXPRESSION)
    
    return alpha_gpt


def main():
    print("\n" + "=" * 60)
    print("⚠️  Alpha-GPT 실전 매매 시스템")
    print("=" * 60)
    print(f"\n설정:")
    print(f"  - 최대 종목 수: {MAX_STOCKS}개")
    print(f"  - 리밸런싱: {REBALANCE_DAYS}영업일마다")
    print(f"  - 손절매: {STOP_LOSS_PCT:.1%}")
    print(f"  - 익절: {TAKE_PROFIT_PCT:.1%}")
    if MAX_INVESTMENT:
        print(f"  - 최대 투자금: {MAX_INVESTMENT:,}원")
    print(f"  - 알파: {ALPHA_EXPRESSION}")
    
    print("\n⚠️  경고: 실제 계좌로 거래됩니다!")
    confirm = input("\n계속하시겠습니까? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("취소되었습니다.")
        return
    
    # KIS API 초기화 (실전투자)
    logger.info("KIS API 연결 중...")
    api = KISApi(
        app_key=os.getenv('KIS_APP_KEY'),
        app_secret=os.getenv('KIS_APP_SECRET'),
        account_no=os.getenv('KIS_ACCOUNT_NO'),
        is_real=True  # ⚠️ 실전투자
    )
    
    # 계좌 정보 확인
    logger.info("\n현재 계좌 상태:")
    balance = api.get_balance()
    logger.info(f"총평가금액: {int(balance['tot_evlu_amt']):,}원")
    logger.info(f"예수금: {int(balance['dnca_tot_amt']):,}원")
    
    # 현재 보유 종목
    holdings = api.get_holdings()
    if holdings:
        logger.info(f"\n보유 종목: {len(holdings)}개")
        for h in holdings[:5]:
            logger.info(f"  {h['prdt_name']}: {h['hldg_qty']}주, 수익률 {h['evlu_pfls_rt']}%")
    else:
        logger.info("\n보유 종목 없음")
    
    # 최종 확인
    print("\n" + "=" * 60)
    print("다음 작업을 선택하세요:")
    print("=" * 60)
    print("1. 알파 신호만 확인 (주문 없음)")
    print("2. 리밸런싱 실행 (실제 주문 발생!)")
    print("3. 일일 체크 (손절/익절만)")
    print("0. 취소")
    
    choice = input("\n선택 (0-3): ").strip()
    
    if choice == '0':
        print("취소되었습니다.")
        return
    
    # AlphaGPT 초기화
    alpha_gpt = get_alpha_gpt()
    
    # AlphaTrader 초기화
    trader = AlphaTrader(
        kis_api=api,
        alpha_gpt=alpha_gpt,
        max_stocks=MAX_STOCKS,
        rebalance_days=REBALANCE_DAYS,
        stop_loss_pct=STOP_LOSS_PCT,
        take_profit_pct=TAKE_PROFIT_PCT
    )
    
    if choice == '1':
        # 신호만 확인
        logger.info("\n알파 매수 신호:")
        signals = trader.generate_alpha_signals()
        print(signals)
        
    elif choice == '2':
        # 리밸런싱 실행
        confirm2 = input("\n⚠️ 실제 주문이 발생합니다. 정말 실행하시겠습니까? (yes/no): ").strip().lower()
        if confirm2 == 'yes':
            trader.rebalance_portfolio(force=True)
        else:
            print("취소되었습니다.")
            
    elif choice == '3':
        # 일일 체크
        trader.run_daily_check()
    
    print("\n" + "=" * 60)
    print("✅ 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()
