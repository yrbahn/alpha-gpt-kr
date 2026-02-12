#!/usr/bin/env python3
"""
Alpha-GPT 기반 실전 매수
- GP 진화 최고 알파 사용
- 예수금 내에서 최적 종목 선택
"""

import os
import pandas as pd
from dotenv import load_dotenv
from alpha_gpt_kr.trading.kis_api import KISApi
from loguru import logger

load_dotenv()

# GP 진화 최고 알파 (IC: 0.0045, Sharpe: 0.57)
BEST_ALPHA = "ops.ts_delta(close, 26)"

def get_alpha_signals_from_db():
    """PostgreSQL에서 알파 신호 생성"""
    from alpha_gpt_kr.core import AlphaGPT
    from alpha_gpt_kr.data.postgres_loader import PostgresDataLoader
    
    logger.info("데이터 로딩 중...")
    db_url = "postgresql://yrbahn:1234@192.168.0.248:5432/marketsense"
    loader = PostgresDataLoader(db_url)
    
    data = loader.load_data(
        start_date="2024-01-01",
        end_date="2025-02-11",
        top_n=500
    )
    
    logger.info(f"알파 팩터 계산 중: {BEST_ALPHA}")
    alpha_gpt = AlphaGPT(
        price_data=data['close'],
        volume_data=data['volume'],
        model_name="gpt-4o-mini"
    )
    
    alpha_values = alpha_gpt.alpha_miner.calculate_alpha(BEST_ALPHA)
    
    # 최신 날짜의 신호
    latest_date = alpha_values.index[-1]
    signals = alpha_values.loc[latest_date].sort_values(ascending=False)
    
    logger.info(f"✅ 알파 신호 생성: {latest_date}, 상위 종목 추출")
    return signals


def main():
    print("=" * 60)
    print("🧠 Alpha-GPT 기반 실전 매수")
    print("=" * 60)
    print(f"알파 팩터: {BEST_ALPHA}")
    print("(GP 진화 최고 성능: IC 0.0045, Sharpe 0.57)")
    print()
    
    # KIS API
    api = KISApi(
        app_key=os.getenv('KIS_APP_KEY'),
        app_secret=os.getenv('KIS_APP_SECRET'),
        account_no=os.getenv('KIS_ACCOUNT_NO'),
        is_real=True
    )
    
    # 계좌 확인
    balance = api.get_balance()
    cash = int(balance.get('dnca_tot_amt', 0))
    print(f"예수금: {cash:,}원\n")
    
    if cash < 10000:
        print("⚠️ 예수금 부족")
        return
    
    # 알파 신호 생성
    try:
        print("[1] 알파 신호 생성 중...")
        signals = get_alpha_signals_from_db()
        
        print(f"    상위 20개 종목 알파 점수:")
        for i, (ticker, score) in enumerate(signals.head(20).items(), 1):
            print(f"    {i:2d}. {ticker}: {score:.4f}")
        
    except Exception as e:
        logger.error(f"알파 생성 실패: {e}")
        print("\n⚠️ 데이터베이스 연결 실패")
        print("대안: 수동으로 종목 선택하거나 simple_test_trade.py 사용")
        return
    
    # 현재가 조회 및 매수 가능 종목 필터링
    print("\n[2] 예수금 내 매수 가능 종목 검색...")
    print("-" * 60)
    
    affordable = []
    
    for ticker in signals.head(50).index:
        try:
            price_info = api.get_current_price(ticker)
            price = int(price_info['stck_prpr'])
            alpha_score = signals[ticker]
            
            if price <= cash:
                qty = cash // price
                cost = price * qty
                
                affordable.append({
                    'ticker': ticker,
                    'price': price,
                    'qty': qty,
                    'cost': cost,
                    'alpha_score': alpha_score
                })
                
                print(f"✓ {ticker}: {price:>8,}원 x {qty}주 = {cost:>8,}원 (알파: {alpha_score:.4f})")
            
            if len(affordable) >= 5:
                break
                
        except Exception as e:
            continue
    
    if not affordable:
        print("\n⚠️ 알파 상위 종목 중 예수금으로 살 수 있는 종목이 없습니다.")
        print("대안:")
        print("1. 예수금 추가 입금")
        print("2. 알파 하위 종목 확장 검색")
        return
    
    # 추천 (알파 점수 최고)
    df = pd.DataFrame(affordable)
    df = df.sort_values('alpha_score', ascending=False)
    
    print("\n" + "=" * 60)
    print("✅ 알파 기반 추천 종목 (알파 점수 순)")
    print("=" * 60)
    
    for i, row in df.iterrows():
        rank = list(df.index).index(i) + 1
        print(f"\n{rank}. {row['ticker']}")
        print(f"   현재가: {row['price']:,}원")
        print(f"   매수 가능: {row['qty']}주 = {row['cost']:,}원")
        print(f"   ⭐ 알파 점수: {row['alpha_score']:.4f}")
    
    # 매수 진행
    print("\n" + "=" * 60)
    print("매수하시겠습니까?")
    print("=" * 60)
    
    choice = input("\n종목 번호 선택 (0=취소): ").strip()
    
    if choice == '0' or not choice.isdigit():
        print("취소되었습니다.")
        return
    
    idx = int(choice) - 1
    if idx < 0 or idx >= len(df):
        print("잘못된 선택입니다.")
        return
    
    selected = df.iloc[idx]
    
    print(f"\n선택: {selected['ticker']}")
    print(f"매수: {selected['qty']}주 @ {selected['price']:,}원")
    print(f"총액: {selected['cost']:,}원")
    print(f"알파 점수: {selected['alpha_score']:.4f}")
    
    confirm = input("\n⚠️ 실제 주문이 발생합니다. 계속하시겠습니까? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("취소되었습니다.")
        return
    
    # 실제 매수
    try:
        print("\n[주문 실행 중...]")
        result = api.buy_stock(
            ticker=selected['ticker'],
            qty=selected['qty'],
            order_type="01"  # 시장가
        )
        
        print("\n✅ 주문 완료!")
        print(f"주문번호: {result.get('ODNO', 'N/A')}")
        
        print("\n📊 주문 후 계좌:")
        balance2 = api.get_balance()
        print(f"예수금: {int(balance2.get('dnca_tot_amt', 0)):,}원")
        
        holdings = api.get_holdings()
        if holdings:
            for h in holdings:
                if h['pdno'] == selected['ticker']:
                    print(f"보유: {h['prdt_name']} {h['hldg_qty']}주")
        
    except Exception as e:
        print(f"\n❌ 주문 실패: {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
