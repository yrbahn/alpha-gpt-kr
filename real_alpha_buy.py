#!/usr/bin/env python3
"""
진짜 Alpha-GPT 기반 포트폴리오 구성
- GP 진화 최고 알파 사용
- 500개 전체 종목 분석
- 알파 점수 상위 종목 자동 선택
"""

import os
import pandas as pd
from dotenv import load_dotenv
from alpha_gpt_kr.trading.kis_api import KISApi
from alpha_gpt_kr.core import AlphaGPT
from alpha_gpt_kr.data.postgres_loader import PostgresDataLoader
from loguru import logger

load_dotenv()

# GP 진화 최고 알파
BEST_ALPHA = "ops.ts_delta(close, 26)"

def main():
    print("=" * 60)
    print("🧠 진짜 Alpha-GPT 기반 포트폴리오")
    print("=" * 60)
    print(f"알파 팩터: {BEST_ALPHA}")
    print("GP 진화 결과: IC 0.0045, Sharpe 0.57")
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
    
    # 현재 보유 종목
    current_holdings = api.get_holdings()
    current_tickers = [h['pdno'] for h in current_holdings]
    
    print(f"예수금: {cash:,}원")
    print(f"현재 보유: {len(current_holdings)}개 종목")
    
    if len(current_holdings) > 0:
        print("\n⚠️  이미 보유 종목이 있습니다.")
        print("현재 포트폴리오:")
        for h in current_holdings:
            print(f"  - {h['prdt_name']} ({h['pdno']}): {h['hldg_qty']}주")
        
        choice = input("\n이미 매수했으므로 분석만 하시겠습니까? (yes/no): ").strip().lower()
        if choice != 'yes':
            print("취소되었습니다.")
            return
    
    # 데이터 로딩
    print("\n[1] PostgreSQL 데이터 로딩...")
    try:
        loader = PostgresDataLoader(
            host="192.168.0.248",
            port=5432,
            database="marketsense",
            user="yrbahn",
            password="1234"
        )
        
        data = loader.load_data(
            start_date="2024-01-01",
            end_date="2025-02-11",
            universe=None  # None = 전체 종목
        )
        
        print(f"    ✅ 데이터 로드 완료: {len(data['close'].columns)}개 종목")
        
    except Exception as e:
        logger.error(f"DB 연결 실패: {e}")
        print(f"\n❌ 데이터베이스 연결 실패")
        print("네트워크나 DB 서버를 확인해주세요.")
        return
    
    # 알파 계산
    print(f"\n[2] 알파 팩터 계산 중... ({BEST_ALPHA})")
    alpha_gpt = AlphaGPT(
        price_data=data['close'],
        volume_data=data['volume'],
        model_name="gpt-4o-mini"
    )
    
    alpha_values = alpha_gpt.alpha_miner.calculate_alpha(BEST_ALPHA)
    
    # 최신 날짜 신호
    latest_date = alpha_values.index[-1]
    signals = alpha_values.loc[latest_date].sort_values(ascending=False)
    
    print(f"    ✅ 알파 계산 완료: {latest_date}")
    print(f"\n    상위 20개 알파 점수:")
    for i, (ticker, score) in enumerate(signals.head(20).items(), 1):
        print(f"    {i:2d}. {ticker}: {score:+.4f}")
    
    # 실시간 가격 확인 및 필터링
    print("\n[3] 실시간 가격 확인 중...")
    
    affordable = []
    target_count = 15
    
    for ticker in signals.head(50).index:
        try:
            price_info = api.get_current_price(ticker)
            price = int(price_info['stck_prpr'])
            alpha_score = signals[ticker]
            
            # 예수금 또는 종목당 예산
            budget_per_stock = cash // target_count if cash > 0 else 500000
            
            if price <= budget_per_stock:
                qty = budget_per_stock // price
                cost = price * qty
                
                affordable.append({
                    'ticker': ticker,
                    'price': price,
                    'qty': qty,
                    'cost': cost,
                    'alpha_score': alpha_score
                })
                
                print(f"    ✓ {ticker}: {price:>8,}원 x {qty}주 = {cost:>9,}원 (알파: {alpha_score:+.4f})")
            
            if len(affordable) >= target_count:
                break
                
        except Exception as e:
            logger.warning(f"    ✗ {ticker}: {e}")
            continue
    
    if not affordable:
        print("\n⚠️  매수 가능한 종목이 없습니다.")
        return
    
    # 선택된 포트폴리오
    df = pd.DataFrame(affordable)
    df = df.sort_values('alpha_score', ascending=False)
    
    print("\n" + "=" * 60)
    print(f"✅ Alpha-GPT 선정 포트폴리오 ({len(df)}개)")
    print("=" * 60)
    
    total_cost = df['cost'].sum()
    
    for i, row in df.iterrows():
        rank = list(df.index).index(i) + 1
        print(f"\n{rank:2d}. {row['ticker']}")
        print(f"    가격: {row['price']:,}원 x {row['qty']}주 = {row['cost']:,}원")
        print(f"    ⭐ 알파 점수: {row['alpha_score']:+.4f}")
    
    print(f"\n총 투자금액: {total_cost:,}원")
    print(f"잔여 예수금: {cash - total_cost:,}원" if cash > 0 else "")
    
    # 비교 분석
    if len(current_holdings) > 0:
        print("\n" + "=" * 60)
        print("📊 현재 vs 알파 기반 포트폴리오 비교")
        print("=" * 60)
        
        current_set = set(current_tickers)
        alpha_set = set(df['ticker'].tolist())
        
        both = current_set & alpha_set
        only_current = current_set - alpha_set
        only_alpha = alpha_set - current_set
        
        print(f"\n공통 종목 ({len(both)}개):")
        for ticker in both:
            print(f"  ✓ {ticker}")
        
        print(f"\n현재만 보유 ({len(only_current)}개):")
        for ticker in only_current:
            alpha_score = signals.get(ticker, None)
            if alpha_score is not None:
                rank = list(signals.index).index(ticker) + 1
                print(f"  - {ticker} (알파 순위: {rank}위, 점수: {alpha_score:+.4f})")
            else:
                print(f"  - {ticker} (알파 데이터 없음)")
        
        print(f"\n알파 추천만 ({len(only_alpha)}개):")
        for ticker in only_alpha:
            alpha_score = signals[ticker]
            rank = list(signals.index).index(ticker) + 1
            print(f"  + {ticker} (알파 순위: {rank}위, 점수: {alpha_score:+.4f})")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
