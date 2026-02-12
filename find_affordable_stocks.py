#!/usr/bin/env python3
"""
예수금 5만원으로 살 수 있는 알파 신호 종목 찾기
"""

import os
import sys
from dotenv import load_dotenv
from alpha_gpt_kr.trading.kis_api import KISApi
from alpha_gpt_kr.core import AlphaGPT
from alpha_gpt_kr.data.postgres_loader import PostgresDataLoader
from loguru import logger
import pandas as pd

# 환경변수 로드
load_dotenv()

def main():
    print("=" * 60)
    print("소액 테스트 - 알파 신호 생성 및 저가 종목 선택")
    print("=" * 60)
    
    # 1. 데이터 로딩
    print("\n[1] 데이터 로딩 중...")
    db_url = "postgresql://yrbahn:1234@192.168.0.248:5432/marketsense"
    loader = PostgresDataLoader(db_url)
    
    data = loader.load_data(
        start_date="2024-01-01",
        end_date="2025-02-11",
        top_n=500
    )
    
    # 2. 알파 팩터 계산
    print("\n[2] 알파 팩터 계산 중...")
    alpha_gpt = AlphaGPT(
        price_data=data['close'],
        volume_data=data['volume'],
        model_name="gpt-4o-mini"
    )
    
    # GP 진화 최고 알파 사용
    alpha_expression = "ops.ts_delta(close, 26)"
    print(f"   알파: {alpha_expression}")
    
    alpha_values = alpha_gpt.alpha_miner.calculate_alpha(alpha_expression)
    
    # 3. 최신 신호 (상위 50개)
    latest_date = alpha_values.index[-1]
    signals = alpha_values.loc[latest_date].sort_values(ascending=False).head(50)
    
    print(f"\n[3] 알파 매수 신호 생성:")
    print(f"   날짜: {latest_date}")
    print(f"   상위 50개 종목 선택")
    
    # 4. KIS API로 현재가 조회
    print("\n[4] 현재가 조회 중...")
    api = KISApi(
        app_key=os.getenv('KIS_APP_KEY'),
        app_secret=os.getenv('KIS_APP_SECRET'),
        account_no=os.getenv('KIS_ACCOUNT_NO'),
        is_real=True
    )
    
    affordable_stocks = []
    
    for i, ticker in enumerate(signals.index[:50]):
        try:
            price_info = api.get_current_price(ticker)
            current_price = int(price_info['stck_prpr'])
            alpha_score = signals[ticker]
            
            # 5만원 이하 종목만
            if current_price <= 50000:
                qty = 50000 // current_price
                affordable_stocks.append({
                    'ticker': ticker,
                    'price': current_price,
                    'alpha_score': alpha_score,
                    'max_qty': qty,
                    'total_cost': current_price * qty
                })
                print(f"   ✓ {ticker}: {current_price:,}원 (알파: {alpha_score:.4f})")
            
            # API 호출 제한 고려 (20개만)
            if i >= 20:
                break
                
        except Exception as e:
            logger.warning(f"   ✗ {ticker}: {e}")
            continue
    
    # 5. 결과 출력
    print("\n" + "=" * 60)
    print("예수금 5만원으로 살 수 있는 종목")
    print("=" * 60)
    
    if affordable_stocks:
        df = pd.DataFrame(affordable_stocks)
        df = df.sort_values('alpha_score', ascending=False)
        
        print(f"\n총 {len(df)}개 종목 발견:\n")
        print(df.to_string(index=False))
        
        print("\n" + "=" * 60)
        print("✅ 추천 종목 (알파 점수 최고)")
        print("=" * 60)
        
        top_pick = df.iloc[0]
        print(f"\n종목코드: {top_pick['ticker']}")
        print(f"현재가: {top_pick['price']:,}원")
        print(f"매수 가능 수량: {top_pick['max_qty']}주")
        print(f"총 매수 금액: {top_pick['total_cost']:,}원")
        print(f"알파 점수: {top_pick['alpha_score']:.4f}")
        
        print("\n⚠️ 이 종목을 매수하시겠습니까?")
        
        # 저장
        df.to_csv('affordable_stocks.csv', index=False)
        print(f"\n📊 결과 저장: affordable_stocks.csv")
        
    else:
        print("\n⚠️ 5만원 이하 종목이 없습니다.")
        print("대안:")
        print("1. 예수금 추가 입금")
        print("2. 모의투자 계좌 사용")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
