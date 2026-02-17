#!/usr/bin/env python3
"""
KOSDAQ 200 알파 트레이딩 (v4 - Combined Alpha)
- 알파: 저변동성 3종 결합 (ATR + Volume + HL Range)
- Test IC: 0.1376 (최고 성능)
- 리밸런싱: 월간 (20영업일)
- 종목수: 3개
"""

import sys
import os
from pathlib import Path
from datetime import datetime, date
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import psycopg2

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.data.postgres_loader import PostgresDataLoader
from alpha_gpt_kr.mining.operators import AlphaOperators as ops
from alpha_gpt_kr.trading.kis_api import KISApi

load_dotenv()

# 설정
TOP_N = 3  # 상위 3개 종목
EXCLUDE_TICKERS = ['042700', '005690', '058470']  # 제외 종목 (한미반도체, 파미셀, 리노공업)


def get_kosdaq_200():
    """KOSDAQ 시총 상위 200개"""
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )
    
    exclude_sql = ','.join([f"'{t}'" for t in EXCLUDE_TICKERS])
    query = f"""
        SELECT ticker, name, market_cap
        FROM stocks
        WHERE is_active = true
          AND index_membership = 'KOSDAQ'
          AND market_cap IS NOT NULL
          AND ticker NOT IN ({exclude_sql})
        ORDER BY market_cap DESC
        LIMIT 200
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def compute_alpha(data):
    """
    Combined Alpha (v4): 저변동성 3종 결합
    - Test IC: 0.1376 (최고 성능)
    - 핵심 인사이트: KOSDAQ은 저변동성 선호 (coiled spring)
    
    구성:
    1. ATR 변동성 (60일 std → 15일 평균): 낮을수록 좋음
    2. 거래량 변동성 (75일 std): 낮을수록 좋음  
    3. 고저 범위 (120일 평균): 낮을수록 좋음
    """
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    
    # ATR ratio 계산
    atr_ratio = (high - low) / close
    high_low_range = (high - low) / close
    
    # ── 저변동성 3종 ──
    # 1. ATR 변동성: neg(ts_mean(ts_std(atr_ratio, 60), 15))
    lv1 = ops.neg(ops.ts_mean(ops.ts_std(atr_ratio, 60), 15))
    
    # 2. 거래량 변동성: neg(ts_std(volume, 75))
    lv2 = ops.neg(ops.ts_std(volume, 75))
    
    # 3. 고저 범위: neg(ts_mean(high_low_range, 120))
    lv3 = ops.neg(ops.ts_mean(high_low_range, 120))
    
    # ── Combined Alpha (z-score 정규화 후 합산) ──
    alpha = ops.add(
        ops.add(ops.zscore_scale(lv1), ops.zscore_scale(lv2)),
        ops.zscore_scale(lv3)
    )
    
    return alpha


def get_top_stocks(alpha, top_n=10):
    """최신 알파 기준 상위 종목 선택"""
    latest_date = alpha.index[-1]
    scores = alpha.loc[latest_date].dropna().sort_values(ascending=False)
    return scores.head(top_n)


def main():
    print("=" * 70)
    print(f"🚀 KOSDAQ 200 알파 트레이딩 (v4 Combined) - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    # 1. KOSDAQ 200 종목 로드
    print("\n📊 1. KOSDAQ 200 종목 로드...")
    stocks_df = get_kosdaq_200()
    tickers = stocks_df['ticker'].tolist()
    print(f"  ✅ {len(tickers)}개 종목")
    
    # 2. 데이터 로드
    print("\n📊 2. 가격 데이터 로드...")
    loader = PostgresDataLoader()
    data = loader.load_data(
        universe=tickers,
        start_date="2023-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
        include_supply_demand=False  # 저변동성 알파는 수급 불필요
    )
    print(f"  ✅ {len(data['close'])}일 데이터")
    
    # 3. 알파 계산
    print("\n📊 3. Combined Alpha 계산 (저변동성 3종)...")
    alpha = compute_alpha(data)
    
    # 4. 상위 종목 선택
    print(f"\n📊 4. 상위 {TOP_N}개 종목 선택...")
    top_stocks = get_top_stocks(alpha, TOP_N)
    
    print("\n" + "=" * 70)
    print("🏆 추천 종목 (알파 순위) - IC 0.1376")
    print("=" * 70)
    
    for i, (ticker, score) in enumerate(top_stocks.items(), 1):
        stock_info = stocks_df[stocks_df['ticker'] == ticker]
        if len(stock_info) > 0:
            name = stock_info.iloc[0]['name']
            price = data['close'].loc[data['close'].index[-1], ticker]
            print(f"  {i:2d}. {ticker} {name:20s} | 알파: {score:+.4f} | 현재가: {price:,.0f}원")
        else:
            print(f"  {i:2d}. {ticker} | 알파: {score:+.4f}")
    
    # 5. 매매 실행 여부
    print("\n" + "=" * 70)
    print("⚠️  매매를 실행하려면 --execute 옵션을 추가하세요")
    print("    예: python trade_kosdaq200_alpha.py --execute")
    print("=" * 70)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--execute':
        execute_trades(stocks_df, top_stocks, data)
    
    return top_stocks


def execute_trades(stocks_df, top_stocks, data):
    """실제 매매 실행"""
    print("\n" + "=" * 70)
    print("🔴 매매 실행 시작")
    print("=" * 70)
    
    kis = KISApi()
    
    # 1. 현재 보유 종목 확인
    print("\n📊 현재 보유 종목 확인...")
    balance = kis.get_balance()
    holdings = balance.get('holdings', [])
    
    if holdings:
        print("  현재 보유:")
        for h in holdings:
            print(f"    - {h['ticker']} {h['name']}: {h['qty']}주 @ {h['avg_price']:,.0f}원")
    else:
        print("  보유 종목 없음")
    
    # 2. 매도 (기존 보유 종목 중 추천에서 빠진 것)
    target_tickers = list(top_stocks.index)
    for h in holdings:
        if h['ticker'] not in target_tickers and h['qty'] > 0:
            print(f"\n📤 매도: {h['ticker']} {h['name']} {h['qty']}주")
            try:
                result = kis.sell_stock(h['ticker'], h['qty'])
                print(f"  ✅ 매도 주문 완료: {result}")
            except Exception as e:
                print(f"  ❌ 매도 실패: {e}")
    
    # 3. 매수 가능 금액 확인
    cash = balance.get('available_cash', 0)
    print(f"\n💰 매수 가능 금액: {cash:,.0f}원")
    
    # 4. 균등 배분 매수
    if cash > 0:
        per_stock = cash // len(target_tickers)
        print(f"  종목당 배분: {per_stock:,.0f}원")
        
        for ticker in target_tickers:
            # 이미 보유 중인지 확인
            already_held = any(h['ticker'] == ticker for h in holdings)
            if already_held:
                print(f"\n⏭️  {ticker}: 이미 보유 중 - 스킵")
                continue
            
            # 현재가 조회
            price = data['close'].loc[data['close'].index[-1], ticker]
            qty = int(per_stock // price)
            
            if qty > 0:
                stock_info = stocks_df[stocks_df['ticker'] == ticker]
                name = stock_info.iloc[0]['name'] if len(stock_info) > 0 else ticker
                print(f"\n📥 매수: {ticker} {name} {qty}주 @ {price:,.0f}원")
                try:
                    result = kis.buy_stock(ticker, qty)
                    print(f"  ✅ 매수 주문 완료: {result}")
                except Exception as e:
                    print(f"  ❌ 매수 실패: {e}")
            else:
                print(f"\n⚠️  {ticker}: 매수 가능 수량 0주 (가격: {price:,.0f}원)")
    
    print("\n" + "=" * 70)
    print("✅ 매매 실행 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
