#!/usr/bin/env python3
"""Best Alpha로 상위 5종목 선정"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.mining.operators import AlphaOperators as ops

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )

# 데이터 로드
print("📊 데이터 로드 중...")
conn = get_db_connection()

# 시총 상위 500개 종목 (ticker, name, market_cap)
stocks_df = pd.read_sql("""
    SELECT s.id, s.ticker, s.name, s.market_cap
    FROM stocks s
    WHERE s.is_active = true
    AND s.market_cap IS NOT NULL
    AND EXISTS (
        SELECT 1 FROM price_data p
        WHERE p.stock_id = s.id
        AND p.date >= CURRENT_DATE - INTERVAL '730 days'
        LIMIT 1
    )
    ORDER BY s.market_cap DESC
    LIMIT 500
""", conn)

stock_ids = stocks_df['id'].tolist()
stock_id_list = ', '.join(map(str, stock_ids))

# ticker → name 매핑
ticker_name = dict(zip(stocks_df['ticker'], stocks_df['name']))
ticker_mcap = dict(zip(stocks_df['ticker'], stocks_df['market_cap']))

price_df = pd.read_sql(f"""
    SELECT s.ticker, p.date, p.close, p.volume
    FROM price_data p
    JOIN stocks s ON p.stock_id = s.id
    WHERE p.stock_id IN ({stock_id_list})
    AND p.date >= CURRENT_DATE - INTERVAL '730 days'
    ORDER BY s.ticker, p.date
""", conn)
conn.close()

close = price_df.pivot(index='date', columns='ticker', values='close')
volume = price_df.pivot(index='date', columns='ticker', values='volume')
returns = close.pct_change()

print(f"✅ {len(close.columns)}개 종목, {len(close)}일 데이터")
print(f"   최신 날짜: {close.index[-1]}")

# Best Alpha 적용 (v3 Enhanced GP: Train IC 0.0389 / Test IC 0.0385)
best_alpha_expr = "ops.normed_rank(ops.cwise_mul(ops.cwise_mul(ops.ts_delta_ratio(close, 25), ops.div(ops.ts_median(volume, 10), ops.ts_std(volume, 15))), ops.ts_maxmin_scale(close, 28)))"

print(f"\n🧮 알파 계산 중...")
print(f"   Expression: {best_alpha_expr}")

alpha_values = eval(best_alpha_expr)

# 최신 날짜 기준 상위 5종목
latest_date = alpha_values.index[-1]
latest_scores = alpha_values.loc[latest_date].dropna().sort_values(ascending=False)

print(f"\n{'='*70}")
print(f"🏆 상위 5종목 (기준일: {latest_date})")
print(f"   Alpha: {best_alpha_expr}")
print(f"   Train IC: 0.0389 / Test IC: 0.0385 (15-day forward)")
print(f"{'='*70}")
print(f"{'순위':>4} {'종목코드':<10} {'종목명':<16} {'알파점수':>10} {'현재가':>12} {'시총(억)':>12}")
print(f"{'-'*70}")

for i, (ticker, score) in enumerate(latest_scores.head(5).items(), 1):
    name = ticker_name.get(ticker, '?')
    price = close.loc[latest_date, ticker]
    mcap = ticker_mcap.get(ticker, 0)
    mcap_억 = mcap / 1e8 if mcap else 0
    print(f"  {i:>2}. {ticker:<10} {name:<16} {score:>10.4f} {price:>12,.0f}원 {mcap_억:>10,.0f}억")

# 하위 5종목도 참고용으로 표시
print(f"\n{'='*70}")
print(f"📉 하위 5종목 (숏 후보)")
print(f"{'='*70}")
print(f"{'순위':>4} {'종목코드':<10} {'종목명':<16} {'알파점수':>10} {'현재가':>12} {'시총(억)':>12}")
print(f"{'-'*70}")

for i, (ticker, score) in enumerate(latest_scores.tail(5).items(), 1):
    name = ticker_name.get(ticker, '?')
    price = close.loc[latest_date, ticker]
    mcap = ticker_mcap.get(ticker, 0)
    mcap_억 = mcap / 1e8 if mcap else 0
    print(f"  {i:>2}. {ticker:<10} {name:<16} {score:>10.4f} {price:>12,.0f}원 {mcap_억:>10,.0f}억")

print(f"\n💡 해석:")
print(f"   - ts_delta_ratio(close, 25): 25일간 수익률 (모멘텀)")
print(f"   - ts_median(volume, 10) / ts_std(volume, 15): 거래량 안정성 (중앙값/변동성)")
print(f"   - ts_maxmin_scale(close, 28): 28일간 가격 레인지에서의 위치 [0,1]")
print(f"   - 가격 모멘텀 × 거래량 안정성 × 레인지 위치 = 다중 팩터 매수 신호")
print(f"   - 15영업일(약 3주) 보유 전략에 최적화")
