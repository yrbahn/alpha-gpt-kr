#!/usr/bin/env python3
"""
KOSPI 200 알파 트레이딩
- 알파: 낙폭과대 + 고변동성 + 아랫꼬리 (KOSPI 최적화)
- Test IC: 0.0884 (낙폭과대), 0.0688 (고변동성), 0.0487 (아랫꼬리)
- 리밸런싱: 월간 (20영업일)
- 종목수: 3개
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import psycopg2

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.mining.operators import AlphaOperators as ops
from alpha_gpt_kr.trading.kis_api import KISApi

load_dotenv()

# 설정
TOP_N = 3
EXCLUDE_TICKERS = []  # 제외 종목


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )


def get_kospi_200():
    """KOSPI 시총 상위 200개"""
    conn = get_db_connection()
    
    query = """
        SELECT ticker, name, market_cap
        FROM stocks
        WHERE is_active = true
          AND index_membership = 'KOSPI'
          AND market_cap IS NOT NULL
        ORDER BY market_cap DESC
        LIMIT 200
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    if EXCLUDE_TICKERS:
        df = df[~df['ticker'].isin(EXCLUDE_TICKERS)]
    
    return df


def load_data(tickers):
    """가격 및 수급 데이터 로드"""
    conn = get_db_connection()
    
    ticker_list = "', '".join(tickers)
    
    # 가격 데이터
    price_query = f"""
        SELECT s.ticker, p.date, p.open, p.high, p.low, p.close, p.volume
        FROM price_data p
        JOIN stocks s ON p.stock_id = s.id
        WHERE s.ticker IN ('{ticker_list}')
        AND p.date >= CURRENT_DATE - INTERVAL '365 days'
        ORDER BY s.ticker, p.date
    """
    price_df = pd.read_sql(price_query, conn)
    
    # 수급 데이터
    flow_query = f"""
        SELECT s.ticker, sd.date, sd.foreign_net_buy, sd.institution_net_buy
        FROM supply_demand_data sd
        JOIN stocks s ON sd.stock_id = s.id
        WHERE s.ticker IN ('{ticker_list}')
        AND sd.date >= CURRENT_DATE - INTERVAL '365 days'
    """
    flow_df = pd.read_sql(flow_query, conn)
    conn.close()
    
    # Pivot
    close = price_df.pivot(index='date', columns='ticker', values='close')
    high = price_df.pivot(index='date', columns='ticker', values='high')
    low = price_df.pivot(index='date', columns='ticker', values='low')
    open_price = price_df.pivot(index='date', columns='ticker', values='open')
    volume = price_df.pivot(index='date', columns='ticker', values='volume')
    
    foreign_net = flow_df.pivot(index='date', columns='ticker', values='foreign_net_buy')
    foreign_net = foreign_net.reindex(index=close.index, columns=close.columns).fillna(0)
    
    return {
        'close': close,
        'high': high,
        'low': low,
        'open': open_price,
        'volume': volume,
        'foreign_net': foreign_net,
    }


def compute_alpha(data):
    """
    KOSPI 최적화 알파:
    - 낙폭과대 (IC 0.0884): 최근 20일 최저 수익률의 반전
    - 고변동성 (IC 0.0688): ATR 높은 종목
    - 아랫꼬리 (IC 0.0487): 지지력 있는 종목
    """
    close = data['close']
    high = data['high']
    low = data['low']
    open_price = data['open']
    volume = data['volume']
    
    returns = close.pct_change()
    
    # 파생 지표
    high_low_range = (high - low) / close
    lower_shadow = (close.clip(upper=open_price) - low) / close
    
    # ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3]).groupby(level=0).max()
    true_range = true_range.reindex(close.index)
    atr_ratio = true_range / close
    
    # ── 1. 낙폭과대 (IC 0.0884) ──
    # 최근 20일 중 최저 수익률 → 음수가 클수록(많이 빠졌을수록) 반등 기대
    oversold = ops.zscore_scale(ops.neg(ops.ts_min(returns, 20)))
    
    # ── 2. 고변동성 (IC 0.0688) ──
    # ATR 변동성이 높은 종목
    high_vol = ops.zscore_scale(ops.ts_mean(ops.ts_std(atr_ratio, 60), 15))
    
    # ── 3. 아랫꼬리 (IC 0.0487) ──
    # 아랫꼬리가 긴 종목 = 지지력
    support = ops.zscore_scale(ops.ts_mean(lower_shadow, 20))
    
    # Combined Alpha (가중 합산)
    # 낙폭과대가 가장 효과적이므로 2배 가중
    combined_alpha = ops.add(
        ops.add(
            ops.cwise_mul(oversold, 2),  # 낙폭과대 2배
            high_vol
        ),
        support
    )
    
    return combined_alpha


def get_top_stocks(alpha, stocks_df, top_n=3):
    """최신 알파 기준 상위 종목"""
    latest_date = alpha.index[-1]
    scores = alpha.loc[latest_date].dropna().sort_values(ascending=False)
    
    result = []
    for ticker, score in scores.items():
        info = stocks_df[stocks_df['ticker'] == ticker]
        if len(info) > 0:
            result.append({
                'ticker': ticker,
                'name': info.iloc[0]['name'],
                'score': score,
            })
        if len(result) >= top_n:
            break
    
    return result


def main():
    print("=" * 70)
    print(f"🚀 KOSPI 200 알파 트레이딩 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("   알파: 낙폭과대 + 고변동성 + 아랫꼬리")
    print("=" * 70)
    
    # 1. KOSPI 200 종목 로드
    print("\n📊 1. KOSPI 200 종목 로드...")
    stocks_df = get_kospi_200()
    tickers = stocks_df['ticker'].tolist()
    print(f"  ✅ {len(tickers)}개 종목")
    
    # 2. 데이터 로드
    print("\n📊 2. 가격 데이터 로드...")
    data = load_data(tickers)
    print(f"  ✅ {len(data['close'])}일 데이터")
    
    # 3. 알파 계산
    print("\n📊 3. 알파 계산...")
    alpha = compute_alpha(data)
    
    # 4. 상위 종목 선택
    print(f"\n📊 4. 상위 {TOP_N}개 종목 선택...")
    top_stocks = get_top_stocks(alpha, stocks_df, TOP_N)
    
    print("\n" + "=" * 70)
    print("🏆 추천 종목 (알파 순위)")
    print("=" * 70)
    
    for i, stock in enumerate(top_stocks, 1):
        ticker = stock['ticker']
        price = data['close'].loc[data['close'].index[-1], ticker]
        print(f"  {i:2d}. {ticker} {stock['name']:20s} | 알파: {stock['score']:+.4f} | 현재가: {price:,.0f}원")
    
    # 5. 매매 실행
    print("\n" + "=" * 70)
    print("📈 매매 실행")
    print("=" * 70)
    
    try:
        kis = KISApi(
            app_key=os.getenv('KIS_APP_KEY'),
            app_secret=os.getenv('KIS_APP_SECRET'),
            account_no=os.getenv('KIS_ACCOUNT_NO')
        )
        
        balance = kis.get_balance()
        print(f"\n예수금: {balance:,.0f}원")
        
        if '--execute' in sys.argv:
            print("\n🔥 매매 실행 중...")
            # 실제 매매 로직
            buy_per_stock = balance // TOP_N
            for stock in top_stocks:
                ticker = stock['ticker']
                price = kis.get_current_price(ticker)
                qty = int(buy_per_stock / price)
                if qty > 0:
                    print(f"  📥 매수: {ticker} {stock['name']} {qty}주 @ {price:,.0f}원")
                    try:
                        kis.buy(ticker, qty)
                        print(f"      ✅ 완료")
                    except Exception as e:
                        print(f"      ❌ 실패: {e}")
        else:
            print("\n⚠️  테스트 모드 (실제 매매: --execute 옵션)")
    
    except Exception as e:
        print(f"\n⚠️  KIS API 오류: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 완료!")
    print("=" * 70)
    
    return top_stocks


if __name__ == "__main__":
    main()
