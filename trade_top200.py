#!/usr/bin/env python3
"""
Alpha-GPT Top 200 Trading - 월간 리밸런싱
Best Alpha (IC 0.0830, IR 1.28): 기관 수급 추종
"""

import argparse
import pandas as pd
import numpy as np
import psycopg2
import os
from dotenv import load_dotenv
from alpha_gpt_kr.mining.operators import AlphaOperators as ops
from alpha_gpt_kr.trading.kis_api import KISApi

load_dotenv()

def get_top_stocks(top_n=5, exclude_tickers=None):
    """Top 200에서 Best Alpha 기준 상위 종목 선정"""
    exclude_tickers = exclude_tickers or []
    
    conn = psycopg2.connect(
        host='192.168.0.248', port=5432, 
        database='marketsense', user='yrbahn', password='1234'
    )
    
    # Top 200 종목
    query_stocks = '''
        SELECT s.id, s.ticker, s.name, s.market_cap
        FROM stocks s
        WHERE s.is_active = true AND s.market_cap IS NOT NULL
        ORDER BY s.market_cap DESC
        LIMIT 200
    '''
    stocks_df = pd.read_sql(query_stocks, conn)
    stock_ids = stocks_df['id'].tolist()
    
    # 가격 데이터
    query = f'''
        SELECT s.ticker, p.date, p.open, p.close, p.volume
        FROM price_data p
        JOIN stocks s ON p.stock_id = s.id
        WHERE p.stock_id IN ({','.join(map(str, stock_ids))})
        AND p.date >= CURRENT_DATE - INTERVAL '180 days'
        ORDER BY s.ticker, p.date
    '''
    price_df = pd.read_sql(query, conn)
    
    open_price = price_df.pivot(index='date', columns='ticker', values='open')
    close = price_df.pivot(index='date', columns='ticker', values='close')
    volume = price_df.pivot(index='date', columns='ticker', values='volume')
    gap = open_price / close.shift(1) - 1
    intraday_ret = close / open_price - 1
    
    # 수급 데이터
    flow_query = f'''
        SELECT s.ticker, sd.date, sd.institution_net_buy
        FROM supply_demand_data sd
        JOIN stocks s ON sd.stock_id = s.id
        WHERE sd.stock_id IN ({','.join(map(str, stock_ids))})
        AND sd.date >= CURRENT_DATE - INTERVAL '180 days'
    '''
    flow_df = pd.read_sql(flow_query, conn)
    inst_net = flow_df.pivot(index='date', columns='ticker', values='institution_net_buy')
    inst_net_ratio = inst_net / (volume * close) * 100
    inst_net_ratio = inst_net_ratio.reindex(close.index).fillna(0).clip(-100, 100)
    
    conn.close()
    
    # Best Alpha (IC 0.0830)
    alpha = ops.normed_rank(
        ops.add(
            ops.add(
                ops.add(
                    ops.normed_rank(ops.ts_delta_ratio(ops.ts_median(open_price, 130), 25)),
                    ops.normed_rank(ops.ts_corr(gap, inst_net_ratio, 60))
                ),
                ops.ts_regression_residual(intraday_ret, close, 20)
            ),
            ops.ts_corr(close, inst_net_ratio, 60)
        )
    )
    
    # 최신 알파값
    latest = alpha.iloc[-1].dropna().sort_values(ascending=False)
    
    # 제외 종목 필터링
    latest = latest[~latest.index.isin(exclude_tickers)]
    
    ticker_to_name = dict(zip(stocks_df['ticker'], stocks_df['name']))
    ticker_to_price = dict(zip(close.columns, close.iloc[-1]))
    
    top_stocks = []
    for ticker in latest.head(top_n).index:
        top_stocks.append({
            'ticker': ticker,
            'name': ticker_to_name.get(ticker, ticker),
            'alpha': latest[ticker],
            'price': ticker_to_price.get(ticker, 0)
        })
    
    return top_stocks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-n', type=int, default=5)
    parser.add_argument('--amount', type=float, default=None, help='총 투자금액 (미지정시 예수금 사용)')
    parser.add_argument('--exclude', nargs='*', default=['042700', '005690', '058470'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🏆 Alpha-GPT Top 200 Trading (월간 리밸런싱)")
    print("=" * 60)
    
    # 상위 종목 선정
    top_stocks = get_top_stocks(args.top_n, args.exclude)
    
    print(f"\n📊 상위 {args.top_n}개 종목:")
    for i, s in enumerate(top_stocks, 1):
        print(f"  {i}. {s['ticker']} {s['name']:12} | 알파: {s['alpha']:.4f} | 현재가: {s['price']:,.0f}원")
    
    if args.dry_run:
        print("\n[DRY RUN] 실제 매수하지 않음")
        return
    
    # KIS API
    api = KISApi(
        app_key=os.getenv('KIS_APP_KEY'),
        app_secret=os.getenv('KIS_APP_SECRET'),
        account_no=os.getenv('KIS_ACCOUNT_NO'),
        is_real=True
    )
    
    # 예수금 확인
    balance = api.get_balance()
    if args.amount:
        total_amount = args.amount
    else:
        total_amount = int(balance.get('output2', [{}])[0].get('dnca_tot_amt', 0))
    
    per_stock = total_amount / len(top_stocks)
    print(f"\n💰 총 투자금액: {total_amount:,.0f}원 (종목당 {per_stock:,.0f}원)")
    
    # 매수 실행
    print("\n🛒 매수 실행:")
    for s in top_stocks:
        qty = int(per_stock / s['price'])
        if qty > 0:
            result = api.buy_market_order(s['ticker'], qty)
            status = "✅" if result.get('rt_cd') == '0' else "❌"
            print(f"  {status} {s['ticker']} {s['name']} | {qty}주 × {s['price']:,.0f}원")
        else:
            print(f"  ⚠️ {s['ticker']} 가격({s['price']:,.0f}원)이 예산 초과")
    
    print("\n🎉 완료!")

if __name__ == '__main__':
    main()
