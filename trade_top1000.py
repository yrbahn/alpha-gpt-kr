#!/usr/bin/env python3
"""
시총 상위 1000개 종목에서 알파 기반 매매
"""
import os
import sys
from pathlib import Path
from datetime import datetime, date
import pandas as pd
from dotenv import load_dotenv
import psycopg2
import argparse

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.trading.kis_api import KISApi
import time

# 환경 변수 로드
load_dotenv()

def get_db_connection():
    """PostgreSQL 연결"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )

def filter_tradable_stocks(df: pd.DataFrame, api: KISApi, top_n: int) -> pd.DataFrame:
    """거래정지 종목 제외 필터링"""
    tradable_stocks = []
    excluded_stocks = []
    
    print("\n🔍 거래가능 종목 확인 중...")
    
    for idx, row in df.iterrows():
        ticker = row['stock_code']
        name = row['stock_name']
        
        try:
            # 현재가 조회 (거래정지 확인)
            price_info = api.get_current_price(ticker)
            
            # 거래정지 여부 확인
            status_code = price_info.get('iscd_stat_cls_code', '00')
            
            if status_code == '58':
                excluded_stocks.append(f"{ticker} ({name})")
                print(f"   ⚠️  {ticker} ({name}): 거래정지")
            else:
                tradable_stocks.append(row)
                print(f"   ✅ {ticker} ({name}): 거래가능")
                
            # API 호출 제한 (초당 20건)
            time.sleep(0.06)
            
            # 충분한 종목 확보 시 중단
            if len(tradable_stocks) >= top_n:
                break
                
        except Exception as e:
            print(f"   ❌ {ticker} ({name}): 조회 실패 ({e})")
            excluded_stocks.append(f"{ticker} ({name})")
    
    df_tradable = pd.DataFrame(tradable_stocks)
    
    print(f"\n📋 필터링 결과:")
    print(f"   거래가능: {len(tradable_stocks)}개")
    if excluded_stocks:
        print(f"   제외된 종목: {', '.join(excluded_stocks)}")
    
    return df_tradable

def load_alpha_scores_top1000(top_n=15, exclude_tickers=None):
    """시총 상위 1000개 종목에서 알파 스코어 로드"""
    conn = get_db_connection()
    
    # 최신 알파 공식 (재무 알파 - ROA)
    alpha_formula = "AlphaOperators.normed_rank((net_income / total_assets) + (operating_income / total_assets))"
    
    # 제외할 종목 조건 추가
    exclude_condition = ""
    if exclude_tickers:
        exclude_list = "', '".join(exclude_tickers)
        exclude_condition = f"AND a.stock_code NOT IN ('{exclude_list}')"
    
    # 시총 상위 1000개에서 알파 스코어 조회
    query = f"""
        WITH top1000 AS (
            SELECT ticker
            FROM stocks
            WHERE market_cap IS NOT NULL
            ORDER BY market_cap DESC
            LIMIT 1000
        )
        SELECT 
            a.stock_code,
            COALESCE(s.name, a.stock_code) as stock_name,
            s.market_cap,
            a.alpha_score,
            a.close_price,
            a.volume
        FROM alpha_scores a
        LEFT JOIN stocks s ON a.stock_code = s.ticker
        INNER JOIN top1000 t ON a.stock_code = t.ticker
        WHERE a.calculation_date = (SELECT MAX(calculation_date) FROM alpha_scores)
        AND a.alpha_formula = '{alpha_formula}'
        AND a.alpha_score IS NOT NULL
        AND NOT (a.alpha_score = 'NaN'::float)
        AND NOT (a.alpha_score = 'Infinity'::float)
        AND NOT (a.alpha_score = '-Infinity'::float)
        {exclude_condition}
        ORDER BY a.alpha_score DESC
        LIMIT {top_n * 2}
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"\n📊 시총 상위 1000개 중 Top {len(df)} stocks (before filtering):")
    if exclude_tickers:
        print(f"   (제외: {', '.join(exclude_tickers)})")
    print(df[['stock_code', 'stock_name', 'market_cap', 'alpha_score', 'close_price']].to_string(index=False))
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top-n', type=int, default=15, help='상위 N개 종목')
    parser.add_argument('--amount', type=int, default=5000000, help='총 투자 금액')
    parser.add_argument('--dry-run', action='store_true', help='시뮬레이션만 (실제 주문 X)')
    parser.add_argument('--exclude', type=str, nargs='+', help='제외할 종목 코드 (예: 042700 000660)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Alpha-GPT-KR: Trade from Top 1000 Market Cap")
    print("=" * 60)
    print(f"Date: {date.today()}")
    print(f"Universe: 시총 상위 1000개")
    print(f"Top N stocks: {args.top_n}")
    print(f"Target amount: {args.amount:,}원")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE TRADING'}")
    print("=" * 60)
    
    # KIS API 초기화 (거래정지 확인용)
    api = KISApi(
        app_key=os.getenv('KIS_APP_KEY'),
        app_secret=os.getenv('KIS_APP_SECRET'),
        account_no=os.getenv('KIS_ACCOUNT_NO'),
        is_real=True
    )
    
    # 알파 스코어 로드
    df_scores = load_alpha_scores_top1000(args.top_n, exclude_tickers=args.exclude)
    
    if df_scores.empty:
        print("\n❌ No alpha scores found!")
        return
    
    # 거래정지 종목 필터링
    df_scores = filter_tradable_stocks(df_scores, api, args.top_n)
    
    if df_scores.empty:
        print("\n❌ No tradable stocks found!")
        return
    
    # 종목당 투자 금액
    amount_per_stock = args.amount / len(df_scores)
    
    print(f"\n💰 Investment plan:")
    print(f"   Total: {args.amount:,}원")
    print(f"   Per stock: {int(amount_per_stock):,}원")
    print(f"   Stocks: {len(df_scores)}개")
    
    if args.dry_run:
        print("\n✅ DRY RUN mode - no actual orders")
        print("\n📋 Would buy:")
        for idx, row in df_scores.iterrows():
            ticker = row['stock_code']
            name = row['stock_name']
            price = row['close_price']
            qty = int(amount_per_stock / price)
            total = qty * price
            
            print(f"   {ticker} ({name}): {qty}주 × {int(price):,}원 = {int(total):,}원")
        
        return
    
    # 실전 매매
    print("\n🚀 Starting live trading...")
    
    # 잔고 조회
    balance = api.get_balance()
    available_cash = int(balance['dnca_tot_amt'])  # 예수금
    
    print(f"\n💵 Available cash: {available_cash:,}원")
    
    if available_cash < args.amount:
        print(f"⚠️  Warning: Not enough cash (need {args.amount:,}원)")
        return
    
    # 주문 실행
    results = []
    
    for idx, row in df_scores.iterrows():
        ticker = row['stock_code']
        name = row['stock_name']
        price = row['close_price']
        qty = int(amount_per_stock / price)
        
        if qty == 0:
            print(f"⏭️  Skip {ticker} ({name}): 가격 너무 높음 ({int(price):,}원)")
            continue
        
        print(f"\n📈 Buying {ticker} ({name}): {qty}주 × {int(price):,}원")
        
        try:
            order = api.buy_stock(ticker, qty, order_type="01")  # 01 = 시장가
            results.append({
                'ticker': ticker,
                'name': name,
                'qty': qty,
                'price': price,
                'status': 'success',
                'order_no': order.get('output', {}).get('ODNO', '')
            })
            print(f"   ✅ Order placed: {order.get('output', {}).get('ODNO', 'N/A')}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append({
                'ticker': ticker,
                'name': name,
                'qty': qty,
                'price': price,
                'status': 'failed',
                'error': str(e)
            })
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("📊 Trading Results")
    print("=" * 60)
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    success_count = len([r for r in results if r['status'] == 'success'])
    print(f"\n✅ Success: {success_count}/{len(results)}")
    
    print("\n🎉 Trading completed!")

if __name__ == "__main__":
    main()
