#!/usr/bin/env python3
"""
최종 확정 알파로 종목 선정
IC: 0.0745
Alpha: AlphaOperators.ts_std(returns, 5) / AlphaOperators.ts_mean(close, 91)
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2
from datetime import date

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.mining.operators import AlphaOperators

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )

def load_latest_data():
    """최신 데이터 로드 (전체 시장, 유동성 필터링)"""
    print("📊 최신 데이터 로드 중... (전체 시장, 유동성 필터링)")
    
    conn = get_db_connection()
    
    # 전체 활성 종목 (유동성 필터: 최근 30일 평균 거래대금 1억원 이상)
    query_stocks = """
        WITH recent_trading AS (
            SELECT 
                stock_id,
                AVG(close * volume) as avg_trading_value
            FROM price_data
            WHERE date >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY stock_id
            HAVING AVG(close * volume) >= 100000000
        )
        SELECT DISTINCT s.id, s.ticker, s.name
        FROM stocks s
        JOIN recent_trading rt ON s.id = rt.stock_id
        WHERE s.is_active = true
        ORDER BY s.ticker
    """
    
    stocks_df = pd.read_sql(query_stocks, conn)
    stock_ids = stocks_df['id'].tolist()
    stock_id_list = ', '.join(map(str, stock_ids))
    
    query_prices = f"""
        SELECT s.ticker, s.name, p.date, p.close
        FROM price_data p
        JOIN stocks s ON p.stock_id = s.id
        WHERE p.stock_id IN ({stock_id_list})
        AND p.date >= CURRENT_DATE - INTERVAL '120 days'
        ORDER BY s.ticker, p.date
    """
    
    price_df = pd.read_sql(query_prices, conn)
    conn.close()
    
    # Pivot
    close = price_df.pivot(index='date', columns='ticker', values='close')
    returns = close.pct_change()
    
    # 종목명 매핑
    name_map = price_df.groupby('ticker')['name'].first().to_dict()
    
    print(f"✅ {len(close.columns)}개 종목, {len(close)}일 데이터")
    
    return close, returns, name_map

def calculate_alpha(close, returns):
    """확정 알파 계산"""
    print("\n🧮 알파 계산 중...")
    print("   Formula: AlphaOperators.ts_std(returns, 5) / AlphaOperators.ts_mean(close, 91)")
    
    alpha = AlphaOperators.ts_std(returns, 5) / AlphaOperators.ts_mean(close, 91)
    
    # 최신 날짜의 알파 값
    latest_alpha = alpha.iloc[-1]
    
    print(f"✅ 알파 계산 완료 (최신: {alpha.index[-1]})")
    
    return latest_alpha

def main():
    print("=" * 80)
    print("Alpha-GPT: 종목 선정 (15-day Forward)")
    print("=" * 80)
    print(f"실행 시간: {date.today()} {pd.Timestamp.now().strftime('%H:%M:%S')}")
    print()
    
    # 데이터 로드
    close, returns, name_map = load_latest_data()
    
    # 알파 계산
    latest_alpha = calculate_alpha(close, returns)
    
    # 상위 종목 선정
    print("\n" + "=" * 80)
    print("📈 알파 상위 종목 (Top 10)")
    print("=" * 80)
    
    top_stocks = latest_alpha.sort_values(ascending=False).head(10)
    
    results = []
    for i, (ticker, alpha_score) in enumerate(top_stocks.items(), 1):
        name = name_map.get(ticker, ticker)
        latest_price = close[ticker].iloc[-1]
        
        results.append({
            '순위': i,
            '종목코드': ticker,
            '종목명': name,
            '알파점수': f"{alpha_score:.6f}",
            '현재가': f"{int(latest_price):,}원"
        })
        
        print(f"{i:2d}. {ticker} ({name:15s}) | 알파: {alpha_score:.6f} | 현재가: {int(latest_price):,}원")
    
    # CSV로 저장
    df_results = pd.DataFrame(results)
    df_results.to_csv('selected_stocks_tomorrow.csv', index=False, encoding='utf-8-sig')
    
    print("\n✅ 결과를 selected_stocks_tomorrow.csv에 저장했습니다!")
    
    # 매수 금액 계산 (8개 종목)
    print("\n" + "=" * 80)
    print("💰 매수 계획 (8개 종목, 총 500만원)")
    print("=" * 80)
    
    top_8 = top_stocks.head(8)
    amount_per_stock = 5_000_000 / 8
    
    print(f"종목당 투자 금액: {int(amount_per_stock):,}원\n")
    
    for i, (ticker, alpha_score) in enumerate(top_8.items(), 1):
        name = name_map.get(ticker, ticker)
        price = close[ticker].iloc[-1]
        qty = int(amount_per_stock / price)
        total = qty * price
        
        print(f"{i}. {ticker} ({name:15s}) | {qty:3d}주 × {int(price):,}원 = {int(total):,}원")
    
    print("\n" + "=" * 80)
    print("⏰ 내일 아침 09:00 실행 명령어")
    print("=" * 80)
    print()
    print("# 1. 기존 종목 전량 매도")
    print("python3 sell_all_holdings.py")
    print()
    print("# 2. 신규 8개 종목 매수")
    print("python3 simple_trade_from_db.py --top-n 8 --amount 5000000")
    print()
    print("=" * 80)
    
    print("\n🎉 준비 완료! 내일 아침 09:00에 실행하세요!")

if __name__ == "__main__":
    main()
