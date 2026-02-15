#!/usr/bin/env python3
"""
재무 알파 계산 후 DB 저장
"""
import os
import sys
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.mining.operators import AlphaOperators

load_dotenv()

# 재무 알파
CURRENT_ALPHA = "AlphaOperators.normed_rank((net_income / total_assets) + (operating_income / total_assets))"
ALPHA_DESCRIPTION = "Fundamental Alpha: ROA + Operating ROA (IC: 0.0751, IR: 0.92)"

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )

def calculate_alpha_scores(top_n=500):
    """재무 알파 계산"""
    print(f"📊 재무 데이터 로드 중... (상위 {top_n}개 종목)")
    
    conn = get_db_connection()
    
    # 재무 데이터 있는 종목 중 시총 상위
    query_stocks = """
        SELECT DISTINCT ON (s.ticker)
            s.id, s.ticker, s.name
        FROM stocks s
        JOIN price_data p ON s.id = p.stock_id
        JOIN financial_statements f ON s.id = f.stock_id
        WHERE s.is_active = true
        AND p.date = (SELECT MAX(date) FROM price_data)
        AND f.revenue IS NOT NULL
        AND f.period_end >= CURRENT_DATE - INTERVAL '365 days'
        ORDER BY s.ticker, (p.close * p.volume) DESC
        LIMIT %s
    """
    
    stocks_df = pd.read_sql(query_stocks, conn, params=(top_n,))
    stock_ids = stocks_df['id'].tolist()
    stock_id_list = ', '.join(map(str, stock_ids))
    
    # 최신 가격
    query_price = f"""
        SELECT s.ticker, p.close, p.volume
        FROM price_data p
        JOIN stocks s ON p.stock_id = s.id
        WHERE p.stock_id IN ({stock_id_list})
        AND p.date = (SELECT MAX(date) FROM price_data)
    """
    price_df = pd.read_sql(query_price, conn)
    
    # 최신 재무 데이터
    query_fin = f"""
        SELECT DISTINCT ON (s.ticker)
            s.ticker,
            f.net_income,
            f.operating_income,
            f.total_assets
        FROM financial_statements f
        JOIN stocks s ON f.stock_id = s.id
        WHERE f.stock_id IN ({stock_id_list})
        AND f.revenue IS NOT NULL
        ORDER BY s.ticker, f.period_end DESC
    """
    fin_df = pd.read_sql(query_fin, conn)
    
    conn.close()
    
    # 데이터 병합
    df = stocks_df[['ticker', 'name']].merge(price_df, on='ticker').merge(fin_df, on='ticker')
    
    print(f"✅ {len(df)}개 종목 데이터 로드 완료")
    print(f"📈 알파 계산: {CURRENT_ALPHA}")
    
    # 알파 계산
    net_income = df['net_income'].values
    operating_income = df['operating_income'].values
    total_assets = df['total_assets'].values
    
    # ROA + Operating ROA
    roa = net_income / total_assets
    operating_roa = operating_income / total_assets
    combined_roa = roa + operating_roa
    
    # Normed rank (0~1)
    alpha_scores = pd.Series(combined_roa).rank(pct=True).values
    
    # 결과 데이터프레임
    results = []
    for i, row in df.iterrows():
        results.append({
            'stock_code': row['ticker'],
            'stock_name': row['name'],
            'alpha_score': float(alpha_scores[i]),
            'market_cap': int(row['close'] * row['volume']) if row['volume'] else 0,
            'close_price': float(row['close']),
            'volume': int(row['volume']) if row['volume'] else 0
        })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('alpha_score', ascending=False)
    result_df['rank'] = range(1, len(result_df) + 1)
    
    print(f"✅ {len(result_df)}개 종목 알파 계산 완료")
    
    return result_df

def save_to_db(df_scores, calculation_date=None):
    """알파 스코어를 DB에 저장"""
    if calculation_date is None:
        calculation_date = date.today()
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # 같은 날짜/알파 조합 삭제
        cur.execute("""
            DELETE FROM alpha_scores 
            WHERE calculation_date = %s AND alpha_formula = %s
        """, (calculation_date, CURRENT_ALPHA))
        
        # 데이터 삽입
        values = [
            (
                calculation_date,
                row['stock_code'],
                row['stock_name'],
                CURRENT_ALPHA,
                row['alpha_score'],
                row['rank'],
                row['market_cap'],
                row['close_price'],
                row['volume']
            )
            for _, row in df_scores.iterrows()
        ]
        
        execute_values(cur, """
            INSERT INTO alpha_scores 
            (calculation_date, stock_code, stock_name, alpha_formula, alpha_score, 
             rank, market_cap, close_price, volume)
            VALUES %s
        """, values)
        
        conn.commit()
        print(f"✅ {len(df_scores)}개 알파 스코어 DB 저장 완료")
        
        # 상위 10개 출력
        print("\n📊 Top 10 종목:")
        print(df_scores[['rank', 'stock_code', 'stock_name', 'alpha_score', 'close_price']].head(10).to_string(index=False))
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def main():
    print("=" * 60)
    print("재무 알파 계산 및 저장")
    print("=" * 60)
    print(f"알파: {CURRENT_ALPHA}")
    print(f"설명: {ALPHA_DESCRIPTION}")
    print(f"계산 날짜: {date.today()}")
    print("=" * 60)
    
    try:
        # 알파 계산
        df_scores = calculate_alpha_scores(top_n=500)
        
        # DB 저장
        save_to_db(df_scores)
        
        print("\n✅ 재무 알파 계산 및 저장 완료!")
        print("\n다음 단계:")
        print("  python3 simple_trade_from_db.py --top-n 15 --amount 5000000 --dry-run")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
