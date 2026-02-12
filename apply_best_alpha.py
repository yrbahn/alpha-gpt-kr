#!/usr/bin/env python3
"""
Alpha-GPT가 생성한 최상위 알파를 500개 종목에 적용
"""

import sys
import os
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.mining.operators import AlphaOperators

# 환경 변수 로드
load_dotenv()

# 최상위 알파 (Alpha-GPT가 생성)
BEST_ALPHA = "AlphaOperators.ts_rank(AlphaOperators.ts_std(returns, 10) / AlphaOperators.ts_std(returns, 20), 10)"
ALPHA_IC = 0.0467

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )

def load_top500_data():
    """시가총액 상위 500개 종목 데이터"""
    print("📊 상위 500개 종목 데이터 로드 중...")
    
    conn = get_db_connection()
    
    # 상위 500개 종목
    query_stocks = """
        SELECT DISTINCT ON (s.ticker)
            s.id, s.ticker, s.name
        FROM stocks s
        JOIN price_data p ON s.id = p.stock_id
        WHERE s.is_active = true
        AND p.date = (SELECT MAX(date) FROM price_data)
        AND p.close IS NOT NULL AND p.volume IS NOT NULL
        ORDER BY s.ticker, (p.close * p.volume) DESC
        LIMIT 500
    """
    
    stocks_df = pd.read_sql(query_stocks, conn)
    stock_ids = stocks_df['id'].tolist()
    
    # 가격 데이터 (최근 60일 - 알파 계산용)
    stock_id_list = ', '.join(map(str, stock_ids))
    query_prices = f"""
        SELECT 
            s.ticker,
            s.name,
            p.date,
            p.close,
            p.volume
        FROM price_data p
        JOIN stocks s ON p.stock_id = s.id
        WHERE p.stock_id IN ({stock_id_list})
        AND p.date >= CURRENT_DATE - INTERVAL '60 days'
        ORDER BY s.ticker, p.date
    """
    
    price_df = pd.read_sql(query_prices, conn)
    conn.close()
    
    # 피벗 테이블
    close_pivot = price_df.pivot(index='date', columns='ticker', values='close')
    volume_pivot = price_df.pivot(index='date', columns='ticker', values='volume')
    
    print(f"✅ {len(close_pivot.columns)}개 종목, {len(close_pivot)}일 데이터")
    
    # 종목명 매핑
    name_map = price_df.groupby('ticker')['name'].first().to_dict()
    
    return {
        'close': close_pivot,
        'volume': volume_pivot,
        'returns': close_pivot.pct_change(),
        'names': name_map
    }

def calculate_alpha(data):
    """알파 계산"""
    print(f"\n📈 알파 계산 중...")
    print(f"   공식: {BEST_ALPHA}")
    
    close = data['close']
    volume = data['volume']
    returns = data['returns']
    
    # 알파 계산
    alpha_values = eval(BEST_ALPHA)
    
    # 최신 날짜의 알파 스코어
    latest_date = alpha_values.index[-1]
    latest_alpha = alpha_values.loc[latest_date]
    
    # 유효한 값만
    valid_alpha = latest_alpha.dropna()
    
    # 정렬
    sorted_alpha = valid_alpha.sort_values(ascending=False)
    
    print(f"✅ {len(sorted_alpha)}개 종목 알파 계산 완료")
    
    return sorted_alpha, data

def save_to_db(alpha_scores, data):
    """알파 스코어를 DB에 저장"""
    calc_date = date.today()
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # 기존 데이터 삭제
        cur.execute("""
            DELETE FROM alpha_scores 
            WHERE calculation_date = %s AND alpha_formula = %s
        """, (calc_date, BEST_ALPHA))
        
        print(f"\n💾 DB 저장 중...")
        
        # 데이터 준비
        values = []
        for rank, (ticker, alpha_score) in enumerate(alpha_scores.items(), 1):
            stock_name = data['names'].get(ticker, ticker)
            latest_close = data['close'][ticker].iloc[-1]
            latest_volume = data['volume'][ticker].iloc[-1]
            market_cap = int(latest_close * latest_volume)
            
            values.append((
                calc_date,
                ticker,
                stock_name,
                BEST_ALPHA,
                float(alpha_score),
                rank,
                market_cap,
                float(latest_close),
                int(latest_volume)
            ))
        
        # 삽입
        execute_values(cur, """
            INSERT INTO alpha_scores 
            (calculation_date, stock_code, stock_name, alpha_formula, alpha_score, 
             rank, market_cap, close_price, volume)
            VALUES %s
        """, values)
        
        conn.commit()
        print(f"✅ {len(values)}개 알파 스코어 DB 저장 완료")
        
        # 상위 15개 출력
        print(f"\n🏆 상위 15개 종목 (Alpha-GPT 생성 알파):")
        for rank, (ticker, alpha_score) in enumerate(list(alpha_scores.items())[:15], 1):
            stock_name = data['names'].get(ticker, ticker)
            latest_close = data['close'][ticker].iloc[-1]
            print(f"   {rank:2d}. {ticker} {stock_name:15s} α={alpha_score:.4f} 가격={latest_close:,.0f}원")
        
    finally:
        cur.close()
        conn.close()

def save_trading_signals(alpha_scores, data, top_n=15):
    """매매 신호 저장"""
    signal_date = date.today()
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # 기존 신호 삭제
        cur.execute("DELETE FROM trading_signals WHERE signal_date = %s", (signal_date,))
        
        # 상위 N개 선택
        top_stocks = list(alpha_scores.items())[:top_n]
        
        values = []
        for rank, (ticker, alpha_score) in enumerate(top_stocks, 1):
            stock_name = data['names'].get(ticker, ticker)
            
            values.append((
                signal_date,
                ticker,
                stock_name,
                'BUY',
                float(alpha_score),
                rank,
                1.0 / top_n,
                f"Alpha-GPT rank #{rank}, IC={ALPHA_IC:.4f}, score={alpha_score:.4f}"
            ))
        
        execute_values(cur, """
            INSERT INTO trading_signals
            (signal_date, stock_code, stock_name, signal_type, alpha_score, rank, target_weight, reason)
            VALUES %s
        """, values)
        
        conn.commit()
        print(f"\n✅ {len(values)}개 매매 신호 저장 완료")
        
    finally:
        cur.close()
        conn.close()

def main():
    print("=" * 70)
    print("Alpha-GPT 생성 알파 적용 (상위 500개 종목)")
    print("=" * 70)
    print(f"알파: {BEST_ALPHA}")
    print(f"IC: {ALPHA_IC:.4f}")
    print("=" * 70)
    print()
    
    try:
        # 1. 데이터 로드
        data = load_top500_data()
        
        # 2. 알파 계산
        alpha_scores, data = calculate_alpha(data)
        
        # 3. DB 저장
        save_to_db(alpha_scores, data)
        
        # 4. 매매 신호 생성
        save_trading_signals(alpha_scores, data, top_n=15)
        
        print("\n" + "=" * 70)
        print("✅ Alpha-GPT 알파 적용 완료!")
        print("=" * 70)
        print(f"\n📊 대시보드: http://localhost:9999/dashboard.html")
        print(f"🔄 대시보드 업데이트:")
        print(f"   cd /Users/yrbahn/.openclaw/workspace/alpha-gpt-kr")
        print(f"   python3 generate_dashboard.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
