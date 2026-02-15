#!/usr/bin/env python3
"""
알파 계산 후 DB 저장 스크립트
매일 오후 실행하여 다음 날 매수에 사용할 알파 스코어 계산
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

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.data.postgres_loader import PostgresDataLoader
from alpha_gpt_kr.mining.operators import AlphaOperators

# 환경 변수 로드
load_dotenv()

# 현재 최적 알파 (재무 알파 - 500종목 검증 완료, IC 0.0751)
CURRENT_ALPHA = "AlphaOperators.normed_rank((net_income / total_assets) + (operating_income / total_assets))"
ALPHA_DESCRIPTION = "Fundamental Alpha: ROA + Operating ROA (IC: 0.0751, IR: 0.92, 500-stock verified)"

def get_db_connection():
    """PostgreSQL 연결"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )

def calculate_alpha_scores(top_n=500):
    """
    알파 계산
    
    Args:
        top_n: 시가총액 상위 N개 종목
    
    Returns:
        DataFrame with columns: stock_code, stock_name, alpha_score, market_cap, close_price, volume
    """
    print(f"📊 Loading data for top {top_n} stocks...")
    
    # PostgreSQL 데이터 로더 초기화
    loader = PostgresDataLoader(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )
    
    # 데이터 로드 (최근 3개월만) - 속도 개선
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    panel_data = loader.load_data(start_date=start_date, end_date=end_date)
    
    if not panel_data or 'close' not in panel_data:
        raise ValueError("No data loaded")
    
    close_df = panel_data['close']  # index=date, columns=ticker
    volume_df = panel_data.get('volume', None)
    
    print(f"📈 Loaded data: {close_df.shape[0]} days, {close_df.shape[1]} stocks")
    
    # 시가총액 상위 top_n 종목 선택
    # 최신 날짜의 가격 * 거래량으로 근사
    latest_date = close_df.index[-1]
    latest_close = close_df.loc[latest_date]
    
    if volume_df is not None:
        latest_volume = volume_df.loc[latest_date]
        market_caps = latest_close * latest_volume
    else:
        # 거래량 없으면 가격으로만
        market_caps = latest_close
    
    top_tickers = market_caps.nlargest(top_n).index.tolist()
    print(f"📊 Top {len(top_tickers)} stocks selected")
    
    print(f"📈 Calculating alpha: {CURRENT_ALPHA}")
    
    # returns 계산 (전체 DataFrame에서)
    returns = close_df.pct_change()
    
    # Cross-sectional 알파 계산 (최신 날짜 기준)
    try:
        # 알파 계산: AlphaOperators.ts_rank(AlphaOperators.ts_mean(returns, 1), 26)
        close = close_df[top_tickers]
        volume = volume_df[top_tickers] if volume_df is not None else None
        returns = close.pct_change()
        
        # 알파 계산
        alpha_values = eval(CURRENT_ALPHA)
        
        # 최신 날짜의 알파값
        latest_alpha = alpha_values.iloc[-1]
        
        # 종목별 결과 생성
        results = []
        for ticker in top_tickers:
            if ticker not in latest_alpha.index or pd.isna(latest_alpha[ticker]):
                continue
            
            latest_close_price = close[ticker].iloc[-1]
            latest_volume = volume[ticker].iloc[-1] if volume is not None else 0
            
            results.append({
                'stock_code': ticker,
                'stock_name': ticker,
                'alpha_score': float(latest_alpha[ticker]),
                'market_cap': int(latest_close_price * latest_volume) if volume is not None else 0,
                'close_price': float(latest_close_price),
                'volume': int(latest_volume) if volume is not None else 0
            })
            
    except Exception as e:
        print(f"❌ Error calculating alpha: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # DataFrame 생성 및 정렬
    result_df = pd.DataFrame(results)
    if result_df.empty:
        raise ValueError("No alpha scores calculated")
    
    result_df = result_df.sort_values('alpha_score', ascending=False)
    result_df['rank'] = range(1, len(result_df) + 1)
    
    print(f"✅ Calculated alpha for {len(result_df)} stocks")
    
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
        print(f"✅ Saved {len(df_scores)} alpha scores to database")
        
        # 상위 10개 출력
        print("\n📊 Top 10 Alpha Scores:")
        print(df_scores[['rank', 'stock_code', 'stock_name', 'alpha_score', 'close_price']].head(10).to_string(index=False))
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def main():
    print("=" * 60)
    print("Alpha-GPT-KR: Calculate and Save Alpha Scores")
    print("=" * 60)
    print(f"Alpha Formula: {CURRENT_ALPHA}")
    print(f"Description: {ALPHA_DESCRIPTION}")
    print(f"Calculation Date: {date.today()}")
    print("=" * 60)
    
    try:
        # 알파 계산
        df_scores = calculate_alpha_scores(top_n=500)
        
        # DB 저장
        save_to_db(df_scores)
        
        print("\n✅ Alpha calculation and save completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
