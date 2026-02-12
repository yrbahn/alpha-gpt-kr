#!/usr/bin/env python3
"""
시가총액 상위 500개 종목 알파 계산 (최적화 버전)
"""
import os
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 알파 계산 함수
def ts_delta(x, period):
    """현재값 - N일 전 값 (간단한 모멘텀)"""
    if len(x) < period:
        return np.full(len(x), np.nan)
    result = np.full(len(x), np.nan)
    result[period:] = x[period:] - x[:-period]
    return result

# 환경 변수 로드
load_dotenv()

# 현재 최적 알파
CURRENT_ALPHA = "ops.ts_delta(close, 26)"
ALPHA_DESCRIPTION = "26-day momentum (GP evolved, IC: 0.0045, Sharpe: 0.57)"

def get_db_connection():
    """PostgreSQL 연결"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )

def get_top_500_stocks():
    """시가총액 상위 500개 종목 가져오기"""
    conn = get_db_connection()
    
    # 최신 날짜의 시가총액 기준 정렬
    query = """
        SELECT DISTINCT ON (s.ticker)
            s.ticker,
            s.name,
            p.close * p.volume as market_cap_proxy
        FROM stocks s
        JOIN price_data p ON s.id = p.stock_id
        WHERE s.is_active = true
        AND p.date = (SELECT MAX(date) FROM price_data)
        AND p.close IS NOT NULL
        AND p.volume IS NOT NULL
        ORDER BY s.ticker, market_cap_proxy DESC
        LIMIT 500
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"✅ 시가총액 상위 {len(df)}개 종목 선택")
    return df

def load_price_data_for_stocks(tickers, days=90):
    """특정 종목들의 가격 데이터만 로드"""
    conn = get_db_connection()
    
    # 날짜 범위
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    print(f"📊 데이터 로드: {start_date} ~ {end_date} ({days}일)")
    
    # 종목 ID 가져오기
    ticker_list = "', '".join([f"{t:0>6}" if len(str(t)) < 6 else str(t) for t in tickers])
    
    query_stocks = f"""
        SELECT id, ticker, name
        FROM stocks
        WHERE ticker IN ('{ticker_list}')
    """
    stocks_df = pd.read_sql(query_stocks, conn)
    stock_ids = stocks_df['id'].tolist()
    
    print(f"✅ {len(stocks_df)}개 종목 매칭")
    
    # 가격 데이터 로드
    stock_id_list = ', '.join(map(str, stock_ids))
    query_prices = f"""
        SELECT 
            s.ticker as stock_code,
            s.name as stock_name,
            p.date,
            p.close,
            p.volume
        FROM price_data p
        JOIN stocks s ON p.stock_id = s.id
        WHERE p.stock_id IN ({stock_id_list})
        AND p.date >= '{start_date}'
        AND p.date <= '{end_date}'
        ORDER BY s.ticker, p.date
    """
    
    price_df = pd.read_sql(query_prices, conn)
    conn.close()
    
    print(f"✅ 가격 데이터 로드: {len(price_df):,} 행")
    
    return price_df

def calculate_alpha_scores(price_df):
    """알파 계산"""
    print(f"📈 알파 계산 중: {CURRENT_ALPHA}")
    
    results = []
    
    # 종목별로 그룹핑
    for stock_code, group in price_df.groupby('stock_code'):
        try:
            # 날짜 순 정렬
            group = group.sort_values('date')
            
            if len(group) < 30:  # 최소 30일 필요
                continue
            
            close = group['close'].values
            
            # 알파 계산: ts_delta(close, 26)
            alpha = ts_delta(close, 26)
            
            if len(alpha) > 0 and not np.isnan(alpha[-1]):
                latest = group.iloc[-1]
                
                results.append({
                    'stock_code': stock_code,
                    'stock_name': latest['stock_name'],
                    'alpha_score': float(alpha[-1]),
                    'market_cap': int(latest['close'] * latest['volume']),
                    'close_price': float(latest['close']),
                    'volume': int(latest['volume'])
                })
                
                if len(results) % 50 == 0:
                    print(f"  ... {len(results)}개 종목 완료")
                    
        except Exception as e:
            print(f"⚠️  {stock_code} 오류: {e}")
            continue
    
    # DataFrame 생성 및 정렬
    result_df = pd.DataFrame(results)
    
    if result_df.empty:
        raise ValueError("알파 스코어 계산 실패")
    
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
        
        print(f"📝 DB 저장 중...")
        
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
        
        # 상위 15개 출력
        print("\n📊 Top 15 Alpha Scores:")
        print(df_scores[['rank', 'stock_code', 'stock_name', 'alpha_score', 'close_price']].head(15).to_string(index=False))
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def save_trading_signals(df_scores, top_n=15):
    """매매 신호 저장"""
    signal_date = date.today()
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # 기존 신호 삭제
        cur.execute("DELETE FROM trading_signals WHERE signal_date = %s", (signal_date,))
        
        # 상위 N개 종목 선택
        top_stocks = df_scores.head(top_n)
        
        values = [
            (
                signal_date,
                row['stock_code'],
                row['stock_name'],
                'BUY',
                row['alpha_score'],
                row['rank'],
                1.0 / top_n,
                f"Alpha rank #{row['rank']}, score={row['alpha_score']:.6f}"
            )
            for _, row in top_stocks.iterrows()
        ]
        
        execute_values(cur, """
            INSERT INTO trading_signals
            (signal_date, stock_code, stock_name, signal_type, alpha_score, rank, target_weight, reason)
            VALUES %s
        """, values)
        
        conn.commit()
        print(f"\n✅ {len(values)}개 매매 신호 저장 완료")
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def main():
    print("=" * 70)
    print("Alpha-GPT-KR: 시가총액 상위 500개 종목 알파 계산")
    print("=" * 70)
    print(f"알파 공식: {CURRENT_ALPHA}")
    print(f"설명: {ALPHA_DESCRIPTION}")
    print(f"계산 날짜: {date.today()}")
    print("=" * 70)
    print()
    
    try:
        # 1. 시가총액 상위 500개 종목 선택
        print("📊 1단계: 시가총액 상위 500개 종목 선택")
        top_stocks = get_top_500_stocks()
        
        # 2. 가격 데이터 로드 (최근 90일)
        print("\n📊 2단계: 가격 데이터 로드")
        price_df = load_price_data_for_stocks(top_stocks['ticker'].tolist(), days=90)
        
        # 3. 알파 계산
        print("\n📊 3단계: 알파 계산")
        df_scores = calculate_alpha_scores(price_df)
        
        # 4. DB 저장
        print("\n📊 4단계: DB 저장")
        save_to_db(df_scores)
        
        # 5. 매매 신호 생성
        print("\n📊 5단계: 매매 신호 생성")
        save_trading_signals(df_scores, top_n=15)
        
        print("\n" + "=" * 70)
        print("✅ 알파 계산 및 저장 완료!")
        print("=" * 70)
        print(f"\n📊 대시보드: http://localhost:9999/dashboard.html")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
