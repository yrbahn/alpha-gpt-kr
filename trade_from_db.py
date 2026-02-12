#!/usr/bin/env python3
"""
DB에 저장된 알파 스코어를 읽어서 매수 실행
매일 아침 장 시작 전 실행
"""
import os
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

# 프로젝트 루트 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.trading.kis_api import KISAPI
from alpha_gpt_kr.trading.trader import AlphaTrader

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

def load_latest_alpha_scores(top_n=15):
    """
    가장 최근 알파 스코어 로드
    
    Args:
        top_n: 상위 N개 종목 선택
    
    Returns:
        DataFrame with top N stocks
    """
    conn = get_db_connection()
    
    # 가장 최근 계산 날짜 찾기
    query_date = """
        SELECT MAX(calculation_date) as latest_date
        FROM alpha_scores
    """
    df_date = pd.read_sql(query_date, conn)
    latest_date = df_date['latest_date'].iloc[0]
    
    if latest_date is None:
        conn.close()
        raise ValueError("No alpha scores found in database")
    
    print(f"📅 Latest alpha calculation date: {latest_date}")
    
    # 상위 N개 종목 로드
    query_scores = """
        SELECT 
            stock_code,
            stock_name,
            alpha_score,
            rank,
            close_price,
            market_cap,
            volume,
            alpha_formula
        FROM alpha_scores
        WHERE calculation_date = %s
        ORDER BY rank
        LIMIT %s
    """
    
    df_scores = pd.read_sql(query_scores, conn, params=(latest_date, top_n))
    conn.close()
    
    if df_scores.empty:
        raise ValueError(f"No alpha scores found for date {latest_date}")
    
    print(f"\n📊 Top {top_n} stocks from DB:")
    print(df_scores[['rank', 'stock_code', 'stock_name', 'alpha_score', 'close_price']].to_string(index=False))
    
    return df_scores, latest_date

def save_trading_signals(df_scores, signal_date, target_amount=5000000):
    """매매 신호를 DB에 저장"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # 균등 분할
    num_stocks = len(df_scores)
    amount_per_stock = target_amount // num_stocks
    
    try:
        # 기존 신호 삭제
        cur.execute("DELETE FROM trading_signals WHERE signal_date = %s", (signal_date,))
        
        # 신호 저장
        values = [
            (
                signal_date,
                row['stock_code'],
                row['stock_name'],
                'BUY',
                row['alpha_score'],
                row['rank'],
                1.0 / num_stocks,  # target_weight
                f"Alpha rank #{row['rank']}, score={row['alpha_score']:.6f}"
            )
            for _, row in df_scores.iterrows()
        ]
        
        execute_values(cur, """
            INSERT INTO trading_signals
            (signal_date, stock_code, stock_name, signal_type, alpha_score, rank, target_weight, reason)
            VALUES %s
        """, values)
        
        conn.commit()
        print(f"✅ Saved {len(values)} trading signals to database")
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def update_signal_execution(stock_code, signal_date, price, quantity):
    """매매 신호 실행 기록 업데이트"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            UPDATE trading_signals
            SET executed = TRUE,
                execution_time = NOW(),
                execution_price = %s,
                execution_quantity = %s
            WHERE signal_date = %s AND stock_code = %s
        """, (price, quantity, signal_date, stock_code))
        
        conn.commit()
    finally:
        cur.close()
        conn.close()

def execute_trades(df_scores, dry_run=False, target_amount=5000000):
    """
    매수 주문 실행
    
    Args:
        df_scores: 매수할 종목 DataFrame
        dry_run: True면 실제 주문 없이 시뮬레이션만
        target_amount: 총 투자 금액
    """
    # KIS API 초기화
    api = KISAPI(
        app_key=os.getenv('KIS_APP_KEY'),
        app_secret=os.getenv('KIS_APP_SECRET'),
        account_no=os.getenv('KIS_ACCOUNT_NO'),
        is_real=os.getenv('KIS_MODE', 'real') == 'real'
    )
    
    trader = AlphaTrader(api)
    
    # 균등 분할
    num_stocks = len(df_scores)
    amount_per_stock = target_amount // num_stocks
    
    print(f"\n💰 Investment Plan:")
    print(f"Total amount: {target_amount:,}원")
    print(f"Per stock: {amount_per_stock:,}원")
    print(f"Number of stocks: {num_stocks}")
    print(f"Mode: {'DRY RUN' if dry_run else 'REAL TRADING'}")
    
    if not dry_run:
        confirm = input("\n⚠️  Real trading mode! Continue? (yes/no): ")
        if confirm.lower() != 'yes':
            print("❌ Trading cancelled")
            return
    
    # 매수 실행
    results = []
    signal_date = date.today()
    
    for _, row in df_scores.iterrows():
        stock_code = row['stock_code']
        stock_name = row['stock_name']
        
        try:
            if dry_run:
                # 시뮬레이션
                current_price = row['close_price']
                quantity = int(amount_per_stock / current_price)
                print(f"\n[DRY RUN] {stock_name} ({stock_code}): {quantity}주 @ {current_price:,}원")
                results.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'quantity': quantity,
                    'price': current_price,
                    'success': True
                })
            else:
                # 실제 주문
                result = trader.buy_stock(stock_code, amount_per_stock)
                if result['success']:
                    print(f"✅ {stock_name} ({stock_code}): {result['quantity']}주 매수 완료")
                    # DB 업데이트
                    update_signal_execution(
                        stock_code, signal_date,
                        result['price'], result['quantity']
                    )
                else:
                    print(f"❌ {stock_name} ({stock_code}): 매수 실패 - {result.get('message', 'Unknown error')}")
                
                results.append(result)
                
        except Exception as e:
            print(f"❌ Error trading {stock_name} ({stock_code}): {e}")
            results.append({
                'stock_code': stock_code,
                'stock_name': stock_name,
                'success': False,
                'message': str(e)
            })
    
    # 결과 요약
    success_count = sum(1 for r in results if r.get('success', False))
    print(f"\n📊 Trading Summary:")
    print(f"Total orders: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    
    return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Trade from DB alpha scores')
    parser.add_argument('--top-n', type=int, default=15, help='Number of top stocks to trade')
    parser.add_argument('--amount', type=int, default=5000000, help='Total investment amount in KRW')
    parser.add_argument('--dry-run', action='store_true', help='Simulation mode (no real orders)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Alpha-GPT-KR: Trade from Database")
    print("=" * 60)
    print(f"Date: {date.today()}")
    print(f"Top N stocks: {args.top_n}")
    print(f"Target amount: {args.amount:,}원")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'REAL TRADING'}")
    print("=" * 60)
    
    try:
        # DB에서 알파 스코어 로드
        df_scores, calc_date = load_latest_alpha_scores(args.top_n)
        
        # 매매 신호 저장
        save_trading_signals(df_scores, date.today(), args.amount)
        
        # 매수 실행
        results = execute_trades(df_scores, dry_run=args.dry_run, target_amount=args.amount)
        
        print("\n✅ Trading completed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
