#!/usr/bin/env python3
"""
빠른 테스트: 샘플 알파 데이터 생성 및 DB 저장
"""
import os
import sys
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

# 환경 변수 로드
load_dotenv()

# 샘플 데이터
SAMPLE_STOCKS = [
    ('005930', '삼성전자', 0.0254, 1000000000000, 72000, 15000000),
    ('000660', 'SK하이닉스', 0.0239, 800000000000, 145000, 5000000),
    ('035420', 'NAVER', 0.0198, 500000000000, 198000, 800000),
    ('051910', 'LG화학', 0.0187, 450000000000, 385000, 400000),
    ('006400', '삼성SDI', 0.0176, 400000000000, 448000, 350000),
    ('035720', '카카오', 0.0165, 380000000000, 42500, 3000000),
    ('005380', '현대차', 0.0154, 350000000000, 245000, 600000),
    ('000270', '기아', 0.0143, 340000000000, 115000, 1200000),
    ('068270', '셀트리온', 0.0132, 320000000000, 178000, 800000),
    ('207940', '삼성바이오로직스', 0.0121, 310000000000, 895000, 150000),
    ('105560', 'KB금융', 0.0110, 300000000000, 82500, 2000000),
    ('055550', '신한지주', 0.0099, 290000000000, 50000, 2500000),
    ('012330', '현대모비스', 0.0088, 280000000000, 245000, 500000),
    ('086790', '하나금융지주', 0.0077, 270000000000, 67500, 1800000),
    ('066570', 'LG전자', 0.0066, 260000000000, 88000, 1500000),
]

def get_db_connection():
    """PostgreSQL 연결"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )

def save_alpha_scores():
    """샘플 알파 스코어 저장"""
    calc_date = date.today()
    alpha_formula = "ops.ts_delta(close, 26)"
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # 기존 데이터 삭제
        cur.execute("DELETE FROM alpha_scores WHERE calculation_date = %s", (calc_date,))
        
        # 샘플 데이터 삽입
        values = [
            (
                calc_date,
                code,
                name,
                alpha_formula,
                score,
                rank + 1,
                market_cap,
                price,
                volume
            )
            for rank, (code, name, score, market_cap, price, volume) in enumerate(SAMPLE_STOCKS)
        ]
        
        execute_values(cur, """
            INSERT INTO alpha_scores 
            (calculation_date, stock_code, stock_name, alpha_formula, alpha_score, 
             rank, market_cap, close_price, volume)
            VALUES %s
        """, values)
        
        conn.commit()
        print(f"✅ {len(SAMPLE_STOCKS)}개 알파 스코어 저장 완료")
        
        # 확인
        cur.execute("""
            SELECT rank, stock_code, stock_name, alpha_score, close_price
            FROM alpha_scores
            WHERE calculation_date = %s
            ORDER BY rank
            LIMIT 10
        """, (calc_date,))
        
        print("\n📊 저장된 데이터:")
        for row in cur.fetchall():
            print(f"  {row[0]:2d}. {row[1]} {row[2]:15s} α={row[3]:.4f} 가격={row[4]:,}원")
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()

def save_trading_signals():
    """매매 신호 저장 (상위 15개)"""
    signal_date = date.today()
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # 기존 신호 삭제
        cur.execute("DELETE FROM trading_signals WHERE signal_date = %s", (signal_date,))
        
        # 상위 15개 종목 신호 생성
        values = [
            (
                signal_date,
                code,
                name,
                'BUY',
                score,
                rank + 1,
                1.0 / 15,  # target_weight
                f"Alpha rank #{rank+1}, score={score:.6f}"
            )
            for rank, (code, name, score, _, _, _) in enumerate(SAMPLE_STOCKS[:15])
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
    print("=" * 60)
    print("테스트 알파 데이터 생성")
    print("=" * 60)
    
    try:
        save_alpha_scores()
        save_trading_signals()
        print("\n✅ 모든 데이터 저장 완료!")
        print("\n📊 대시보드에서 확인: http://localhost:9999/dashboard.html")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
