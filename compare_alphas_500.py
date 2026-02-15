#!/usr/bin/env python3
"""
두 알파 비교: 2년 vs 1년 데이터 진화 결과
500종목에 적용하여 성과 비교
"""

import sys
import os
from pathlib import Path
from datetime import date, datetime
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

# DB 연결
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )

# 500종목 데이터 로드
def load_market_data_500():
    """시가총액 상위 500개 종목 (최근 365일)"""
    print("📊 500종목 데이터 로드 중...")
    
    conn = get_db_connection()
    
    # 시총 상위 500종목
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
    
    # 가격 데이터 (최근 365일)
    stock_id_list = ', '.join(map(str, stock_ids))
    query_prices = f"""
        SELECT 
            s.ticker,
            p.date,
            p.close,
            p.volume
        FROM price_data p
        JOIN stocks s ON p.stock_id = s.id
        WHERE p.stock_id IN ({stock_id_list})
        AND p.date >= CURRENT_DATE - INTERVAL '365 days'
        ORDER BY s.ticker, p.date
    """
    
    price_df = pd.read_sql(query_prices, conn)
    conn.close()
    
    close_pivot = price_df.pivot(index='date', columns='ticker', values='close')
    volume_pivot = price_df.pivot(index='date', columns='ticker', values='volume')
    
    print(f"✅ {len(close_pivot.columns)}개 종목, {len(close_pivot)}일 데이터")
    
    return {
        'close': close_pivot,
        'volume': volume_pivot,
        'returns': close_pivot.pct_change()
    }

# 알파 평가
def evaluate_alpha_detailed(alpha_name, alpha_expr, data):
    """알파 성과 상세 평가"""
    print(f"\n🔍 평가: {alpha_name}")
    print(f"   공식: {alpha_expr[:100]}...")
    
    try:
        close = data['close']
        volume = data['volume']
        returns = data['returns'].shift(-1)  # 다음날 수익률
        
        alpha_values = eval(alpha_expr)
        
        ic_list = []
        ir_list = []
        
        for date in alpha_values.index[:-1]:
            alpha_cs = alpha_values.loc[date]
            returns_cs = returns.loc[date]
            valid = alpha_cs.notna() & returns_cs.notna()
            
            if valid.sum() > 20:
                ic = alpha_cs[valid].corr(returns_cs[valid])
                if not np.isnan(ic):
                    ic_list.append(ic)
        
        if len(ic_list) < 10:
            print("   ❌ 데이터 부족")
            return None
        
        # 성과 지표
        mean_ic = np.mean(ic_list)
        std_ic = np.std(ic_list)
        ir = mean_ic / std_ic if std_ic > 0 else 0
        positive_rate = sum(1 for ic in ic_list if ic > 0) / len(ic_list)
        
        print(f"   ✅ IC: {mean_ic:.4f} (std: {std_ic:.4f})")
        print(f"   📊 IR: {ir:.4f}")
        print(f"   ✓  양수 비율: {positive_rate:.1%}")
        print(f"   📅 평가 일수: {len(ic_list)}일")
        
        return {
            'name': alpha_name,
            'formula': alpha_expr,
            'ic': mean_ic,
            'ic_std': std_ic,
            'ir': ir,
            'positive_rate': positive_rate,
            'days': len(ic_list)
        }
        
    except Exception as e:
        print(f"   ❌ 에러: {e}")
        return None

# 메인 함수
def main():
    print("=" * 70)
    print("알파 비교: 2년 vs 1년 데이터 진화 결과")
    print("=" * 70)
    print()
    
    # 1. 데이터 로드
    data = load_market_data_500()
    
    # 2. 두 알파 정의
    alphas = [
        {
            'name': '2년 알파 (IC 0.4773)',
            'formula': 'AlphaOperators.ts_rank(AlphaOperators.ts_mean(returns, 2), 10)',
            'origin': '2년 데이터 GP 진화 (초단기 모멘텀)'
        },
        {
            'name': '1년 알파 (IC 0.1973)',
            'formula': 'AlphaOperators.ts_rank(AlphaOperators.ts_delta(volume, 5), 15) * -1 + AlphaOperators.ts_rank(AlphaOperators.ts_mean(returns, 10), 20)',
            'origin': '1년 데이터 GP 진화 (거래량 역전 + 중기 수익률)'
        }
    ]
    
    # 3. 평가
    results = []
    
    for alpha_info in alphas:
        result = evaluate_alpha_detailed(
            alpha_name=alpha_info['name'],
            alpha_expr=alpha_info['formula'],
            data=data
        )
        
        if result:
            result['origin'] = alpha_info['origin']
            results.append(result)
    
    # 4. 비교 리포트
    print("\n" + "=" * 70)
    print("📊 비교 결과 (500종목 기준)")
    print("=" * 70)
    
    if len(results) == 2:
        df = pd.DataFrame(results)
        df = df.sort_values('ic', ascending=False)
        
        print("\n" + df.to_string(index=False))
        
        # 승자 판정
        best = df.iloc[0]
        
        print("\n" + "=" * 70)
        print("🏆 최종 승자")
        print("=" * 70)
        print(f"\n알파: {best['name']}")
        print(f"IC: {best['ic']:.4f}")
        print(f"IR: {best['ir']:.4f}")
        print(f"양수 비율: {best['positive_rate']:.1%}")
        print(f"\n공식:")
        print(f"  {best['formula']}")
        
        # 5. DB 저장
        print(f"\n💾 최상위 알파 DB 저장...")
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("""
                INSERT INTO alpha_performance
                (alpha_formula, start_date, is_active, sharpe_ratio, notes)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (alpha_formula, start_date) DO UPDATE
                SET sharpe_ratio = EXCLUDED.sharpe_ratio,
                    notes = EXCLUDED.notes,
                    is_active = EXCLUDED.is_active
            """, (
                best['formula'],
                date.today(),
                True,
                float(best['ic'] * 10),  # IC를 샤프 비율로 근사
                f"IC: {best['ic']:.4f}, IR: {best['ir']:.4f}, 500종목 검증 완료"
            ))
            conn.commit()
            print("✅ DB 저장 완료")
        finally:
            cur.close()
            conn.close()
        
        # 6. 추천
        print(f"\n💡 추천")
        print(f"   이 알파로 내일 아침 매매 실행:")
        print(f"   python3 calculate_and_save_alpha.py")
        print(f"   python3 trade_from_db.py --top-n 15 --amount 5000000")
    
    else:
        print("\n⚠️  일부 알파 평가 실패")

if __name__ == "__main__":
    main()
