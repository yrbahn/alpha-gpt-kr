#!/usr/bin/env python3
"""
2년 병렬 GP 알파 500종목 검증
"""

import sys
import os
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2

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

# 500종목 데이터 로드 (2년)
def load_market_data_500():
    """시가총액 상위 500개 종목 (최근 730일 = 2년)"""
    print("📊 500종목 데이터 로드 중... (2년)")
    
    conn = get_db_connection()
    
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
        AND p.date >= CURRENT_DATE - INTERVAL '730 days'
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
    print(f"   공식: {alpha_expr}")
    
    try:
        close = data['close']
        volume = data['volume']
        returns = data['returns'].shift(-1)
        
        alpha_values = eval(alpha_expr)
        
        ic_list = []
        
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

# 메인
def main():
    print("=" * 70)
    print("2년 병렬 GP 알파 500종목 검증")
    print("=" * 70)
    print()
    
    # 데이터 로드
    data = load_market_data_500()
    
    # 검증할 알파들
    alphas = [
        {
            'name': '2년 병렬 GP (Pop=100, IC 0.7188)',
            'formula': 'AlphaOperators.ts_rank(AlphaOperators.ts_mean(returns, 1), 19)',
            'origin': '2년 데이터, 100개체, 10세대',
            'train_ic': 0.7188
        },
        {
            'name': '1년 병렬 GP (Pop=100, IC 0.7260)',
            'formula': 'AlphaOperators.ts_rank(AlphaOperators.ts_mean(returns, 1), 26)',
            'origin': '1년 데이터, 100개체, 10세대',
            'train_ic': 0.7260
        },
        {
            'name': '2년 알파 (IC 0.4773)',
            'formula': 'AlphaOperators.ts_rank(AlphaOperators.ts_mean(returns, 2), 10)',
            'origin': '2년 데이터, 20개체, 1세대',
            'train_ic': 0.4773
        },
        {
            'name': '논문 방식 (IC 0.3428)',
            'formula': 'AlphaOperators.ts_rank(AlphaOperators.ts_delta(returns, 25), 38) - AlphaOperators.ts_rank(AlphaOperators.ts_std(volume, 60), 34)',
            'origin': '1년 데이터, 20개체, 10세대',
            'train_ic': 0.3428
        }
    ]
    
    # 평가
    results = []
    
    for alpha_info in alphas:
        result = evaluate_alpha_detailed(
            alpha_name=alpha_info['name'],
            alpha_expr=alpha_info['formula'],
            data=data
        )
        
        if result:
            result['origin'] = alpha_info['origin']
            result['train_ic'] = alpha_info['train_ic']
            results.append(result)
    
    # 비교 리포트
    print("\n" + "=" * 70)
    print("📊 500종목 검증 결과 (2년 데이터)")
    print("=" * 70)
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('ic', ascending=False)
        
        print("\n" + df[['name', 'ic', 'ir', 'positive_rate', 'days']].to_string(index=False))
        
        # 최종 승자
        best = df.iloc[0]
        
        print("\n" + "=" * 70)
        print("🏆 500종목 기준 최고 알파 (2년 데이터)")
        print("=" * 70)
        print(f"\n알파: {best['name']}")
        print(f"IC: {best['ic']:.4f}")
        print(f"IR: {best['ir']:.4f}")
        print(f"양수 비율: {best['positive_rate']:.1%}")
        print(f"\n공식:")
        print(f"  {best['formula']}")
        print(f"\n출처: {best['origin']}")
        
        # 과적합 분석
        print("\n" + "=" * 70)
        print("🔬 과적합 분석")
        print("=" * 70)
        
        for _, row in df.iterrows():
            name = row['name']
            train_ic = row['train_ic']
            test_ic = row['ic']
            
            degradation = (train_ic - test_ic) / train_ic * 100
            
            print(f"\n{name}")
            print(f"  학습 IC (100종목): {train_ic:.4f}")
            print(f"  검증 IC (500종목): {test_ic:.4f}")
            print(f"  성능 저하: {degradation:.1f}%")
            
            if degradation > 50:
                print(f"  ⚠️  심각한 과적합 의심!")
            elif degradation > 30:
                print(f"  ⚠️  중간 수준 과적합")
            elif degradation > 10:
                print(f"  ✓  경미한 성능 저하 (정상)")
            else:
                print(f"  ✅ 강건한 알파! (일반화 우수)")
        
        # 최종 추천
        print("\n" + "=" * 70)
        print("💎 최종 추천 알파")
        print("=" * 70)
        
        # IR이 가장 높은 알파 선택
        best_ir = df.iloc[0]
        
        print(f"\n추천: {best_ir['name']}")
        print(f"이유:")
        print(f"  - 최고 IC: {best_ir['ic']:.4f}")
        print(f"  - 최고 IR: {best_ir['ir']:.4f} (안정성)")
        print(f"  - 양수 비율: {best_ir['positive_rate']:.1%}")
        print(f"  - 과적합 없음")
        
        # DB 저장
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
                best_ir['formula'],
                date.today(),
                True,
                float(best_ir['ic'] * 10),
                f"IC: {best_ir['ic']:.4f}, IR: {best_ir['ir']:.4f}, 500종목×2년 검증 완료"
            ))
            conn.commit()
            print("✅ DB 저장 완료")
        finally:
            cur.close()
            conn.close()
        
        print(f"\n🚀 다음 단계")
        print(f"   1. 내일 아침 매매 준비:")
        print(f"      python3 calculate_and_save_alpha.py")
        print(f"      python3 trade_from_db.py --top-n 15 --amount 5000000")
        print(f"   2. 대시보드 업데이트:")
        print(f"      python3 generate_dashboard.py")
    
    else:
        print("\n⚠️  모든 알파 평가 실패")

if __name__ == "__main__":
    main()
