#!/usr/bin/env python3
"""
모든 생성된 알파 500종목 검증
"""

import sys
import os
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2

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

def load_comprehensive_data_500():
    """500종목 종합 데이터 (2년)"""
    print("📊 500종목 종합 데이터 로드 중... (2년)")
    
    conn = get_db_connection()
    
    query_stocks = """
        SELECT DISTINCT ON (s.ticker)
            s.id, s.ticker, s.name
        FROM stocks s
        JOIN price_data p ON s.id = p.stock_id
        WHERE s.is_active = true
        AND p.date = (SELECT MAX(date) FROM price_data)
        ORDER BY s.ticker, (p.close * p.volume) DESC
        LIMIT 500
    """
    stocks_df = pd.read_sql(query_stocks, conn)
    stock_ids = stocks_df['id'].tolist()
    stock_id_list = ', '.join(map(str, stock_ids))
    
    # 가격
    query_price = f"""
        SELECT s.ticker, p.date, p.close, p.volume
        FROM price_data p
        JOIN stocks s ON p.stock_id = s.id
        WHERE p.stock_id IN ({stock_id_list})
        AND p.date >= CURRENT_DATE - INTERVAL '730 days'
        ORDER BY s.ticker, p.date
    """
    price_df = pd.read_sql(query_price, conn)
    close = price_df.pivot(index='date', columns='ticker', values='close')
    volume = price_df.pivot(index='date', columns='ticker', values='volume')
    
    # 기술적 지표
    query_tech = f"""
        SELECT s.ticker, t.date, t.macd, t.macd_signal, t.rsi_14, t.volatility_20d
        FROM technical_indicators t
        JOIN stocks s ON t.stock_id = s.id
        WHERE t.stock_id IN ({stock_id_list})
        AND t.date >= CURRENT_DATE - INTERVAL '730 days'
        ORDER BY s.ticker, t.date
    """
    tech_df = pd.read_sql(query_tech, conn)
    macd = tech_df.pivot(index='date', columns='ticker', values='macd')
    macd_signal = tech_df.pivot(index='date', columns='ticker', values='macd_signal')
    rsi = tech_df.pivot(index='date', columns='ticker', values='rsi_14')
    volatility = tech_df.pivot(index='date', columns='ticker', values='volatility_20d')
    
    # 재무
    query_fin = f"""
        SELECT s.ticker, f.period_end as date,
               f.net_income, f.operating_income, f.total_assets
        FROM financial_statements f
        JOIN stocks s ON f.stock_id = s.id
        WHERE f.stock_id IN ({stock_id_list})
        AND f.period_end >= CURRENT_DATE - INTERVAL '730 days'
        AND f.revenue IS NOT NULL
        ORDER BY s.ticker, f.period_end
    """
    fin_df = pd.read_sql(query_fin, conn)
    net_income = fin_df.pivot(index='date', columns='ticker', values='net_income')
    operating_income = fin_df.pivot(index='date', columns='ticker', values='operating_income')
    total_assets = fin_df.pivot(index='date', columns='ticker', values='total_assets')
    
    # Forward fill
    all_dates = close.index
    net_income = net_income.reindex(all_dates).fillna(method='ffill')
    operating_income = operating_income.reindex(all_dates).fillna(method='ffill')
    total_assets = total_assets.reindex(all_dates).fillna(method='ffill')
    
    conn.close()
    
    print(f"✅ {len(close.columns)}개 종목, {len(close)}일 데이터")
    
    return {
        'close': close, 'volume': volume, 'returns': close.pct_change(),
        'macd': macd, 'macd_signal': macd_signal, 'rsi': rsi, 'volatility': volatility,
        'net_income': net_income, 'operating_income': operating_income, 'total_assets': total_assets
    }

def evaluate_alpha(alpha_name, alpha_expr, data):
    """알파 평가"""
    print(f"\n🔍 {alpha_name}")
    print(f"   공식: {alpha_expr[:80]}...")
    
    try:
        close = data['close']
        volume = data['volume']
        returns = data['returns']
        macd = data['macd']
        macd_signal = data['macd_signal']
        rsi = data['rsi']
        volatility = data['volatility']
        net_income = data['net_income']
        operating_income = data['operating_income']
        total_assets = data['total_assets']
        
        # 10일 후 수익률
        returns_forward_10 = close.pct_change(10).shift(-10)
        
        alpha_values = eval(alpha_expr)
        
        ic_list = []
        for date in alpha_values.index[:-10]:
            alpha_cs = alpha_values.loc[date]
            returns_cs = returns_forward_10.loc[date]
            valid = alpha_cs.notna() & returns_cs.notna() & (alpha_cs != np.inf) & (alpha_cs != -np.inf)
            
            if valid.sum() > 20:
                ic = alpha_cs[valid].corr(returns_cs[valid])
                if not np.isnan(ic):
                    ic_list.append(ic)
        
        if len(ic_list) < 10:
            print("   ❌ 데이터 부족")
            return None
        
        mean_ic = np.mean(ic_list)
        std_ic = np.std(ic_list)
        ir = mean_ic / std_ic if std_ic > 0 else 0
        positive_rate = sum(1 for ic in ic_list if ic > 0) / len(ic_list)
        
        print(f"   ✅ IC: {mean_ic:.4f} (IR: {ir:.2f}, 양수: {positive_rate:.1%})")
        
        return {
            'name': alpha_name,
            'formula': alpha_expr,
            'ic': mean_ic,
            'ir': ir,
            'positive_rate': positive_rate,
            'days': len(ic_list)
        }
        
    except Exception as e:
        print(f"   ❌ 에러: {e}")
        return None

def main():
    print("=" * 70)
    print("모든 알파 500종목 최종 검증")
    print("=" * 70)
    print()
    
    data = load_comprehensive_data_500()
    
    # 검증할 알파들
    alphas = [
        {
            'name': '🥇 종합 알파 (27개 지표)',
            'formula': 'AlphaOperators.ts_rank(macd - macd_signal, 20) * AlphaOperators.ts_mean(volume, 10) / AlphaOperators.ts_std(volume, 20)',
            'train_ic': 0.0781
        },
        {
            'name': '🥈 고급 알파 (샤프)',
            'formula': 'AlphaOperators.ts_rank(AlphaOperators.ts_mean(returns, 10) / volatility, 30)',
            'train_ic': 0.0477
        },
        {
            'name': '🥉 재무 알파 (ROA)',
            'formula': 'AlphaOperators.normed_rank((net_income / total_assets) + (operating_income / total_assets))',
            'train_ic': 0.0500
        }
    ]
    
    results = []
    
    for alpha_info in alphas:
        result = evaluate_alpha(
            alpha_name=alpha_info['name'],
            alpha_expr=alpha_info['formula'],
            data=data
        )
        
        if result:
            result['train_ic'] = alpha_info['train_ic']
            results.append(result)
    
    # 결과
    print("\n" + "=" * 70)
    print("📊 500종목 최종 검증 결과")
    print("=" * 70)
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('ic', ascending=False)
        
        print("\n" + df[['name', 'ic', 'ir', 'positive_rate']].to_string(index=False))
        
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
            
            if degradation < 10:
                status = "✅ 강건한 알파!"
            elif degradation < 30:
                status = "✓  정상 범위"
            else:
                status = "⚠️  과적합 의심"
            
            print(f"  {status}")
        
        # 최종 추천
        best = df.iloc[0]
        
        print("\n" + "=" * 70)
        print("🏆 최종 추천 알파 (500종목 검증 완료)")
        print("=" * 70)
        
        print(f"\n알파: {best['name']}")
        print(f"IC: {best['ic']:.4f}")
        print(f"IR: {best['ir']:.2f}")
        print(f"양수 비율: {best['positive_rate']:.1%}")
        print(f"\n공식:")
        print(f"  {best['formula']}")
        
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
                SET sharpe_ratio = EXCLUDED.sharpe_ratio, notes = EXCLUDED.notes, is_active = EXCLUDED.is_active
            """, (
                best['formula'],
                date.today(),
                True,
                float(best['ic'] * 10),
                f"IC: {best['ic']:.4f}, IR: {best['ir']:.2f}, 500종목 최종 검증 완료"
            ))
            conn.commit()
            print("✅ DB 저장 완료")
        finally:
            cur.close()
            conn.close()
        
        print(f"\n🚀 내일 아침 매매 준비")
        print(f"   1. 알파 계산:")
        print(f"      python3 calculate_and_save_alpha.py")
        print(f"   2. 매매 시뮬레이션:")
        print(f"      python3 simple_trade_from_db.py --top-n 15 --amount 5000000 --dry-run")
        print(f"   3. 실전 매매:")
        print(f"      python3 simple_trade_from_db.py --top-n 15 --amount 5000000")

if __name__ == "__main__":
    main()
