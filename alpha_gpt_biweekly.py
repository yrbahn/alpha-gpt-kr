#!/usr/bin/env python3
"""
Alpha-GPT: Bi-weekly Rebalancing (15-day forward)
월 2회 리밸런싱 전략 (논문 표준)
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.agents.quant_developer import QuantDeveloper
from alpha_gpt_kr.mining.operators import AlphaOperators

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

def load_data_from_postgres(limit_stocks=500, years=2):
    """PostgreSQL에서 데이터 로드 (시총 상위 N개, 최근 N년)"""
    conn = get_db_connection()
    
    # 1. 시총 상위 종목 선택
    query_stocks = f"""
        SELECT ticker, market_cap
        FROM stocks 
        WHERE market_cap IS NOT NULL 
        ORDER BY market_cap DESC 
        LIMIT {limit_stocks}
    """
    stocks = pd.read_sql(query_stocks, conn)
    stock_list = stocks['ticker'].tolist()
    
    # 2. 가격 데이터
    query_price = f"""
        SELECT 
            stock_code,
            date,
            close,
            volume,
            high,
            low,
            open
        FROM price_data
        WHERE stock_code = ANY(%s)
        AND date >= CURRENT_DATE - INTERVAL '{years} years'
        ORDER BY stock_code, date
    """
    df_price = pd.read_sql(query_price, conn, params=(stock_list,))
    
    # 3. 기술적 지표
    query_tech = f"""
        SELECT 
            stock_code,
            date,
            rsi_14,
            macd,
            macd_signal,
            bb_upper,
            bb_middle,
            bb_lower,
            sma_5,
            sma_20,
            sma_60,
            volatility_20
        FROM technical_indicators
        WHERE stock_code = ANY(%s)
        AND date >= CURRENT_DATE - INTERVAL '{years} years'
        ORDER BY stock_code, date
    """
    df_tech = pd.read_sql(query_tech, conn, params=(stock_list,))
    
    # 4. 수급 데이터
    query_supply = f"""
        SELECT 
            stock_code,
            date,
            foreign_net_buy,
            institution_net_buy,
            foreign_ownership_ratio,
            institution_ownership_ratio,
            short_ratio
        FROM supply_demand_data
        WHERE stock_code = ANY(%s)
        AND date >= CURRENT_DATE - INTERVAL '{years} years'
        ORDER BY stock_code, date
    """
    df_supply = pd.read_sql(query_supply, conn, params=(stock_list,))
    
    conn.close()
    
    # 데이터 병합
    df = df_price.merge(df_tech, on=['stock_code', 'date'], how='left')
    df = df.merge(df_supply, on=['stock_code', 'date'], how='left')
    
    # 수익률 계산 (15-day forward return)
    df = df.sort_values(['stock_code', 'date'])
    df['returns'] = df.groupby('stock_code')['close'].pct_change()
    df['forward_return_15d'] = df.groupby('stock_code')['close'].shift(-15) / df['close'] - 1
    
    # NaN 제거
    df = df.dropna(subset=['forward_return_15d'])
    
    print(f"\n✅ Data loaded:")
    print(f"   Stocks: {df['stock_code'].nunique()}")
    print(f"   Days: {df['date'].nunique()}")
    print(f"   Total rows: {len(df):,}")
    print(f"   Date range: {df['date'].min()} ~ {df['date'].max()}")
    
    return df

def calculate_ic(df, alpha_expr):
    """Information Coefficient 계산"""
    try:
        # AlphaOperators로 알파 계산
        alpha_values = eval(alpha_expr)
        
        # IC 계산 (알파 vs 15일 수익률)
        ic_values = []
        for date in df['date'].unique():
            df_date = df[df['date'] == date].copy()
            if len(df_date) < 30:
                continue
            
            df_date['alpha'] = alpha_values[df['date'] == date]
            
            # Spearman correlation
            corr = df_date[['alpha', 'forward_return_15d']].corr(method='spearman').iloc[0, 1]
            if not np.isnan(corr):
                ic_values.append(corr)
        
        ic = np.mean(ic_values) if ic_values else 0
        return ic
        
    except Exception as e:
        return 0

def main():
    print("=" * 80)
    print("Alpha-GPT: Bi-weekly Rebalancing Strategy (15-day forward)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 데이터 로드
    print("📊 Loading data from PostgreSQL...")
    df = load_data_from_postgres(limit_stocks=500, years=2)
    
    # QuantDeveloper 초기화
    print("\n🤖 Initializing Quant Developer (LLM)...")
    quant_dev = QuantDeveloper(
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4'
    )
    
    # 프롬프트 (15일 전략 명시)
    prompt = """
Generate alpha factors for Korean stock market with **15-day forward prediction** (bi-weekly rebalancing).

**Available Features (27 indicators):**

Price & Volume:
- close, open, high, low
- volume
- returns (daily returns)

Technical Indicators:
- rsi_14 (RSI 14-day)
- macd, macd_signal (MACD and signal line)
- bb_upper, bb_middle, bb_lower (Bollinger Bands)
- sma_5, sma_20, sma_60 (Simple Moving Averages)
- volatility_20 (20-day volatility)

Supply & Demand (Korean market specific):
- foreign_net_buy (Foreign net buying)
- institution_net_buy (Institution net buying)
- foreign_ownership_ratio (Foreign ownership %)
- institution_ownership_ratio (Institution ownership %)
- short_ratio (Short selling ratio)

**Strategy Context:**
- Rebalancing: Bi-weekly (every 15 days)
- Holding period: 15 days
- Transaction cost: ~0.3% per trade
- Prefer medium-term momentum and trend-following strategies

**Operators:**
AlphaOperators.ts_rank(x, window)
AlphaOperators.ts_mean(x, window)
AlphaOperators.ts_std(x, window)
AlphaOperators.ts_corr(x, y, window)
AlphaOperators.normed_rank(x)

**Examples for 15-day strategies:**
1. Momentum with trend confirmation:
   AlphaOperators.ts_rank(returns, 15) * AlphaOperators.ts_rank(volume, 15)

2. Supply-demand with moving average:
   (foreign_net_buy + institution_net_buy) / AlphaOperators.ts_std(volume, 20)

3. MACD trend with volatility filter:
   AlphaOperators.ts_rank(macd - macd_signal, 15) / volatility_20

Generate 20 diverse, sophisticated alpha expressions focusing on 15-day prediction.
Combine technical, supply/demand, and volatility factors.
"""
    
    # LLM으로 알파 생성
    print("\n🧠 Generating seed alphas with LLM...")
    seed_alphas = quant_dev.generate_alpha_ideas(prompt, num_ideas=20)
    
    print(f"\n✅ Generated {len(seed_alphas)} seed alphas")
    
    # IC 평가
    print("\n📊 Evaluating seed alphas...")
    results = []
    
    for i, alpha_expr in enumerate(seed_alphas, 1):
        print(f"\n[{i}/{len(seed_alphas)}] Testing: {alpha_expr[:80]}...")
        
        ic = calculate_ic(df, alpha_expr)
        
        results.append({
            'alpha': alpha_expr,
            'ic': ic
        })
        
        print(f"   IC: {ic:.4f}")
    
    # 결과 정렬
    df_results = pd.DataFrame(results).sort_values('ic', ascending=False)
    
    # 상위 10개 출력
    print("\n" + "=" * 80)
    print("📈 Top 10 Alphas (15-day forward)")
    print("=" * 80)
    
    for i, row in df_results.head(10).iterrows():
        print(f"\n{i+1}. IC: {row['ic']:.4f}")
        print(f"   {row['alpha']}")
    
    # 베스트 알파 저장
    best_alpha = df_results.iloc[0]
    
    print("\n" + "=" * 80)
    print("🏆 BEST ALPHA (15-day forward)")
    print("=" * 80)
    print(f"IC: {best_alpha['ic']:.4f}")
    print(f"Expression: {best_alpha['alpha']}")
    
    # DB에 저장
    save_to_db = input("\n💾 Save best alpha to database? (y/n): ")
    
    if save_to_db.lower() == 'y':
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 알파 공식 저장
        cursor.execute("""
            INSERT INTO alpha_formulas (formula, ic_score, description, created_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (formula) DO UPDATE
            SET ic_score = EXCLUDED.ic_score, updated_at = NOW()
        """, (
            best_alpha['alpha'],
            float(best_alpha['ic']),
            '15-day forward alpha (bi-weekly rebalancing)'
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Saved to database!")
    
    print(f"\n🎉 Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
