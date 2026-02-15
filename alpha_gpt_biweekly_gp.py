#!/usr/bin/env python3
"""
Alpha-GPT + GP: Bi-weekly Rebalancing (15-day forward)
LLM Seed Generation + Genetic Programming Evolution
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
from concurrent.futures import ProcessPoolExecutor, as_completed

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.agents.quant_developer import QuantDeveloper
from alpha_gpt_kr.mining.genetic_programming import AlphaGeneticProgramming
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
    """PostgreSQL에서 데이터 로드"""
    conn = get_db_connection()
    
    query_stocks = f"""
        SELECT ticker, market_cap
        FROM stocks 
        WHERE market_cap IS NOT NULL 
        ORDER BY market_cap DESC 
        LIMIT {limit_stocks}
    """
    stocks = pd.read_sql(query_stocks, conn)
    stock_list = stocks['ticker'].tolist()
    
    query_price = f"""
        SELECT 
            stock_code, date, close, volume, high, low, open
        FROM price_data
        WHERE stock_code = ANY(%s)
        AND date >= CURRENT_DATE - INTERVAL '{years} years'
        ORDER BY stock_code, date
    """
    df_price = pd.read_sql(query_price, conn, params=(stock_list,))
    
    query_tech = f"""
        SELECT 
            stock_code, date,
            rsi_14, macd, macd_signal,
            bb_upper, bb_middle, bb_lower,
            sma_5, sma_20, sma_60, volatility_20
        FROM technical_indicators
        WHERE stock_code = ANY(%s)
        AND date >= CURRENT_DATE - INTERVAL '{years} years'
        ORDER BY stock_code, date
    """
    df_tech = pd.read_sql(query_tech, conn, params=(stock_list,))
    
    query_supply = f"""
        SELECT 
            stock_code, date,
            foreign_net_buy, institution_net_buy,
            foreign_ownership_ratio, institution_ownership_ratio,
            short_ratio
        FROM supply_demand_data
        WHERE stock_code = ANY(%s)
        AND date >= CURRENT_DATE - INTERVAL '{years} years'
        ORDER BY stock_code, date
    """
    df_supply = pd.read_sql(query_supply, conn, params=(stock_list,))
    
    conn.close()
    
    df = df_price.merge(df_tech, on=['stock_code', 'date'], how='left')
    df = df.merge(df_supply, on=['stock_code', 'date'], how='left')
    
    df = df.sort_values(['stock_code', 'date'])
    df['returns'] = df.groupby('stock_code')['close'].pct_change()
    df['forward_return_15d'] = df.groupby('stock_code')['close'].shift(-15) / df['close'] - 1
    
    df = df.dropna(subset=['forward_return_15d'])
    
    print(f"\n✅ Data loaded:")
    print(f"   Stocks: {df['stock_code'].nunique()}")
    print(f"   Days: {df['date'].nunique()}")
    print(f"   Total rows: {len(df):,}")
    print(f"   Date range: {df['date'].min()} ~ {df['date'].max()}")
    
    return df

def fitness_function_wrapper(alpha_expr, df):
    """Fitness = IC (15-day forward return)"""
    try:
        alpha_values = eval(alpha_expr)
        
        ic_values = []
        for date in df['date'].unique():
            df_date = df[df['date'] == date].copy()
            if len(df_date) < 30:
                continue
            
            df_date['alpha'] = alpha_values[df['date'] == date]
            corr = df_date[['alpha', 'forward_return_15d']].corr(method='spearman').iloc[0, 1]
            
            if not np.isnan(corr):
                ic_values.append(corr)
        
        ic = np.mean(ic_values) if ic_values else 0
        return ic
        
    except:
        return 0.0

def main():
    print("=" * 80)
    print("Alpha-GPT + GP: Bi-weekly Rebalancing (15-day forward)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 데이터 로드
    print("📊 Loading data from PostgreSQL...")
    df = load_data_from_postgres(limit_stocks=500, years=2)
    
    # Phase 1: LLM Seed Generation
    print("\n" + "=" * 80)
    print("PHASE 1: LLM Seed Generation")
    print("=" * 80)
    
    quant_dev = QuantDeveloper(
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4'
    )
    
    prompt = """
Generate alpha factors for 15-day forward prediction (bi-weekly rebalancing).

Available: close, volume, returns, rsi_14, macd, macd_signal, bb_upper, bb_middle, bb_lower,
sma_5, sma_20, sma_60, volatility_20, foreign_net_buy, institution_net_buy,
foreign_ownership_ratio, institution_ownership_ratio, short_ratio

Focus on medium-term momentum (15 days), trend-following, and supply-demand factors.

Generate 20 diverse expressions using:
- AlphaOperators.ts_rank(x, window)
- AlphaOperators.ts_mean(x, window)
- AlphaOperators.ts_std(x, window)
- AlphaOperators.normed_rank(x)
"""
    
    seed_alphas = quant_dev.generate_alpha_ideas(prompt, num_ideas=20)
    print(f"✅ Generated {len(seed_alphas)} seed alphas")
    
    # Phase 2: Genetic Programming Evolution
    print("\n" + "=" * 80)
    print("PHASE 2: Genetic Programming Evolution")
    print("=" * 80)
    print("Population: 100")
    print("Generations: 30")
    print("Crossover: 0.7 | Mutation: 0.2")
    print()
    
    gp = AlphaGeneticProgramming(
        fitness_func=lambda expr: fitness_function_wrapper(expr, df),
        population_size=100,
        generations=30,
        crossover_prob=0.7,
        mutation_prob=0.2
    )
    
    # GP 실행
    print("🧬 Starting evolution...")
    best_individual = gp.evolve(seed_alphas=seed_alphas)
    
    best_alpha = best_individual.expression
    best_ic = best_individual.fitness
    
    # 결과 출력
    print("\n" + "=" * 80)
    print("🏆 BEST ALPHA (15-day forward, Bi-weekly Rebalancing)")
    print("=" * 80)
    print(f"IC: {best_ic:.4f}")
    print(f"Expression: {best_alpha}")
    print()
    
    # Backtest simulation
    print("=" * 80)
    print("📊 Backtest Summary (15-day rebalancing)")
    print("=" * 80)
    
    alpha_values = eval(best_alpha)
    df['alpha'] = alpha_values
    
    # 15일마다 리밸런싱
    dates = sorted(df['date'].unique())
    rebalance_dates = dates[::15]  # 15일 간격
    
    print(f"Total rebalancing events: {len(rebalance_dates)}")
    print(f"Average holding period: 15 days")
    
    # IC by rebalancing period
    ic_by_period = []
    for i, date in enumerate(rebalance_dates):
        df_date = df[df['date'] == date].copy()
        if len(df_date) < 30:
            continue
        
        corr = df_date[['alpha', 'forward_return_15d']].corr(method='spearman').iloc[0, 1]
        if not np.isnan(corr):
            ic_by_period.append(corr)
    
    print(f"\nIC Statistics:")
    print(f"  Mean IC: {np.mean(ic_by_period):.4f}")
    print(f"  IC Std: {np.std(ic_by_period):.4f}")
    print(f"  IC > 0: {sum(1 for ic in ic_by_period if ic > 0)}/{len(ic_by_period)} ({100*sum(1 for ic in ic_by_period if ic > 0)/len(ic_by_period):.1f}%)")
    
    # 거래비용 고려
    transaction_cost = 0.003  # 0.3%
    annual_rebalances = 365 / 15  # ~24회/년
    total_cost_per_year = transaction_cost * annual_rebalances * 2  # 매수+매도
    
    print(f"\nTransaction Cost Analysis:")
    print(f"  Per rebalance: {transaction_cost*100:.2f}%")
    print(f"  Rebalances per year: ~{annual_rebalances:.0f}")
    print(f"  Total cost per year: ~{total_cost_per_year*100:.1f}%")
    
    # DB 저장
    save_to_db = input("\n💾 Save to database? (y/n): ")
    
    if save_to_db.lower() == 'y':
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alpha_formulas (formula, ic_score, description, created_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (formula) DO UPDATE
            SET ic_score = EXCLUDED.ic_score, updated_at = NOW()
        """, (
            best_alpha,
            float(best_ic),
            '15-day forward alpha (bi-weekly rebalancing, LLM+GP evolved)'
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Saved!")
    
    print(f"\n🎉 Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
