#!/usr/bin/env python3
"""
Alpha-GPT: Bi-weekly Rebalancing (15-day forward) - Simplified
월 2회 리밸런싱 전략
"""
import sys
import os
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2
import openai
import random

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

def load_biweekly_data():
    """15일 forward return용 데이터 로드"""
    print("📊 15일 forward 데이터 로드 중...")
    
    conn = get_db_connection()
    
    # 시총 상위 500
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
    
    # 가격 데이터 (2년)
    query_price = f"""
        SELECT s.ticker, p.date, p.close, p.open, p.high, p.low, p.volume
        FROM price_data p
        JOIN stocks s ON p.stock_id = s.id
        WHERE p.stock_id IN ({stock_id_list})
        AND p.date >= CURRENT_DATE - INTERVAL '730 days'
        ORDER BY s.ticker, p.date
    """
    price_df = pd.read_sql(query_price, conn)
    close = price_df.pivot(index='date', columns='ticker', values='close')
    open_px = price_df.pivot(index='date', columns='ticker', values='open')
    high = price_df.pivot(index='date', columns='ticker', values='high')
    low = price_df.pivot(index='date', columns='ticker', values='low')
    volume = price_df.pivot(index='date', columns='ticker', values='volume')
    
    # 기술적 지표
    query_tech = f"""
        SELECT s.ticker, t.date, t.rsi_14, t.macd, t.macd_signal,
               t.bb_upper, t.bb_middle, t.bb_lower,
               t.sma_20, t.sma_50, t.volatility_20d
        FROM technical_indicators t
        JOIN stocks s ON t.stock_id = s.id
        WHERE t.stock_id IN ({stock_id_list})
        AND t.date >= CURRENT_DATE - INTERVAL '730 days'
        ORDER BY s.ticker, t.date
    """
    tech_df = pd.read_sql(query_tech, conn)
    rsi = tech_df.pivot(index='date', columns='ticker', values='rsi_14')
    macd = tech_df.pivot(index='date', columns='ticker', values='macd')
    macd_signal = tech_df.pivot(index='date', columns='ticker', values='macd_signal')
    bb_upper = tech_df.pivot(index='date', columns='ticker', values='bb_upper')
    bb_middle = tech_df.pivot(index='date', columns='ticker', values='bb_middle')
    bb_lower = tech_df.pivot(index='date', columns='ticker', values='bb_lower')
    sma_20 = tech_df.pivot(index='date', columns='ticker', values='sma_20')
    sma_50 = tech_df.pivot(index='date', columns='ticker', values='sma_50')
    volatility = tech_df.pivot(index='date', columns='ticker', values='volatility_20d')
    
    # 수급 데이터
    query_supply = f"""
        SELECT s.ticker, sd.date, sd.foreign_net_buy, sd.institution_net_buy,
               sd.foreign_ownership, sd.short_ratio
        FROM supply_demand_data sd
        JOIN stocks s ON sd.stock_id = s.id
        WHERE sd.stock_id IN ({stock_id_list})
        AND sd.date >= CURRENT_DATE - INTERVAL '730 days'
        ORDER BY s.ticker, sd.date
    """
    supply_df = pd.read_sql(query_supply, conn)
    foreign_net = supply_df.pivot(index='date', columns='ticker', values='foreign_net_buy')
    institution_net = supply_df.pivot(index='date', columns='ticker', values='institution_net_buy')
    foreign_own = supply_df.pivot(index='date', columns='ticker', values='foreign_ownership')
    short_ratio = supply_df.pivot(index='date', columns='ticker', values='short_ratio')
    
    conn.close()
    
    # 15일 forward return 계산
    returns = close.pct_change()
    forward_return_15d = close.shift(-15) / close - 1
    
    print(f"✅ {len(close.columns)}개 종목, {len(close)}일 데이터")
    print(f"   15일 forward return 범위: {forward_return_15d.min().min():.2%} ~ {forward_return_15d.max().max():.2%}")
    
    return {
        'close': close, 'open': open_px, 'high': high, 'low': low, 'volume': volume,
        'returns': returns, 'forward_return_15d': forward_return_15d,
        'rsi': rsi, 'macd': macd, 'macd_signal': macd_signal,
        'bb_upper': bb_upper, 'bb_middle': bb_middle, 'bb_lower': bb_lower,
        'sma_20': sma_20, 'sma_50': sma_50, 'volatility': volatility,
        'foreign_net': foreign_net, 'institution_net': institution_net,
        'foreign_own': foreign_own, 'short_ratio': short_ratio
    }

def calculate_ic(alpha, forward_ret):
    """Information Coefficient 계산"""
    alpha_flat = alpha.values.flatten()
    ret_flat = forward_ret.values.flatten()
    
    # NaN 제거
    mask = ~(np.isnan(alpha_flat) | np.isnan(ret_flat) | np.isinf(alpha_flat) | np.isinf(ret_flat))
    alpha_clean = alpha_flat[mask]
    ret_clean = ret_flat[mask]
    
    if len(alpha_clean) < 30:
        return 0
    
    # Spearman correlation
    from scipy.stats import spearmanr
    corr, _ = spearmanr(alpha_clean, ret_clean)
    return corr if not np.isnan(corr) else 0

def generate_biweekly_alphas(num_ideas=20):
    """15일 forward용 알파 생성"""
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    prompt = """Generate alpha factors for 15-day forward prediction (bi-weekly rebalancing).

Available features:
- Price: close, open, high, low, volume, returns
- Technical: rsi, macd, macd_signal, bb_upper, bb_middle, bb_lower, sma_20, sma_50, volatility
- Supply/Demand: foreign_net, institution_net, foreign_own, short_ratio

Operators:
- AlphaOperators.ts_rank(x, window): time-series rank 0~1
- AlphaOperators.ts_mean(x, window): moving average
- AlphaOperators.ts_std(x, window): moving std
- AlphaOperators.ts_delta(x, period): current - N days ago
- AlphaOperators.zscore_scale(x): z-score normalization
- AlphaOperators.normed_rank(x): cross-sectional rank 0~1

Strategy focus:
- 15-day holding period (bi-weekly rebalancing)
- Medium-term momentum and trend-following
- Supply-demand shifts
- Transaction cost ~0.3% per trade

Generate 20 diverse alpha expressions focusing on:
1. 15-day momentum with confirmation
2. Trend-following with supply/demand
3. MACD crossovers with volume
4. Foreign/institution buying patterns
5. Volatility-adjusted returns

Output ONLY the Python expressions, one per line, no explanations."""

    response = client.chat.completions.create(
        model='gpt-4',
        messages=[
            {'role': 'system', 'content': 'You are a WorldQuant-level quant researcher.'},
            {'role': 'user', 'content': prompt}
        ],
        temperature=0.8,
        max_tokens=2000
    )
    
    alphas_text = response.choices[0].message.content
    alphas = [line.strip() for line in alphas_text.strip().split('\n') if line.strip() and 'AlphaOperators' in line]
    
    return alphas[:num_ideas]

def main():
    print("=" * 80)
    print("Alpha-GPT: Bi-weekly Rebalancing (15-day forward)")
    print("=" * 80)
    print()
    
    # 데이터 로드
    data = load_biweekly_data()
    
    # globals에 데이터 추가 (eval 시 필요)
    globals().update(data)
    
    # 알파 생성
    print("\n🤖 LLM으로 15일 forward 알파 생성 중...")
    seed_alphas = generate_biweekly_alphas(num_ideas=20)
    print(f"✅ {len(seed_alphas)}개 알파 생성 완료")
    
    # IC 평가
    print("\n📊 IC 평가 중...")
    results = []
    
    for i, alpha_expr in enumerate(seed_alphas, 1):
        print(f"\n[{i}/{len(seed_alphas)}] Testing...")
        print(f"  {alpha_expr[:80]}...")
        
        try:
            alpha = eval(alpha_expr)
            ic = calculate_ic(alpha, data['forward_return_15d'])
            
            results.append({
                'alpha': alpha_expr,
                'ic': ic
            })
            
            print(f"  IC: {ic:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'alpha': alpha_expr,
                'ic': 0
            })
    
    # 결과 정렬
    df_results = pd.DataFrame(results).sort_values('ic', ascending=False)
    
    # 상위 10개 출력
    print("\n" + "=" * 80)
    print("📈 Top 10 Alphas (15-day forward)")
    print("=" * 80)
    
    for i, row in df_results.head(10).iterrows():
        print(f"\n{i+1}. IC: {row['ic']:.4f}")
        print(f"   {row['alpha']}")
    
    # 베스트 알파
    best = df_results.iloc[0]
    
    print("\n" + "=" * 80)
    print("🏆 BEST ALPHA (15-day forward)")
    print("=" * 80)
    print(f"IC: {best['ic']:.4f}")
    print(f"Expression: {best['alpha']}")
    
    # 거래비용 분석
    print("\n" + "=" * 80)
    print("💰 Transaction Cost Analysis")
    print("=" * 80)
    print(f"Rebalancing frequency: Every 15 days")
    print(f"Rebalances per year: ~24")
    print(f"Transaction cost per trade: 0.3%")
    print(f"Total annual cost: ~14.4% (0.3% × 24 × 2)")
    print(f"\nNet IC after costs: ~{best['ic'] - 0.02:.4f}")
    
    # DB 저장
    save = input("\n💾 Save to database? (y/n): ")
    
    if save.lower() == 'y':
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alpha_formulas (formula, ic_score, description, created_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (formula) DO UPDATE
            SET ic_score = EXCLUDED.ic_score, updated_at = NOW()
        """, (
            best['alpha'],
            float(best['ic']),
            '15-day forward alpha (bi-weekly rebalancing)'
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Saved!")
    
    print("\n🎉 완료!")

if __name__ == "__main__":
    main()
