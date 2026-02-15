#!/usr/bin/env python3
"""
Alpha-GPT: 15-day Forward with Comprehensive Indicators
가격 + 기술적 지표 + 수급 지표 종합
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2
import openai
import random
import gc
from multiprocessing import Pool

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

def load_comprehensive_data():
    """500개 종목 종합 데이터 로드 (가격 + 기술적 + 수급)"""
    print("📊 종합 데이터 로드 중... (시총 상위 500종목, 2년)")
    
    conn = get_db_connection()
    
    # 시가총액 상위 500개
    query_stocks = """
        SELECT s.id, s.ticker, s.name
        FROM stocks s
        WHERE s.is_active = true
        AND s.market_cap IS NOT NULL
        ORDER BY s.market_cap DESC
        LIMIT 500
    """
    
    stocks_df = pd.read_sql(query_stocks, conn)
    stock_ids = stocks_df['id'].tolist()
    stock_id_list = ', '.join(map(str, stock_ids))
    
    # 1. 가격 데이터
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
    
    # 2. 기술적 지표
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
    
    # 3. 수급 데이터
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
    
    returns = close.pct_change()
    
    print(f"✅ {len(close.columns)}개 종목, {len(close)}일 데이터")
    print(f"   가격: close, open, high, low, volume, returns")
    print(f"   기술적: rsi, macd, bb, sma, volatility")
    print(f"   수급: foreign_net, institution_net, ownership, short_ratio")
    
    return {
        # 가격
        'close': close, 'open': open_px, 'high': high, 'low': low, 
        'volume': volume, 'returns': returns,
        # 기술적 지표
        'rsi': rsi, 'macd': macd, 'macd_signal': macd_signal,
        'bb_upper': bb_upper, 'bb_middle': bb_middle, 'bb_lower': bb_lower,
        'sma_20': sma_20, 'sma_50': sma_50, 'volatility': volatility,
        # 수급
        'foreign_net': foreign_net, 'institution_net': institution_net,
        'foreign_own': foreign_own, 'short_ratio': short_ratio
    }

def generate_comprehensive_alphas():
    """종합 지표로 알파 생성"""
    print("\n🤖 GPT-4o로 종합 알파 20개 생성 중...")
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    prompt = """Generate EXACTLY 20 diverse alpha expressions for 15-day forward prediction using comprehensive indicators.

**Available data (27 indicators):**

Price & Volume:
- close, open, high, low, volume, returns

Technical Indicators:
- rsi: RSI(14) indicator
- macd, macd_signal: MACD line and signal
- bb_upper, bb_middle, bb_lower: Bollinger Bands
- sma_20, sma_50: Moving averages
- volatility: 20-day volatility

Supply & Demand (Korean market):
- foreign_net: Foreign net buying
- institution_net: Institution net buying
- foreign_own: Foreign ownership ratio
- short_ratio: Short selling ratio

**Operators:**
- AlphaOperators.ts_delta(x, period)
- AlphaOperators.ts_mean(x, window)
- AlphaOperators.ts_std(x, window)
- AlphaOperators.ts_rank(x, window)

**Strategy ideas:**
1. RSI oversold + foreign buying
2. MACD golden cross + volume surge
3. Bollinger lower touch + institution buying
4. Price above SMA + low volatility
5. Foreign ownership increase + price momentum
6. Short ratio decrease + price rise

Generate 20 DIVERSE expressions combining:
- Technical + Supply/Demand
- Price + Volume patterns
- Multiple timeframes (5, 10, 15, 20 days)

Output ONLY Python expressions, one per line. NO explanations, NO numbering."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a quantitative researcher. Output only Python code expressions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9,
        max_tokens=2000
    )
    
    content = response.choices[0].message.content
    alphas = []
    for line in content.split('\n'):
        line = line.strip()
        if not line or '```' in line or line.startswith('#'):
            continue
        import re
        line = re.sub(r'^\d+[\.\)\]:\-\s]+', '', line)
        if 'AlphaOperators' in line:
            if '#' in line:
                line = line.split('#')[0].strip()
            line = line.strip('"\'')
            alphas.append(line)
    
    if len(alphas) < 10:
        print(f"⚠️  Only {len(alphas)} parsed, adding fallback")
        fallback = [
            "AlphaOperators.ts_rank(rsi, 10)",
            "AlphaOperators.ts_rank(macd - macd_signal, 15)",
            "AlphaOperators.ts_rank((close - bb_lower) / (bb_upper - bb_lower), 10)",
            "AlphaOperators.ts_rank(foreign_net + institution_net, 10)",
            "AlphaOperators.ts_rank(close / sma_20, 15)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_delta(foreign_own, 20), 10)",
            "AlphaOperators.ts_rank(returns / volatility, 10)",
            "AlphaOperators.ts_rank(volume * AlphaOperators.ts_delta(close, 5), 10)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_mean(foreign_net, 10), 15)",
            "AlphaOperators.ts_rank(sma_20 / sma_50, 10)"
        ]
        alphas = alphas + [f for f in fallback if f not in alphas]
    
    print(f"✅ {len(alphas)}개 종합 알파 생성")
    return alphas[:20]

# 전역 데이터
_global_data = None

def set_global_data(data):
    global _global_data
    _global_data = data

def evaluate_alpha_worker(alpha_expr):
    """병렬 처리용 알파 평가"""
    global _global_data
    data = _global_data
    
    try:
        # 모든 변수를 로컬에 추가
        close = data['close']
        open = data['open']
        high = data['high']
        low = data['low']
        volume = data['volume']
        returns = data['returns']
        rsi = data['rsi']
        macd = data['macd']
        macd_signal = data['macd_signal']
        bb_upper = data['bb_upper']
        bb_middle = data['bb_middle']
        bb_lower = data['bb_lower']
        sma_20 = data['sma_20']
        sma_50 = data['sma_50']
        volatility = data['volatility']
        foreign_net = data['foreign_net']
        institution_net = data['institution_net']
        foreign_own = data['foreign_own']
        short_ratio = data['short_ratio']
        
        forward_return_15d = close.shift(-15) / close - 1
        
        alpha_values = eval(alpha_expr)
        
        ic_list = []
        for date in alpha_values.index[:-15]:
            alpha_cs = alpha_values.loc[date]
            returns_cs = forward_return_15d.loc[date]
            valid = alpha_cs.notna() & returns_cs.notna()
            
            if valid.sum() > 30:
                ic = alpha_cs[valid].corr(returns_cs[valid])
                if not np.isnan(ic):
                    ic_list.append(ic)
        
        if len(ic_list) < 10:
            return (alpha_expr, -999.0)
        
        return (alpha_expr, np.mean(ic_list))
        
    except:
        return (alpha_expr, -999.0)

def mutate_alpha(alpha_expr):
    """알파 변이"""
    try:
        operators = ['ts_delta', 'ts_mean', 'ts_std', 'ts_rank']
        for op in operators:
            if op in alpha_expr:
                import re
                match = re.search(rf'{op}\([^,]+,\s*(\d+)\)', alpha_expr)
                if match:
                    old_window = int(match.group(1))
                    new_window = max(5, old_window + random.choice([-5, -2, 2, 5]))
                    return alpha_expr.replace(f', {old_window})', f', {new_window})')
        return None
    except:
        return None

def crossover_alphas(alpha1, alpha2):
    """알파 교차"""
    try:
        operators = ['ts_delta', 'ts_mean', 'ts_std', 'ts_rank']
        for op in operators:
            if op in alpha1 and op in alpha2:
                import re
                match1 = re.search(rf'{op}\(([^,]+),\s*(\d+)\)', alpha1)
                match2 = re.search(rf'{op}\(([^,]+),\s*(\d+)\)', alpha2)
                if match1 and match2:
                    var1, win1 = match1.groups()
                    var2, win2 = match2.groups()
                    return alpha1.replace(f'{op}({var1}, {win1})', f'{op}({var1}, {win2})')
        return None
    except:
        return None

def genetic_programming(seed_alphas, data, generations=20, population_size=100):
    """GP 진화"""
    
    print(f"\n🧬 병렬 GP 시작")
    print(f"   Seed: {len(seed_alphas)}개, 세대: {generations}, 개체수: {population_size}, 워커: 4")
    
    population = seed_alphas[:population_size]
    while len(population) < population_size:
        parent = random.choice(seed_alphas)
        mutated = mutate_alpha(parent)
        if mutated:
            population.append(mutated)
    
    set_global_data(data)
    best_ever = (None, -999.0)
    
    for gen in range(1, generations + 1):
        print(f"\n  세대 {gen}/{generations}")
        
        with Pool(4, initializer=set_global_data, initargs=(data,)) as pool:
            results = pool.map(evaluate_alpha_worker, population)
        
        fitness_scores = sorted(results, key=lambda x: x[1], reverse=True)
        
        best_ic = fitness_scores[0][1]
        print(f"    최고 IC: {best_ic:.4f}")
        
        if best_ic > best_ever[1]:
            best_ever = fitness_scores[0]
            print(f"    🏆 신기록!")
        
        next_population = []
        elite_count = population_size // 5
        for alpha, _ in fitness_scores[:elite_count]:
            next_population.append(alpha)
        
        while len(next_population) < population_size:
            if random.random() < 0.7:
                parent1 = random.choice([a for a, ic in fitness_scores[:20]])
                parent2 = random.choice([a for a, ic in fitness_scores[:20]])
                child = crossover_alphas(parent1, parent2)
                if child:
                    next_population.append(child)
                else:
                    next_population.append(parent1)
            else:
                parent = random.choice([a for a, ic in fitness_scores[:20]])
                mutated = mutate_alpha(parent)
                if mutated:
                    next_population.append(mutated)
                else:
                    next_population.append(parent)
        
        population = next_population[:population_size]
        
        del results, fitness_scores, next_population
        gc.collect()
    
    return best_ever

def main():
    print("=" * 80)
    print("Alpha-GPT: 15-day Forward with Comprehensive Indicators")
    print("=" * 80)
    print()
    
    data = load_comprehensive_data()
    seed_alphas = generate_comprehensive_alphas()
    
    best_alpha, best_ic = genetic_programming(
        seed_alphas, 
        data, 
        generations=20,
        population_size=100
    )
    
    print("\n" + "=" * 80)
    print("🏆 BEST ALPHA (15-day forward, Comprehensive)")
    print("=" * 80)
    print(f"IC: {best_ic:.4f}")
    print(f"Expression: {best_alpha}")
    print()
    
    # 파일로 저장
    with open('BEST_ALPHA_15DAY_COMPREHENSIVE.txt', 'w') as f:
        f.write(f"IC: {best_ic:.4f}\n")
        f.write(f"Alpha: {best_alpha}\n")
    
    print("✅ 결과를 BEST_ALPHA_15DAY_COMPREHENSIVE.txt에 저장했습니다!")
    print("\n🎉 완료!")

if __name__ == "__main__":
    main()
