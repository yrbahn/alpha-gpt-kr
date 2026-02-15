#!/usr/bin/env python3
"""
Alpha-GPT: 15-day Forward with Gemini 2.5 Pro
Gemini 2.5 Pro로 Seed 20개 생성
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2
import google.generativeai as genai
import random
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

def load_market_data():
    """500개 종목 데이터 로드"""
    print("📊 데이터 로드 중... (500종목, 2년)")
    
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
    
    close = price_df.pivot(index='date', columns='ticker', values='close')
    volume = price_df.pivot(index='date', columns='ticker', values='volume')
    
    print(f"✅ {len(close.columns)}개 종목, {len(close)}일 데이터")
    
    return {
        'close': close,
        'volume': volume,
        'returns': close.pct_change()
    }

def generate_seed_alphas_gemini():
    """Gemini 2.5 Pro로 초기 알파 생성"""
    print("\n🤖 Gemini 2.5 Pro로 초기 알파 20개 생성 중...")
    
    # Gemini API 설정
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY 또는 GEMINI_API_KEY 환경변수 필요!")
        # Fallback to defaults
        return [
            "AlphaOperators.ts_rank(AlphaOperators.ts_delta(close, 15), 10)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_mean(returns, 15), 10)",
            "AlphaOperators.ts_rank(close / AlphaOperators.ts_mean(close, 20), 15)",
            "AlphaOperators.ts_rank(volume / AlphaOperators.ts_mean(volume, 15), 10)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_delta(close, 15) / AlphaOperators.ts_std(close, 20), 10)"
        ]
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    prompt = """You are a quantitative researcher. Generate 20 diverse alpha factor expressions for predicting 15-day forward stock returns.

Available data:
- close: stock closing price
- volume: trading volume  
- returns: daily returns (close.pct_change())

Available operators (use EXACTLY as shown):
- AlphaOperators.ts_delta(x, period): current value - value N days ago
- AlphaOperators.ts_mean(x, window): N-day moving average
- AlphaOperators.ts_std(x, window): N-day standard deviation
- AlphaOperators.ts_rank(x, window): N-day rank normalized to 0-1

Strategy focus:
- 15-day holding period (bi-weekly rebalancing)
- Combine momentum, trend, and volume patterns
- Use different time windows (5, 10, 15, 20 days)

Generate EXACTLY 20 diverse expressions. Output format:
Each line should be a valid Python expression using AlphaOperators.
NO explanations, NO numbering, NO extra text.
Just the expressions, one per line.

Example format:
AlphaOperators.ts_rank(AlphaOperators.ts_delta(close, 15), 10)
AlphaOperators.ts_rank(volume / AlphaOperators.ts_mean(volume, 20), 10)
"""
    
    response = model.generate_content(prompt)
    content = response.text
    
    # 파싱 개선
    alphas = []
    for line in content.split('\n'):
        line = line.strip()
        
        # 불필요한 부분 제거
        if '```' in line:
            continue
        if line.startswith('#'):
            continue
        if not line:
            continue
            
        # 번호 제거 (1., 1), [1] 등)
        import re
        line = re.sub(r'^\d+[\.\)\]]\s*', '', line)
        
        # AlphaOperators가 포함된 것만
        if 'AlphaOperators' in line:
            # 주석 제거
            if '#' in line:
                line = line.split('#')[0].strip()
            alphas.append(line)
    
    # Fallback
    if len(alphas) < 10:
        print(f"⚠️  Gemini가 {len(alphas)}개만 생성, 기본 알파 추가")
        fallback = [
            "AlphaOperators.ts_rank(AlphaOperators.ts_delta(close, 15), 10)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_mean(returns, 15), 10)",
            "AlphaOperators.ts_rank(close / AlphaOperators.ts_mean(close, 20), 15)",
            "AlphaOperators.ts_rank(volume / AlphaOperators.ts_mean(volume, 15), 10)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_delta(close, 15) / AlphaOperators.ts_std(close, 20), 10)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_delta(volume, 10), 15)",
            "AlphaOperators.ts_rank(returns / AlphaOperators.ts_std(returns, 20), 10)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_mean(close, 10) / AlphaOperators.ts_mean(close, 20), 15)",
            "AlphaOperators.ts_rank(volume * AlphaOperators.ts_delta(close, 5), 10)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_std(returns, 15), 20)"
        ]
        alphas = alphas + [f for f in fallback if f not in alphas]
    
    print(f"✅ {len(alphas)}개 초기 알파 생성")
    return alphas[:20]

# 전역 데이터
_global_data = None

def set_global_data(data):
    global _global_data
    _global_data = data

def evaluate_alpha_worker(alpha_expr):
    """병렬 처리용 알파 평가 (15일 forward)"""
    global _global_data
    data = _global_data
    
    try:
        close = data['close']
        volume = data['volume']
        returns = data['returns']
        
        # 15일 forward return
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

def genetic_programming(seed_alphas, data, generations=30, population_size=100):
    """간단한 병렬 GP"""
    
    print(f"\n🧬 병렬 GP 시작")
    print(f"   Seed: {len(seed_alphas)}개, 세대: {generations}, 개체수: {population_size}, 워커: 4")
    
    # 초기 개체군
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
        
        # 병렬 평가
        with Pool(4, initializer=set_global_data, initargs=(data,)) as pool:
            results = pool.map(evaluate_alpha_worker, population)
        
        fitness_scores = sorted(results, key=lambda x: x[1], reverse=True)
        
        best_ic = fitness_scores[0][1]
        print(f"    최고 IC: {best_ic:.4f}")
        
        if best_ic > best_ever[1]:
            best_ever = fitness_scores[0]
            print(f"    🏆 신기록!")
        
        # 다음 세대
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
    
    return best_ever

def main():
    print("=" * 80)
    print("Alpha-GPT: 15-day Forward with Gemini 2.5 Pro")
    print("=" * 80)
    print()
    
    # 데이터 로드
    data = load_market_data()
    
    # Gemini로 Seed 알파 생성
    seed_alphas = generate_seed_alphas_gemini()
    
    # GP 실행
    best_alpha, best_ic = genetic_programming(
        seed_alphas, 
        data, 
        generations=30, 
        population_size=100
    )
    
    # 결과
    print("\n" + "=" * 80)
    print("🏆 BEST ALPHA (15-day forward, Gemini 2.5 Pro)")
    print("=" * 80)
    print(f"IC: {best_ic:.4f}")
    print(f"Expression: {best_alpha}")
    print()
    
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
            best_alpha,
            float(best_ic),
            '15-day forward alpha (500 stocks, Gemini 2.5 Pro, worker=4)'
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Saved!")
    
    print("\n🎉 완료!")

if __name__ == "__main__":
    main()
