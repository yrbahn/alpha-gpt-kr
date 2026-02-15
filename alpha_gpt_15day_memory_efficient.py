#!/usr/bin/env python3
"""
Alpha-GPT: 15-day Forward + LLM + GP (Memory Efficient)
메모리 효율적 버전 - 2000개 종목 처리 가능
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
from multiprocessing import Pool, cpu_count

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

def load_market_data_efficient():
    """메모리 효율적 데이터 로드 - 2000개 종목"""
    print("📊 메모리 효율적 데이터 로드 중... (2000종목, 2년)")
    
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
        LIMIT 2000
    """
    
    stocks_df = pd.read_sql(query_stocks, conn)
    stock_ids = stocks_df['id'].tolist()
    
    stock_id_list = ', '.join(map(str, stock_ids))
    
    # ⚡ 메모리 최적화 1: float64 대신 float32 사용
    query_prices = f"""
        SELECT 
            s.ticker,
            p.date,
            p.close::float AS close,
            p.volume::float AS volume
        FROM price_data p
        JOIN stocks s ON p.stock_id = s.id
        WHERE p.stock_id IN ({stock_id_list})
        AND p.date >= CURRENT_DATE - INTERVAL '730 days'
        ORDER BY s.ticker, p.date
    """
    
    price_df = pd.read_sql(query_prices, conn)
    conn.close()
    
    # ⚡ 메모리 최적화 2: 타입 변환
    close_pivot = price_df.pivot(index='date', columns='ticker', values='close').astype('float32')
    volume_pivot = price_df.pivot(index='date', columns='ticker', values='volume').astype('float32')
    
    # 메모리 해제
    del price_df
    
    # 15일 forward return
    returns_15d = (close_pivot.shift(-15) / close_pivot - 1).astype('float32')
    
    print(f"✅ {len(close_pivot.columns)}개 종목, {len(close_pivot)}일 데이터")
    print(f"   메모리 사용: ~{(close_pivot.memory_usage().sum() + volume_pivot.memory_usage().sum() + returns_15d.memory_usage().sum()) / 1024**2:.1f} MB")
    
    return {
        'close': close_pivot,
        'volume': volume_pivot,
        'returns': close_pivot.pct_change().astype('float32'),
        'forward_return_15d': returns_15d
    }

def generate_seed_alphas_with_llm(num_seeds=20):
    """LLM으로 초기 알파 생성"""
    print(f"\n🤖 LLM이 초기 {num_seeds}개 알파 생성 중...")
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    prompt = f"""Generate {num_seeds} alpha factors for 15-day forward prediction.

Available: close, volume, returns
Operators: AlphaOperators.ts_delta(x, period), ts_mean(x, window), ts_std(x, window), ts_rank(x, window), zscore_scale(x), normed_rank(x)

Focus on 15-day momentum, trend-following, and volume patterns.
Output ONLY Python expressions, one per line, no explanations."""

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a quantitative researcher."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9,
        max_tokens=1500
    )
    
    content = response.choices[0].message.content
    alphas = []
    for line in content.split('\n'):
        line = line.strip()
        if 'AlphaOperators' in line:
            if ':' in line:
                line = line.split(':', 1)[1].strip()
            if '#' in line:
                line = line.split('#')[0].strip()
            if line:
                alphas.append(line)
    
    if len(alphas) < 5:
        alphas = [
            "AlphaOperators.ts_rank(AlphaOperators.ts_delta(close, 15), 10)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_mean(returns, 15), 10)",
            "AlphaOperators.ts_rank(close / AlphaOperators.ts_mean(close, 20), 15)",
            "AlphaOperators.ts_rank(volume / AlphaOperators.ts_mean(volume, 15), 10)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_delta(close, 15) / AlphaOperators.ts_std(close, 20), 10)"
        ] + alphas
    
    print(f"✅ {len(alphas)}개 초기 알파 생성")
    return alphas[:num_seeds]

# ⚡ 메모리 최적화 3: 샘플링 평가
def evaluate_alpha_sampled(alpha_expr, data, sample_ratio=0.5):
    """메모리 절감: 전체 데이터 중 일부만 샘플링하여 평가"""
    try:
        close = data['close']
        volume = data['volume']
        returns = data['returns']
        forward_return_15d = data['forward_return_15d']
        
        # ⚡ 랜덤 샘플링: 50% 종목만 사용
        n_stocks = int(len(close.columns) * sample_ratio)
        sampled_tickers = random.sample(list(close.columns), n_stocks)
        
        close_sample = close[sampled_tickers]
        volume_sample = volume[sampled_tickers]
        returns_sample = returns[sampled_tickers]
        forward_sample = forward_return_15d[sampled_tickers]
        
        # 로컬 변수로 재할당 (eval에서 사용)
        close = close_sample
        volume = volume_sample
        returns = returns_sample
        
        alpha_values = eval(alpha_expr)
        
        ic_list = []
        for date in alpha_values.index[:-15]:
            alpha_cs = alpha_values.loc[date]
            returns_cs = forward_sample.loc[date]
            valid = alpha_cs.notna() & returns_cs.notna()
            
            if valid.sum() > 30:
                ic = alpha_cs[valid].corr(returns_cs[valid])
                if not np.isnan(ic):
                    ic_list.append(ic)
        
        if len(ic_list) < 10:
            return -999.0
        
        return np.mean(ic_list)
        
    except:
        return -999.0

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

def tournament_select(fitness_scores, k=3):
    """토너먼트 선택"""
    tournament = random.sample(fitness_scores, min(k, len(fitness_scores)))
    return max(tournament, key=lambda x: x[1])[0]

def genetic_programming_efficient(seed_alphas, data, generations=30, population_size=100):
    """메모리 효율적 GP - Sequential 처리"""
    
    print(f"\n🧬 메모리 효율적 GP 진화 시작")
    print(f"   세대: {generations}, 개체수: {population_size}")
    print(f"   샘플링: 각 평가마다 50% 종목 랜덤 선택")
    
    # 초기 개체군
    population = seed_alphas[:population_size]
    while len(population) < population_size:
        parent = random.choice(seed_alphas)
        mutated = mutate_alpha(parent)
        if mutated:
            population.append(mutated)
    
    best_ever = (None, -999.0)
    
    for gen in range(generations):
        print(f"\n  세대 {gen+1}/{generations}")
        
        # ⚡ Sequential 평가 (메모리 안전)
        fitness_scores = []
        for i, alpha in enumerate(population):
            if i % 20 == 0:
                print(f"    평가 진행: {i}/{len(population)}", end='\r')
            ic = evaluate_alpha_sampled(alpha, data, sample_ratio=0.5)
            fitness_scores.append((alpha, ic))
        
        fitness_scores = sorted(fitness_scores, key=lambda x: x[1], reverse=True)
        
        best_ic = fitness_scores[0][1]
        print(f"    최고 IC: {best_ic:.4f}" + " " * 30)
        
        if best_ic > best_ever[1]:
            best_ever = fitness_scores[0]
            print(f"    🏆 신기록! IC: {best_ic:.4f}")
        
        # 다음 세대
        next_population = []
        elite_count = population_size // 5
        for alpha, _ in fitness_scores[:elite_count]:
            next_population.append(alpha)
        
        while len(next_population) < population_size:
            if random.random() < 0.7:
                parent1 = tournament_select(fitness_scores)
                parent2 = tournament_select(fitness_scores)
                child = crossover_alphas(parent1, parent2)
                if child:
                    next_population.append(child)
                else:
                    next_population.append(parent1)
            else:
                parent = tournament_select(fitness_scores)
                mutated = mutate_alpha(parent)
                if mutated:
                    next_population.append(mutated)
                else:
                    next_population.append(parent)
        
        population = next_population[:population_size]
    
    return best_ever

def main():
    print("=" * 80)
    print("Alpha-GPT: 15-day Forward (Memory Efficient)")
    print("=" * 80)
    print()
    
    # 메모리 효율적 데이터 로드
    data = load_market_data_efficient()
    
    # LLM seed 생성
    seed_alphas = generate_seed_alphas_with_llm(num_seeds=20)
    
    # 메모리 효율적 GP
    best_alpha, best_ic = genetic_programming_efficient(
        seed_alphas, 
        data, 
        generations=30, 
        population_size=100
    )
    
    # 최종 평가 (전체 데이터)
    print("\n🔍 최종 알파를 전체 데이터로 재평가 중...")
    final_ic = evaluate_alpha_sampled(best_alpha, data, sample_ratio=1.0)
    
    # 결과
    print("\n" + "=" * 80)
    print("🏆 BEST ALPHA (15-day forward, Memory Efficient)")
    print("=" * 80)
    print(f"샘플 IC: {best_ic:.4f}")
    print(f"전체 IC: {final_ic:.4f}")
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
            float(final_ic),
            '15-day forward alpha (2000 stocks, memory efficient)'
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Saved!")
    
    print("\n🎉 완료!")

if __name__ == "__main__":
    main()
