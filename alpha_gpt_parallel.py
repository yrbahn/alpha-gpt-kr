#!/usr/bin/env python3
"""
Alpha-GPT 병렬 처리 버전
multiprocessing으로 개체 평가 병렬화 → population 대폭 증가 가능
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
from functools import partial

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

# 데이터 로드
def load_market_data():
    """시가총액 상위 100개 종목 (2년 데이터)"""
    print("📊 데이터 로드 중... (2년)")
    
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
        LIMIT 100
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

# LLM으로 초기 알파 생성
def generate_seed_alphas_with_llm(num_seeds=10):
    """LLM으로 초기 알파 생성"""
    
    print(f"\n🤖 LLM이 초기 {num_seeds}개 알파 생성 중...")
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    prompt = f"""당신은 퀀트 개발자입니다. 한국 증시에서 강한 모멘텀과 낮은 변동성을 가진 종목을 찾는 알파 표현식을 생성하세요.

사용 가능한 데이터:
- close: 종가
- volume: 거래량
- returns: 수익률

사용 가능한 연산자:
- ts_delta(x, period), ts_mean(x, window), ts_std(x, window), ts_rank(x, window)

{num_seeds}개의 다양한 알파를 생성하세요. 각각 한 줄로:

ALPHA_1: [표현식]
ALPHA_2: [표현식]
...
"""

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a quantitative researcher."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9,
        max_tokens=1000
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
    
    if len(alphas) == 0:
        alphas = [
            "AlphaOperators.ts_rank(AlphaOperators.ts_delta(close, 20), 10)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_std(returns, 10) / AlphaOperators.ts_std(returns, 20), 10)",
            "AlphaOperators.ts_rank(close / AlphaOperators.ts_mean(close, 20), 10)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_delta(close, 5) / AlphaOperators.ts_std(close, 20), 15)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_mean(returns, 5), 10)"
        ]
    
    print(f"✅ {len(alphas)}개 초기 알파 생성")
    return alphas

# 알파 평가 (병렬 처리용 - global data 사용)
_global_data = None

def set_global_data(data):
    """전역 데이터 설정 (multiprocessing용)"""
    global _global_data
    _global_data = data

def evaluate_alpha_ic_worker(alpha_expr):
    """병렬 처리용 알파 평가 함수"""
    global _global_data
    data = _global_data
    
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
            
            if valid.sum() > 10:
                ic = alpha_cs[valid].corr(returns_cs[valid])
                if not np.isnan(ic):
                    ic_list.append(ic)
        
        if len(ic_list) < 10:
            return (alpha_expr, -999.0)
        
        return (alpha_expr, np.mean(ic_list))
        
    except:
        return (alpha_expr, -999.0)

# 병렬 GP 진화
def genetic_programming_parallel(seed_alphas, data, generations=10, population_size=100, num_workers=None):
    """병렬 처리 Genetic Programming"""
    
    if num_workers is None:
        num_workers = min(cpu_count(), 8)  # 최대 8개 코어
    
    print(f"\n🧬 병렬 GP 진화 시작")
    print(f"   세대: {generations}, 개체수: {population_size}, 워커: {num_workers}")
    
    # 초기 개체군
    population = seed_alphas[:population_size]
    while len(population) < population_size:
        parent = random.choice(seed_alphas)
        mutated = mutate_alpha(parent)
        if mutated:
            population.append(mutated)
    
    # 전역 데이터 설정
    set_global_data(data)
    
    for gen in range(generations):
        print(f"\n  세대 {gen+1}/{generations}")
        
        # 🚀 병렬 평가
        with Pool(num_workers, initializer=set_global_data, initargs=(data,)) as pool:
            results = pool.map(evaluate_alpha_ic_worker, population)
        
        # 정렬
        fitness_scores = sorted(results, key=lambda x: x[1], reverse=True)
        
        best_ic = fitness_scores[0][1]
        print(f"    최고 IC: {best_ic:.4f} (병렬 처리 완료)")
        
        # 다음 세대 생성
        next_population = []
        
        # 엘리트 보존
        elite_count = population_size // 5
        for alpha, _ in fitness_scores[:elite_count]:
            next_population.append(alpha)
        
        # 교차 + 변이
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
    
    # 최종 평가
    with Pool(num_workers, initializer=set_global_data, initargs=(data,)) as pool:
        final_results = pool.map(evaluate_alpha_ic_worker, population)
    
    final_fitness = sorted(final_results, key=lambda x: x[1], reverse=True)
    
    print(f"\n✅ 병렬 GP 진화 완료!")
    print(f"   최종 최고 IC: {final_fitness[0][1]:.4f}")
    
    return final_fitness

# GP 연산자들
def tournament_select(fitness_scores, k=3):
    candidates = random.sample(fitness_scores, min(k, len(fitness_scores)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

def crossover_alphas(alpha1, alpha2):
    try:
        tokens1 = alpha1.split('(')
        tokens2 = alpha2.split('(')
        
        if len(tokens1) > 2 and len(tokens2) > 2:
            point = random.randint(1, min(len(tokens1), len(tokens2)) - 1)
            child_tokens = tokens1[:point] + tokens2[point:]
            return '('.join(child_tokens)
        
        return None
    except:
        return None

def mutate_alpha(alpha):
    try:
        import re
        numbers = re.findall(r'\d+', alpha)
        
        if numbers:
            old_num = random.choice(numbers)
            new_num = str(int(old_num) + random.randint(-5, 5))
            if int(new_num) > 0:
                return alpha.replace(old_num, new_num, 1)
        
        return None
    except:
        return None

# 메인
def main():
    print("=" * 70)
    print("Alpha-GPT 병렬 처리 버전 (Population 대폭 증가)")
    print("=" * 70)
    print()
    
    # CPU 정보
    num_cpus = cpu_count()
    print(f"💻 사용 가능 CPU: {num_cpus}개")
    print(f"   병렬 처리 워커: {min(num_cpus, 8)}개")
    print()
    
    # 데이터 로드
    data = load_market_data()
    
    # LLM seed 생성
    seed_alphas = generate_seed_alphas_with_llm(num_seeds=10)
    
    # 병렬 GP 진화
    evolved_alphas = genetic_programming_parallel(
        seed_alphas=seed_alphas,
        data=data,
        generations=10,
        population_size=100,  # 🚀 20 → 100으로 증가!
        num_workers=None  # 자동 선택
    )
    
    # 결과 출력
    print("\n" + "=" * 70)
    print("🏆 진화된 상위 5개 알파")
    print("=" * 70)
    
    for i, (alpha, ic) in enumerate(evolved_alphas[:5], 1):
        print(f"\n{i}. IC: {ic:.4f}")
        print(f"   {alpha}")
    
    # DB 저장
    best_alpha, best_ic = evolved_alphas[0]
    
    print(f"\n💾 최상위 알파 DB 저장...")
    print(f"   IC: {best_ic:.4f}")
    
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
            best_alpha,
            date.today(),
            True,
            float(best_ic * 10),
            f"IC: {best_ic:.4f}, Parallel GP (pop=100, 2year data)"
        ))
        conn.commit()
        print("✅ DB 저장 완료")
    finally:
        cur.close()
        conn.close()
    
    print("\n🎉 병렬 Alpha-GPT 완료!")

if __name__ == "__main__":
    main()
