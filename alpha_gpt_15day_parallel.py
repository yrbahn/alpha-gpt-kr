#!/usr/bin/env python3
"""
Alpha-GPT: 15-day Forward + LLM + GP (Parallel)
월 2회 리밸런싱 (논문 표준)
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
    """시가총액 상위 2000개 종목 (2년 데이터)"""
    print("📊 데이터 로드 중... (2000종목, 2년)")
    
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
    
    # 15일 forward return
    returns_15d = close_pivot.shift(-15) / close_pivot - 1
    
    print(f"✅ {len(close_pivot.columns)}개 종목, {len(close_pivot)}일 데이터")
    print(f"   15일 forward return 범위: {returns_15d.min().min():.2%} ~ {returns_15d.max().max():.2%}")
    
    return {
        'close': close_pivot,
        'volume': volume_pivot,
        'returns': close_pivot.pct_change(),
        'forward_return_15d': returns_15d
    }

def generate_seed_alphas_with_llm(num_seeds=20):
    """LLM으로 초기 알파 생성 (15일 전략)"""
    
    print(f"\n🤖 LLM이 초기 {num_seeds}개 알파 생성 중...")
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    prompt = f"""당신은 WorldQuant 수준의 퀀트 개발자입니다. 
한국 증시에서 15일 보유 시 수익이 높을 종목을 찾는 알파를 생성하세요.

**전략 목표:**
- 리밸런싱 주기: 15일 (월 2회)
- 거래비용: ~0.3% per trade
- 목표: 중기 모멘텀 + 트렌드 추종

**사용 가능한 데이터:**
- close: 종가
- volume: 거래량
- returns: 일간 수익률

**사용 가능한 연산자:**
- AlphaOperators.ts_delta(x, period): 현재값 - N일 전 값
- AlphaOperators.ts_mean(x, window): N일 이동평균
- AlphaOperators.ts_std(x, window): N일 이동 표준편차
- AlphaOperators.ts_rank(x, window): N일 기준 순위 0~1
- AlphaOperators.zscore_scale(x): Z-score 정규화
- AlphaOperators.normed_rank(x): 횡단면 순위 0~1

**전략 아이디어:**
1. 15일 모멘텀 + 거래량 확인: ts_rank(ts_delta(close, 15), 10) * ts_rank(volume, 5)
2. 안정적 상승: ts_rank(close, 15) / ts_std(returns, 15)
3. 가격 대비 거래량: ts_rank(volume / ts_mean(volume, 20), 10)
4. 변동성 조정 수익률: ts_mean(returns, 15) / ts_std(returns, 15)
5. 15일 추세 강도: ts_rank(ts_delta(close, 15) / ts_std(close, 15), 10)

{num_seeds}개의 다양한 알파를 생성하세요. 
출력 형식: Python 표현식만, 한 줄에 하나씩, 설명 없이."""

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a quantitative researcher at WorldQuant."},
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
    
    if len(alphas) == 0:
        alphas = [
            "AlphaOperators.ts_rank(AlphaOperators.ts_delta(close, 15), 10)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_mean(returns, 15), 10)",
            "AlphaOperators.ts_rank(close / AlphaOperators.ts_mean(close, 20), 15)",
            "AlphaOperators.ts_rank(volume / AlphaOperators.ts_mean(volume, 15), 10)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_delta(close, 15) / AlphaOperators.ts_std(close, 20), 10)"
        ]
    
    print(f"✅ {len(alphas)}개 초기 알파 생성")
    return alphas

# 전역 데이터 (병렬 처리용)
_global_data = None

def set_global_data(data):
    global _global_data
    _global_data = data

def evaluate_alpha_ic_worker(alpha_expr):
    """병렬 처리용 알파 평가 (15일 forward)"""
    global _global_data
    data = _global_data
    
    try:
        close = data['close']
        volume = data['volume']
        returns = data['returns']
        forward_return_15d = data['forward_return_15d']
        
        alpha_values = eval(alpha_expr)
        
        ic_list = []
        for date in alpha_values.index[:-15]:  # 15일 forward이므로
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
                old_window = None
                import re
                match = re.search(rf'{op}\([^,]+,\s*(\d+)\)', alpha_expr)
                if match:
                    old_window = int(match.group(1))
                    new_window = max(5, old_window + random.choice([-5, -2, 2, 5]))
                    new_alpha = alpha_expr.replace(f', {old_window})', f', {new_window})')
                    return new_alpha
        
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
                    
                    # 변수는 alpha1, 윈도우는 alpha2
                    new_alpha = alpha1.replace(f'{op}({var1}, {win1})', f'{op}({var1}, {win2})')
                    return new_alpha
        
        return None
    except:
        return None

def tournament_select(fitness_scores, k=3):
    """토너먼트 선택"""
    tournament = random.sample(fitness_scores, min(k, len(fitness_scores)))
    return max(tournament, key=lambda x: x[1])[0]

def genetic_programming_parallel(seed_alphas, data, generations=30, population_size=100, num_workers=None):
    """병렬 처리 Genetic Programming"""
    
    if num_workers is None:
        num_workers = min(cpu_count(), 8)
    
    print(f"\n🧬 병렬 GP 진화 시작")
    print(f"   세대: {generations}, 개체수: {population_size}, 워커: {num_workers}")
    
    # 초기 개체군
    population = seed_alphas[:population_size]
    while len(population) < population_size:
        parent = random.choice(seed_alphas)
        mutated = mutate_alpha(parent)
        if mutated:
            population.append(mutated)
    
    set_global_data(data)
    
    best_ever = (None, -999.0)
    
    for gen in range(generations):
        print(f"\n  세대 {gen+1}/{generations}")
        
        # 병렬 평가
        with Pool(num_workers, initializer=set_global_data, initargs=(data,)) as pool:
            results = pool.map(evaluate_alpha_ic_worker, population)
        
        fitness_scores = sorted(results, key=lambda x: x[1], reverse=True)
        
        best_ic = fitness_scores[0][1]
        print(f"    최고 IC: {best_ic:.4f}")
        
        if best_ic > best_ever[1]:
            best_ever = fitness_scores[0]
            print(f"    🏆 신기록! IC: {best_ic:.4f}")
        
        # 다음 세대
        next_population = []
        
        # 엘리트
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
    
    return best_ever

def main():
    print("=" * 80)
    print("Alpha-GPT: 15-day Forward + LLM + GP (Parallel)")
    print("=" * 80)
    print()
    
    # 데이터 로드
    data = load_market_data()
    
    # LLM seed 생성
    seed_alphas = generate_seed_alphas_with_llm(num_seeds=20)
    
    # GP 진화
    best_alpha, best_ic = genetic_programming_parallel(
        seed_alphas, 
        data, 
        generations=30, 
        population_size=100
    )
    
    # 결과
    print("\n" + "=" * 80)
    print("🏆 BEST ALPHA (15-day forward, LLM+GP)")
    print("=" * 80)
    print(f"IC: {best_ic:.4f}")
    print(f"Expression: {best_alpha}")
    print()
    
    # 거래비용 분석
    print("=" * 80)
    print("💰 Transaction Cost Analysis")
    print("=" * 80)
    print(f"Rebalancing frequency: Every 15 days")
    print(f"Rebalances per year: ~24")
    print(f"Transaction cost per trade: 0.3%")
    print(f"Total annual cost: ~14.4% (0.3% × 24 × 2)")
    print(f"\nNet IC after costs: ~{best_ic - 0.02:.4f}")
    
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
            '15-day forward alpha (bi-weekly rebalancing, LLM+GP parallel)'
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Saved!")
    
    print("\n🎉 완료!")

if __name__ == "__main__":
    main()
