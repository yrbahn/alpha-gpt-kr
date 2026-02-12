#!/usr/bin/env python3
"""
Alpha-GPT 완전판: LLM 생성 + GP 진화
"""

import sys
import os
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
import openai
import random

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
    """시가총액 상위 100개 종목 (GP 진화용 - 빠른 평가)"""
    print("📊 데이터 로드 중...")
    
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
    
    # 가격 데이터 (최근 180일)
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
        AND p.date >= CURRENT_DATE - INTERVAL '180 days'
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
def generate_seed_alphas_with_llm(num_seeds=5):
    """LLM으로 초기 알파 생성"""
    
    print(f"\n🤖 LLM이 초기 {num_seeds}개 알파 생성 중...")
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    prompt = f"""당신은 퀀트 개발자입니다. 한국 증시에서 강한 모멘텀과 낮은 변동성을 가진 종목을 찾는 알파 표현식을 생성하세요.

사용 가능한 데이터:
- close: 종가 (pandas DataFrame)
- volume: 거래량 (pandas DataFrame)  
- returns: 수익률 (pandas DataFrame)

사용 가능한 연산자 (AlphaOperators):
- ts_delta(x, period): 현재값 - N일 전 값
- ts_mean(x, window): 이동 평균
- ts_std(x, window): 이동 표준편차
- ts_rank(x, window): 순위 (0~1)

규칙:
1. 간단하고 실행 가능한 표현식
2. 한 줄로 작성
3. AlphaOperators. 접두사 사용

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
            # "ALPHA_1:" 등의 접두사 제거
            if ':' in line:
                line = line.split(':', 1)[1].strip()
            # 주석 제거
            if '#' in line:
                line = line.split('#')[0].strip()
            if line:
                alphas.append(line)
    
    # LLM 실패시 미리 정의된 seed alphas 사용
    if len(alphas) == 0:
        print("   ⚠️  LLM 생성 실패, 기본 seed alphas 사용")
        alphas = [
            "AlphaOperators.ts_rank(AlphaOperators.ts_delta(close, 20), 10)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_std(returns, 10) / AlphaOperators.ts_std(returns, 20), 10)",
            "AlphaOperators.ts_rank(close / AlphaOperators.ts_mean(close, 20), 10)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_delta(close, 5) / AlphaOperators.ts_std(close, 20), 15)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_mean(returns, 5), 10)"
        ]
    
    print(f"✅ {len(alphas)}개 초기 알파 생성")
    return alphas

# 알파 평가 (적합도 함수)
def evaluate_alpha_ic(alpha_expr, data):
    """알파의 IC 계산"""
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
            return -999.0  # 페널티
        
        return np.mean(ic_list)
        
    except:
        return -999.0

# GP 진화 알고리즘
def genetic_programming_evolution(seed_alphas, data, generations=20, population_size=30):
    """Genetic Programming으로 알파 진화"""
    
    print(f"\n🧬 GP 진화 시작: {generations}세대, 개체수 {population_size}")
    
    # 초기 개체군 생성
    population = seed_alphas[:population_size]
    
    # 부족하면 변이로 채우기
    while len(population) < population_size:
        parent = random.choice(seed_alphas)
        mutated = mutate_alpha(parent)
        if mutated:
            population.append(mutated)
    
    best_ic_history = []
    
    for gen in range(generations):
        print(f"\n  세대 {gen+1}/{generations}")
        
        # 적합도 평가
        fitness_scores = []
        for i, alpha in enumerate(population):
            ic = evaluate_alpha_ic(alpha, data)
            fitness_scores.append((ic, alpha))
            
            if (i+1) % 10 == 0:
                print(f"    평가 중... {i+1}/{len(population)}")
        
        # 정렬 (높은 IC 우선)
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        # 상위 IC 출력
        best_ic = fitness_scores[0][0]
        best_ic_history.append(best_ic)
        print(f"    최고 IC: {best_ic:.4f}")
        
        # 조기 종료 (IC가 충분히 높으면)
        if best_ic > 0.05:
            print(f"    ✅ 목표 달성! IC > 0.05")
            break
        
        # 다음 세대 생성
        next_population = []
        
        # 엘리트 보존 (상위 20%)
        elite_count = population_size // 5
        for _, alpha in fitness_scores[:elite_count]:
            next_population.append(alpha)
        
        # 교차 + 변이로 나머지 채우기
        while len(next_population) < population_size:
            if random.random() < 0.7:  # 70% 확률로 교차
                parent1 = tournament_select(fitness_scores)
                parent2 = tournament_select(fitness_scores)
                child = crossover_alphas(parent1, parent2)
                if child:
                    next_population.append(child)
                else:
                    next_population.append(parent1)
            else:  # 30% 확률로 변이
                parent = tournament_select(fitness_scores)
                mutated = mutate_alpha(parent)
                if mutated:
                    next_population.append(mutated)
                else:
                    next_population.append(parent)
        
        population = next_population[:population_size]
    
    # 최종 평가
    final_fitness = [(evaluate_alpha_ic(alpha, data), alpha) for alpha in population]
    final_fitness.sort(key=lambda x: x[0], reverse=True)
    
    print(f"\n✅ GP 진화 완료!")
    print(f"   최종 최고 IC: {final_fitness[0][0]:.4f}")
    
    return final_fitness

# GP 연산자들
def tournament_select(fitness_scores, k=3):
    """토너먼트 선택"""
    candidates = random.sample(fitness_scores, k)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def crossover_alphas(alpha1, alpha2):
    """두 알파 표현식 교차"""
    try:
        # 간단한 교차: 부분 표현식 교환
        tokens1 = alpha1.split('(')
        tokens2 = alpha2.split('(')
        
        if len(tokens1) > 2 and len(tokens2) > 2:
            # 중간 부분 교환
            point = random.randint(1, min(len(tokens1), len(tokens2)) - 1)
            child_tokens = tokens1[:point] + tokens2[point:]
            return '('.join(child_tokens)
        
        return None
    except:
        return None

def mutate_alpha(alpha):
    """알파 표현식 변이"""
    try:
        # 숫자 파라미터 변경
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

# 메인 함수
def main():
    print("=" * 70)
    print("Alpha-GPT 완전판: LLM 초기 생성 + GP 진화")
    print("=" * 70)
    print()
    
    # 1. 데이터 로드
    data = load_market_data()
    
    # 2. LLM으로 초기 알파 생성
    seed_alphas = generate_seed_alphas_with_llm(num_seeds=5)
    
    print("\n📊 초기 알파:")
    for i, alpha in enumerate(seed_alphas, 1):
        print(f"   {i}. {alpha[:80]}...")
    
    # 3. GP 진화
    evolved_alphas = genetic_programming_evolution(
        seed_alphas=seed_alphas,
        data=data,
        generations=30,
        population_size=20
    )
    
    # 4. 상위 5개 출력
    print("\n" + "=" * 70)
    print("🏆 진화된 상위 5개 알파")
    print("=" * 70)
    
    for i, (ic, alpha) in enumerate(evolved_alphas[:5], 1):
        print(f"\n{i}. IC: {ic:.4f}")
        print(f"   {alpha}")
    
    # 5. 최상위 알파 DB 저장
    best_ic, best_alpha = evolved_alphas[0]
    
    print(f"\n💾 최상위 알파 DB 저장...")
    print(f"   IC: {best_ic:.4f}")
    print(f"   공식: {best_alpha}")
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            INSERT INTO alpha_performance
            (alpha_formula, start_date, is_active, sharpe_ratio, notes)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (alpha_formula, start_date) DO NOTHING
        """, (
            best_alpha,
            date.today(),
            True,
            float(best_ic * 10),  # IC를 샤프 비율로 근사
            f"IC: {best_ic:.4f}, Generated by Alpha-GPT (LLM+GP)"
        ))
        conn.commit()
        print("✅ DB 저장 완료")
    finally:
        cur.close()
        conn.close()
    
    print("\n🎉 Alpha-GPT 완전판 실행 완료!")
    print(f"\n다음 단계:")
    print(f"  python3 apply_best_alpha_gp.py")

if __name__ == "__main__":
    main()
