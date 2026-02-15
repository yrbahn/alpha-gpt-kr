#!/usr/bin/env python3
"""
Alpha-GPT: 15-day Forward + Checkpoint (Memory Safe)
각 세대마다 체크포인트 저장 → 메모리 리프레시
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2
import openai
import random
import json
import gc

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.mining.operators import AlphaOperators

load_dotenv()

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )

def load_market_data():
    """2000개 종목 데이터 로드"""
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
    
    close_pivot = price_df.pivot(index='date', columns='ticker', values='close').astype('float32')
    volume_pivot = price_df.pivot(index='date', columns='ticker', values='volume').astype('float32')
    
    del price_df
    gc.collect()
    
    returns_15d = (close_pivot.shift(-15) / close_pivot - 1).astype('float32')
    
    print(f"✅ {len(close_pivot.columns)}개 종목, {len(close_pivot)}일 데이터")
    
    return {
        'close': close_pivot,
        'volume': volume_pivot,
        'returns': close_pivot.pct_change().astype('float32'),
        'forward_return_15d': returns_15d
    }

def generate_seed_alphas(num_seeds=20):
    """LLM으로 초기 알파 생성"""
    print(f"\n🤖 LLM이 초기 {num_seeds}개 알파 생성 중...")
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    prompt = f"""Generate {num_seeds} alpha factors for 15-day forward prediction.

Available: close, volume, returns
Operators: AlphaOperators.ts_delta(x, p), ts_mean(x, w), ts_std(x, w), ts_rank(x, w), zscore_scale(x), normed_rank(x)

Output ONLY Python expressions, one per line."""

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

def evaluate_alpha(alpha_expr, data):
    """알파 평가 (전체 데이터)"""
    try:
        close = data['close']
        volume = data['volume']
        returns = data['returns']
        forward_return_15d = data['forward_return_15d']
        
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

def save_checkpoint(generation, population, best_ever, checkpoint_file):
    """체크포인트 저장"""
    checkpoint = {
        'generation': generation,
        'population': population,
        'best_alpha': best_ever[0],
        'best_ic': best_ever[1],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"    💾 체크포인트 저장: 세대 {generation}")

def load_checkpoint(checkpoint_file):
    """체크포인트 로드"""
    if not checkpoint_file.exists():
        return None
    
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    
    print(f"\n✅ 체크포인트 로드: 세대 {checkpoint['generation']} (IC: {checkpoint['best_ic']:.4f})")
    return checkpoint

def genetic_programming_with_checkpoint(seed_alphas, data, generations=30, population_size=100):
    """체크포인트 기반 GP"""
    
    checkpoint_file = CHECKPOINT_DIR / "gp_checkpoint.json"
    
    # 체크포인트 확인
    checkpoint = load_checkpoint(checkpoint_file)
    
    if checkpoint:
        start_gen = checkpoint['generation'] + 1
        population = checkpoint['population']
        best_ever = (checkpoint['best_alpha'], checkpoint['best_ic'])
        print(f"   세대 {start_gen}부터 재개...")
    else:
        start_gen = 1
        population = seed_alphas[:population_size]
        while len(population) < population_size:
            parent = random.choice(seed_alphas)
            mutated = mutate_alpha(parent)
            if mutated:
                population.append(mutated)
        best_ever = (None, -999.0)
    
    print(f"\n🧬 체크포인트 기반 GP 진화")
    print(f"   세대: {start_gen}~{generations}, 개체수: {population_size}")
    print(f"   체크포인트: {checkpoint_file}")
    
    for gen in range(start_gen, generations + 1):
        print(f"\n  세대 {gen}/{generations}")
        
        # 평가
        fitness_scores = []
        for i, alpha in enumerate(population):
            if i % 20 == 0:
                print(f"    평가 진행: {i}/{len(population)}", end='\r')
            ic = evaluate_alpha(alpha, data)
            fitness_scores.append((alpha, ic))
        
        # 정렬
        fitness_scores = sorted(fitness_scores, key=lambda x: x[1], reverse=True)
        
        best_ic = fitness_scores[0][1]
        print(f"    최고 IC: {best_ic:.4f}" + " " * 30)
        
        if best_ic > best_ever[1]:
            best_ever = fitness_scores[0]
            print(f"    🏆 신기록! IC: {best_ic:.4f}")
        
        # 💾 체크포인트 저장 (매 세대)
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
        
        # 체크포인트 저장
        save_checkpoint(gen, population, best_ever, checkpoint_file)
        
        # 🧹 메모리 정리
        del fitness_scores
        gc.collect()
    
    return best_ever

def main():
    print("=" * 80)
    print("Alpha-GPT: 15-day Forward (Checkpoint-based)")
    print("=" * 80)
    print()
    
    # 데이터 로드
    data = load_market_data()
    
    # Seed 알파 생성 (또는 재사용)
    seed_file = CHECKPOINT_DIR / "seed_alphas.json"
    
    if seed_file.exists():
        print("\n✅ 기존 seed alphas 로드")
        with open(seed_file, 'r') as f:
            seed_alphas = json.load(f)
    else:
        seed_alphas = generate_seed_alphas(num_seeds=20)
        with open(seed_file, 'w') as f:
            json.dump(seed_alphas, f, indent=2)
        print(f"    💾 Seed alphas 저장: {seed_file}")
    
    # GP 실행
    best_alpha, best_ic = genetic_programming_with_checkpoint(
        seed_alphas, 
        data, 
        generations=30, 
        population_size=100
    )
    
    # 결과
    print("\n" + "=" * 80)
    print("🏆 BEST ALPHA (15-day forward, Checkpoint-based)")
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
            '15-day forward alpha (2000 stocks, checkpoint-based)'
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ Saved!")
    
    # 체크포인트 정리
    cleanup = input("\n🗑️  Delete checkpoints? (y/n): ")
    if cleanup.lower() == 'y':
        import shutil
        shutil.rmtree(CHECKPOINT_DIR)
        print("✅ Checkpoints deleted")
    
    print("\n🎉 완료!")

if __name__ == "__main__":
    main()
