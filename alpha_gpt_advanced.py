#!/usr/bin/env python3
"""
고급 알파: 기술적 지표 + 재무 + 수급 복합
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

# 프로젝트 루트
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

def load_advanced_data():
    """모든 지표 통합 로드"""
    print("📊 고급 데이터 로드 중...")
    print("   - 가격 데이터")
    print("   - 기술적 지표 (RSI, MACD, BB)")
    print("   - 수급 데이터 (외국인, 기관)")
    
    conn = get_db_connection()
    
    # 시총 상위 100 종목
    query_stocks = """
        SELECT DISTINCT ON (s.ticker)
            s.id, s.ticker, s.name
        FROM stocks s
        JOIN price_data p ON s.id = p.stock_id
        WHERE s.is_active = true
        AND p.date = (SELECT MAX(date) FROM price_data)
        ORDER BY s.ticker, (p.close * p.volume) DESC
        LIMIT 100
    """
    stocks_df = pd.read_sql(query_stocks, conn)
    stock_ids = stocks_df['id'].tolist()
    stock_id_list = ', '.join(map(str, stock_ids))
    
    # 1. 가격 데이터
    query_price = f"""
        SELECT s.ticker, p.date, p.close, p.volume
        FROM price_data p
        JOIN stocks s ON p.stock_id = s.id
        WHERE p.stock_id IN ({stock_id_list})
        AND p.date >= CURRENT_DATE - INTERVAL '730 days'
        ORDER BY s.ticker, p.date
    """
    price_df = pd.read_sql(query_price, conn)
    close = price_df.pivot(index='date', columns='ticker', values='close')
    volume = price_df.pivot(index='date', columns='ticker', values='volume')
    
    # 2. 기술적 지표
    query_tech = f"""
        SELECT s.ticker, t.date, t.rsi_14, t.macd, t.bb_upper, t.bb_lower, 
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
    bb_upper = tech_df.pivot(index='date', columns='ticker', values='bb_upper')
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
    
    print(f"✅ {len(close.columns)}개 종목, {len(close)}일 데이터")
    print(f"   - RSI: {len(rsi)}일")
    print(f"   - 외국인 순매수: {len(foreign_net)}일")
    
    return {
        'close': close,
        'volume': volume,
        'returns': close.pct_change(),
        'rsi': rsi,
        'macd': macd,
        'bb_upper': bb_upper,
        'bb_lower': bb_lower,
        'sma_20': sma_20,
        'sma_50': sma_50,
        'volatility': volatility,
        'foreign_net': foreign_net,
        'institution_net': institution_net,
        'foreign_own': foreign_own,
        'short_ratio': short_ratio
    }

def generate_advanced_alphas():
    """고급 복합 알파 생성"""
    print("\n🤖 LLM이 고급 복합 알파 생성 중...")
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    prompt = """당신은 퀀트 개발자입니다. 한국 증시에서 10일 보유 시 수익이 높을 종목을 찾는 고급 알파를 생성하세요.

사용 가능한 데이터:
- close, volume, returns (가격)
- rsi, macd, bb_upper, bb_lower (기술적 지표)
- sma_20, sma_50, volatility (추세/변동성)
- foreign_net, institution_net (수급)
- foreign_own, short_ratio (외국인 지분, 공매도)

전략 아이디어:
1. RSI 과매도 + 외국인 순매수 증가 = 반등 기대
2. MACD 상향 돌파 + 거래량 증가 = 모멘텀
3. 볼린저 하단 터치 + 기관 매수 = 저점 매수
4. 낮은 변동성 + 꾸준한 상승 = 안정적 수익
5. 외국인 지분 증가 추세 = 장기 강세

연산자: ts_delta, ts_mean, ts_std, ts_rank

10개의 다양한 고급 알파를 생성하세요:

ALPHA_1: [표현식]
..."""

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
    
    if len(alphas) == 0:
        # 기본 고급 알파
        alphas = [
            "AlphaOperators.ts_rank((rsi < 30) * foreign_net, 20)",
            "AlphaOperators.ts_rank(macd, 20) * AlphaOperators.ts_rank(AlphaOperators.ts_delta(volume, 5), 20)",
            "AlphaOperators.ts_rank((close < bb_lower) * institution_net, 20)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_mean(returns, 10) / volatility, 30)",
            "AlphaOperators.ts_rank(AlphaOperators.ts_delta(foreign_own, 20), 40)"
        ]
    
    print(f"✅ {len(alphas)}개 고급 알파 생성")
    return alphas

_global_data = None

def set_global_data(data):
    global _global_data
    _global_data = data

def evaluate_alpha_ic_worker(alpha_expr):
    """10일 후 수익률 예측"""
    global _global_data
    data = _global_data
    
    try:
        # 변수 바인딩
        close = data['close']
        volume = data['volume']
        returns = data['returns']
        rsi = data['rsi']
        macd = data['macd']
        bb_upper = data['bb_upper']
        bb_lower = data['bb_lower']
        sma_20 = data['sma_20']
        sma_50 = data['sma_50']
        volatility = data['volatility']
        foreign_net = data['foreign_net']
        institution_net = data['institution_net']
        foreign_own = data['foreign_own']
        short_ratio = data['short_ratio']
        
        # 10일 후 수익률
        returns_forward_10 = close.pct_change(10).shift(-10)
        
        alpha_values = eval(alpha_expr)
        
        ic_list = []
        for date in alpha_values.index[:-10]:
            alpha_cs = alpha_values.loc[date]
            returns_cs = returns_forward_10.loc[date]
            valid = alpha_cs.notna() & returns_cs.notna()
            
            if valid.sum() > 10:
                ic = alpha_cs[valid].corr(returns_cs[valid])
                if not np.isnan(ic):
                    ic_list.append(ic)
        
        if len(ic_list) < 10:
            return (alpha_expr, -999.0)
        
        return (alpha_expr, np.mean(ic_list))
        
    except Exception as e:
        return (alpha_expr, -999.0)

def genetic_programming_parallel(seed_alphas, data, generations=10, population_size=100):
    """병렬 GP"""
    num_workers = min(cpu_count(), 8)
    
    print(f"\n🧬 병렬 GP 진화 (고급 알파)")
    print(f"   세대: {generations}, 개체수: {population_size}")
    
    population = seed_alphas[:population_size]
    while len(population) < population_size:
        parent = random.choice(seed_alphas)
        population.append(parent)  # 복잡한 알파는 변이 스킵
    
    set_global_data(data)
    
    for gen in range(generations):
        print(f"\n  세대 {gen+1}/{generations}")
        
        with Pool(num_workers, initializer=set_global_data, initargs=(data,)) as pool:
            results = pool.map(evaluate_alpha_ic_worker, population)
        
        fitness_scores = sorted(results, key=lambda x: x[1], reverse=True)
        
        best_ic = fitness_scores[0][1]
        print(f"    최고 IC: {best_ic:.4f}")
        
        next_population = []
        
        # 엘리트
        elite_count = population_size // 5
        for alpha, _ in fitness_scores[:elite_count]:
            next_population.append(alpha)
        
        # 나머지는 상위에서 복제
        while len(next_population) < population_size:
            parent = random.choice([a for a, _ in fitness_scores[:population_size//2]])
            next_population.append(parent)
        
        population = next_population
    
    # 최종 평가
    with Pool(num_workers, initializer=set_global_data, initargs=(data,)) as pool:
        final_results = pool.map(evaluate_alpha_ic_worker, population)
    
    final_fitness = sorted(final_results, key=lambda x: x[1], reverse=True)
    
    print(f"\n✅ GP 완료! 최고 IC: {final_fitness[0][1]:.4f}")
    
    return final_fitness

def main():
    print("=" * 70)
    print("고급 복합 알파: 기술적 지표 + 수급 데이터")
    print("=" * 70)
    print()
    
    # 데이터 로드
    data = load_advanced_data()
    
    # LLM 알파 생성
    seed_alphas = generate_advanced_alphas()
    
    print("\n📊 초기 알파:")
    for i, alpha in enumerate(seed_alphas[:5], 1):
        print(f"   {i}. {alpha[:80]}...")
    
    # GP 진화
    evolved_alphas = genetic_programming_parallel(
        seed_alphas=seed_alphas,
        data=data,
        generations=10,
        population_size=50  # 복잡한 알파는 작게
    )
    
    # 결과
    print("\n" + "=" * 70)
    print("🏆 진화된 상위 5개 고급 알파")
    print("=" * 70)
    
    for i, (alpha, ic) in enumerate(evolved_alphas[:5], 1):
        print(f"\n{i}. IC: {ic:.4f}")
        print(f"   {alpha}")
    
    # DB 저장
    best_alpha, best_ic = evolved_alphas[0]
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            INSERT INTO alpha_performance
            (alpha_formula, start_date, is_active, sharpe_ratio, notes)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (alpha_formula, start_date) DO UPDATE
            SET sharpe_ratio = EXCLUDED.sharpe_ratio, notes = EXCLUDED.notes
        """, (
            best_alpha,
            date.today(),
            True,
            float(best_ic * 10),
            f"IC: {best_ic:.4f}, Advanced (Tech+Supply+Demand), 10-day forward"
        ))
        conn.commit()
        print("\n✅ DB 저장 완료")
    finally:
        cur.close()
        conn.close()
    
    print("\n🎉 고급 알파 생성 완료!")

if __name__ == "__main__":
    main()
