#!/usr/bin/env python3
"""
재무제표 기반 알파 (Fundamental Alpha)
매출, 영업이익, EPS, ROE 등 펀더멘털 지표 활용
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

def load_fundamental_data():
    """재무제표 + 가격 데이터 통합"""
    print("📊 재무제표 데이터 로드 중...")
    print("   - 손익계산서 (매출, 영업이익, 순이익)")
    print("   - 재무상태표 (자산, 부채, 자본)")
    print("   - 현금흐름표")
    print("   - 주가 데이터")
    
    conn = get_db_connection()
    
    # 재무 데이터 있는 종목 중 시총 상위 100
    query_stocks = """
        SELECT DISTINCT ON (s.ticker)
            s.id, s.ticker, s.name
        FROM stocks s
        JOIN price_data p ON s.id = p.stock_id
        JOIN financial_statements f ON s.id = f.stock_id
        WHERE s.is_active = true
        AND p.date = (SELECT MAX(date) FROM price_data)
        AND f.revenue IS NOT NULL
        AND f.period_end >= CURRENT_DATE - INTERVAL '365 days'
        ORDER BY s.ticker, (p.close * p.volume) DESC
        LIMIT 100
    """
    stocks_df = pd.read_sql(query_stocks, conn)
    stock_ids = stocks_df['id'].tolist()
    stock_id_list = ', '.join(map(str, stock_ids))
    
    # 1. 가격 데이터 (2년)
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
    
    # 2. 재무제표 (분기별, EPS 제외)
    query_financial = f"""
        SELECT 
            s.ticker,
            f.period_end as date,
            f.revenue,
            f.operating_income,
            f.net_income,
            f.total_assets,
            f.total_equity,
            f.total_liabilities,
            f.operating_cash_flow,
            f.free_cash_flow
        FROM financial_statements f
        JOIN stocks s ON f.stock_id = s.id
        WHERE f.stock_id IN ({stock_id_list})
        AND f.period_end >= CURRENT_DATE - INTERVAL '730 days'
        AND f.revenue IS NOT NULL
        ORDER BY s.ticker, f.period_end
    """
    
    fin_df = pd.read_sql(query_financial, conn)
    conn.close()
    
    # 재무 데이터를 일별 데이터로 변환 (forward fill)
    revenue = fin_df.pivot(index='date', columns='ticker', values='revenue')
    operating_income = fin_df.pivot(index='date', columns='ticker', values='operating_income')
    net_income = fin_df.pivot(index='date', columns='ticker', values='net_income')
    total_assets = fin_df.pivot(index='date', columns='ticker', values='total_assets')
    total_equity = fin_df.pivot(index='date', columns='ticker', values='total_equity')
    total_liabilities = fin_df.pivot(index='date', columns='ticker', values='total_liabilities')
    operating_cf = fin_df.pivot(index='date', columns='ticker', values='operating_cash_flow')
    free_cf = fin_df.pivot(index='date', columns='ticker', values='free_cash_flow')
    
    # 일별 가격 인덱스에 맞춰 재무 데이터 forward fill
    all_dates = close.index
    revenue = revenue.reindex(all_dates).fillna(method='ffill')
    operating_income = operating_income.reindex(all_dates).fillna(method='ffill')
    net_income = net_income.reindex(all_dates).fillna(method='ffill')
    total_assets = total_assets.reindex(all_dates).fillna(method='ffill')
    total_equity = total_equity.reindex(all_dates).fillna(method='ffill')
    total_liabilities = total_liabilities.reindex(all_dates).fillna(method='ffill')
    operating_cf = operating_cf.reindex(all_dates).fillna(method='ffill')
    free_cf = free_cf.reindex(all_dates).fillna(method='ffill')
    
    print(f"✅ {len(close.columns)}개 종목, {len(close)}일 데이터")
    print(f"   재무제표: {len(fin_df)}개 분기 데이터")
    
    return {
        # 가격
        'close': close,
        'volume': volume,
        'returns': close.pct_change(),
        # 재무제표 (EPS 제외)
        'revenue': revenue,
        'operating_income': operating_income,
        'net_income': net_income,
        'total_assets': total_assets,
        'total_equity': total_equity,
        'total_liabilities': total_liabilities,
        'operating_cf': operating_cf,
        'free_cf': free_cf
    }

def generate_fundamental_alphas():
    """재무제표 기반 알파 생성"""
    
    print("\n🤖 재무제표 기반 알파 생성 중...")
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    prompt = """당신은 Value 투자 전문 퀀트입니다.
재무제표를 활용한 10일 보유 알파를 생성하세요.

## 📊 사용 가능한 재무 데이터

### 손익계산서
- revenue: 매출
- operating_income: 영업이익
- net_income: 순이익

### 재무상태표
- total_assets: 총자산
- total_equity: 자본 (순자산)
- total_liabilities: 부채

### 현금흐름표
- operating_cf: 영업활동현금흐름
- free_cf: 잉여현금흐름 (FCF)

⚠️ EPS 데이터 없음 - 사용하지 마세요

### 가격 데이터
- close: 주가
- volume: 거래량
- returns: 수익률

## 🔧 연산자
- ts_delta(x, period): N일 전 대비 변화
- ts_mean(x, window): 이동평균
- ts_rank(x, window): 순위 0~1
- normed_rank(x): 횡단면 순위

## 💡 Value & Quality 전략

### 1. Profitability (수익성)
- **ROE**: net_income / total_equity (높을수록 좋음)
- **영업이익률**: operating_income / revenue
- **순이익률**: net_income / revenue
- **ROA**: net_income / total_assets

### 2. Growth (성장성)
- **매출 성장**: ts_delta(revenue, 365) / revenue
- **순이익 성장**: ts_delta(net_income, 365) / net_income
- **영업이익 성장**: ts_delta(operating_income, 365) / operating_income
- **자본 성장**: ts_delta(total_equity, 365) / total_equity

### 3. Quality (재무 건전성)
- **부채비율**: total_liabilities / total_equity (낮을수록 좋음)
- **자기자본비율**: total_equity / total_assets (높을수록 좋음)
- **FCF 마진**: free_cf / revenue
- **이익의 질**: operating_cf / net_income (1 이상 좋음)

### 4. 복합 전략
- **고ROE + 저부채**: net_income/total_equity * (-total_liabilities/total_equity)
- **성장 + 수익성**: ts_delta(revenue, 365)/revenue * operating_income/revenue
- **현금흐름 우수**: free_cf/revenue * operating_cf/net_income

## 📝 알파 50개 생성

매우 다양한 접근으로 50개를 작성하세요. 단순한 것부터 복잡한 조합까지:

ALPHA_1: AlphaOperators.normed_rank(net_income / total_equity)
ALPHA_2: AlphaOperators.normed_rank(operating_income / revenue)
ALPHA_3: AlphaOperators.ts_rank(AlphaOperators.ts_delta(revenue, 365) / revenue, 60)
ALPHA_4: AlphaOperators.normed_rank(-total_liabilities / total_equity)
ALPHA_5: AlphaOperators.normed_rank(free_cf / revenue)
ALPHA_6: AlphaOperators.normed_rank(operating_cf / net_income)
ALPHA_7: AlphaOperators.ts_rank(AlphaOperators.ts_delta(net_income, 365) / net_income, 60)
ALPHA_8: AlphaOperators.normed_rank(total_equity / total_assets)
ALPHA_9: AlphaOperators.normed_rank(net_income / total_assets)
ALPHA_10: AlphaOperators.normed_rank(net_income / revenue)
ALPHA_11: AlphaOperators.ts_rank(AlphaOperators.ts_delta(operating_income, 365) / operating_income, 60)
ALPHA_12: AlphaOperators.normed_rank(operating_income / total_assets)
ALPHA_13: AlphaOperators.normed_rank(revenue / total_assets)
ALPHA_14: AlphaOperators.ts_rank(AlphaOperators.ts_delta(total_equity, 365) / total_equity, 60)
ALPHA_15: AlphaOperators.normed_rank(free_cf / net_income)
ALPHA_16: AlphaOperators.normed_rank((net_income / total_equity) * (-total_liabilities / total_equity))
ALPHA_17: AlphaOperators.normed_rank((operating_income / revenue) * (total_equity / total_assets))
ALPHA_18: AlphaOperators.ts_rank(AlphaOperators.ts_delta(revenue, 365) / revenue, 60) * AlphaOperators.normed_rank(operating_income / revenue)
ALPHA_19: AlphaOperators.normed_rank((free_cf / revenue) * (operating_cf / net_income))
ALPHA_20: AlphaOperators.normed_rank(net_income / total_equity) + AlphaOperators.normed_rank(-total_liabilities / total_equity)
...
ALPHA_50: [더 복잡한 조합, 4-5개 지표 결합]

매우 창의적인 조합을 만드세요! 단순한 것부터 매우 복잡한 것까지 골고루!
"""

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a fundamental analysis expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=1.0,  # 높은 다양성
        max_tokens=4000  # 50개 알파 생성
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
    
    print(f"✅ {len(alphas)}개 재무 알파 생성")
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
        # 변수 바인딩 (EPS 제외)
        close = data['close']
        volume = data['volume']
        returns = data['returns']
        revenue = data['revenue']
        operating_income = data['operating_income']
        net_income = data['net_income']
        total_assets = data['total_assets']
        total_equity = data['total_equity']
        total_liabilities = data['total_liabilities']
        operating_cf = data['operating_cf']
        free_cf = data['free_cf']
        
        # 10일 후 수익률
        returns_forward_10 = close.pct_change(10).shift(-10)
        
        alpha_values = eval(alpha_expr)
        
        ic_list = []
        for date in alpha_values.index[:-10]:
            alpha_cs = alpha_values.loc[date]
            returns_cs = returns_forward_10.loc[date]
            valid = alpha_cs.notna() & returns_cs.notna() & (alpha_cs != np.inf) & (alpha_cs != -np.inf)
            
            if valid.sum() > 10:
                ic = alpha_cs[valid].corr(returns_cs[valid])
                if not np.isnan(ic):
                    ic_list.append(ic)
        
        if len(ic_list) < 10:
            return (alpha_expr, -999.0)
        
        return (alpha_expr, np.mean(ic_list))
        
    except Exception as e:
        return (alpha_expr, -999.0)

def genetic_programming_parallel(seed_alphas, data, generations=15, population_size=200):
    """병렬 GP"""
    num_workers = min(cpu_count(), 8)
    
    print(f"\n🧬 병렬 GP 진화 (재무 알파)")
    print(f"   세대: {generations}, 개체수: {population_size}")
    
    population = seed_alphas[:population_size]
    while len(population) < population_size:
        population.append(random.choice(seed_alphas))
    
    set_global_data(data)
    
    for gen in range(generations):
        print(f"\n  세대 {gen+1}/{generations}")
        
        with Pool(num_workers, initializer=set_global_data, initargs=(data,)) as pool:
            results = pool.map(evaluate_alpha_ic_worker, population)
        
        fitness_scores = sorted(results, key=lambda x: x[1], reverse=True)
        
        best_ic = fitness_scores[0][1]
        print(f"    최고 IC: {best_ic:.4f}")
        
        next_population = []
        
        elite_count = population_size // 5
        for alpha, _ in fitness_scores[:elite_count]:
            next_population.append(alpha)
        
        while len(next_population) < population_size:
            parent = random.choice([a for a, _ in fitness_scores[:population_size//2]])
            next_population.append(parent)
        
        population = next_population
    
    with Pool(num_workers, initializer=set_global_data, initargs=(data,)) as pool:
        final_results = pool.map(evaluate_alpha_ic_worker, population)
    
    final_fitness = sorted(final_results, key=lambda x: x[1], reverse=True)
    
    print(f"\n✅ GP 완료! 최고 IC: {final_fitness[0][1]:.4f}")
    
    return final_fitness

def main():
    print("=" * 70)
    print("재무제표 기반 알파 (Fundamental Alpha)")
    print("=" * 70)
    print()
    
    # 데이터 로드
    data = load_fundamental_data()
    
    # LLM 알파 생성
    seed_alphas = generate_fundamental_alphas()
    
    print("\n📊 생성된 재무 알파:")
    for i, alpha in enumerate(seed_alphas, 1):
        print(f"   {i}. {alpha[:80]}...")
    
    # GP 진화 (Large-scale: 50 seeds → 200 population)
    evolved_alphas = genetic_programming_parallel(
        seed_alphas=seed_alphas,
        data=data,
        generations=15,  # 더 많은 세대
        population_size=200  # 대규모 탐색
    )
    
    # 결과
    print("\n" + "=" * 70)
    print("🏆 진화된 상위 5개 재무 알파")
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
            f"IC: {best_ic:.4f}, Fundamental (Value+Quality+Growth), 10-day forward"
        ))
        conn.commit()
        print("\n✅ DB 저장 완료")
    finally:
        cur.close()
        conn.close()
    
    print("\n🎉 재무 알파 생성 완료!")

if __name__ == "__main__":
    main()
