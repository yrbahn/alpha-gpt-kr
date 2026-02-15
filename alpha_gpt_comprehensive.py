#!/usr/bin/env python3
"""
종합 알파 생성: 논문 방식 + 모든 지표 명시
LLM에게 사용 가능한 모든 데이터와 연산자를 상세히 제공
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

def load_comprehensive_data():
    """모든 지표 통합 로드"""
    print("📊 종합 데이터 로드 중...")
    
    conn = get_db_connection()
    
    # 시총 상위 100
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
        SELECT s.ticker, t.date, t.rsi_14, t.macd, t.macd_signal, t.macd_hist,
               t.bb_upper, t.bb_middle, t.bb_lower, t.atr_14,
               t.sma_20, t.sma_50, t.sma_200, t.volatility_20d
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
    macd_hist = tech_df.pivot(index='date', columns='ticker', values='macd_hist')
    bb_upper = tech_df.pivot(index='date', columns='ticker', values='bb_upper')
    bb_middle = tech_df.pivot(index='date', columns='ticker', values='bb_middle')
    bb_lower = tech_df.pivot(index='date', columns='ticker', values='bb_lower')
    atr = tech_df.pivot(index='date', columns='ticker', values='atr_14')
    sma_20 = tech_df.pivot(index='date', columns='ticker', values='sma_20')
    sma_50 = tech_df.pivot(index='date', columns='ticker', values='sma_50')
    sma_200 = tech_df.pivot(index='date', columns='ticker', values='sma_200')
    volatility = tech_df.pivot(index='date', columns='ticker', values='volatility_20d')
    
    # 3. 수급 데이터
    query_supply = f"""
        SELECT s.ticker, sd.date, sd.foreign_net_buy, sd.institution_net_buy,
               sd.individual_net_buy, sd.foreign_ownership,
               sd.short_ratio, sd.margin_ratio
        FROM supply_demand_data sd
        JOIN stocks s ON sd.stock_id = s.id
        WHERE sd.stock_id IN ({stock_id_list})
        AND sd.date >= CURRENT_DATE - INTERVAL '730 days'
        ORDER BY s.ticker, sd.date
    """
    supply_df = pd.read_sql(query_supply, conn)
    foreign_net = supply_df.pivot(index='date', columns='ticker', values='foreign_net_buy')
    institution_net = supply_df.pivot(index='date', columns='ticker', values='institution_net_buy')
    individual_net = supply_df.pivot(index='date', columns='ticker', values='individual_net_buy')
    foreign_own = supply_df.pivot(index='date', columns='ticker', values='foreign_ownership')
    short_ratio = supply_df.pivot(index='date', columns='ticker', values='short_ratio')
    margin_ratio = supply_df.pivot(index='date', columns='ticker', values='margin_ratio')
    
    conn.close()
    
    print(f"✅ {len(close.columns)}개 종목, {len(close)}일 데이터")
    
    return {
        # 가격
        'close': close, 'open': open_px, 'high': high, 'low': low, 'volume': volume,
        'returns': close.pct_change(),
        # 기술적
        'rsi': rsi, 'macd': macd, 'macd_signal': macd_signal, 'macd_hist': macd_hist,
        'bb_upper': bb_upper, 'bb_middle': bb_middle, 'bb_lower': bb_lower,
        'atr': atr, 'sma_20': sma_20, 'sma_50': sma_50, 'sma_200': sma_200,
        'volatility': volatility,
        # 수급
        'foreign_net': foreign_net, 'institution_net': institution_net,
        'individual_net': individual_net, 'foreign_own': foreign_own,
        'short_ratio': short_ratio, 'margin_ratio': margin_ratio
    }

def generate_comprehensive_alphas():
    """논문 방식 프롬프트: 모든 지표와 연산자 명시"""
    
    print("\n🤖 LLM에게 모든 지표 제공하여 종합 알파 생성 중...")
    
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # 논문 방식: 상세한 데이터 필드 및 연산자 설명
    prompt = """당신은 WorldQuant 수준의 퀀트 개발자입니다.
한국 증시에서 10일 보유 시 수익이 높을 종목을 찾는 알파를 생성하세요.

## 📊 사용 가능한 데이터 필드 (전체 목록)

### 1. 가격 데이터
- close: 종가
- open: 시가
- high: 고가
- low: 저가
- volume: 거래량
- returns: 수익률 (close.pct_change())

### 2. 기술적 지표
- rsi: RSI(14) - 상대강도지수 (0~100, 30이하 과매도, 70이상 과매수)
- macd: MACD - 모멘텀 지표
- macd_signal: MACD 시그널선
- macd_hist: MACD 히스토그램 (macd - signal)
- bb_upper: 볼린저 밴드 상단
- bb_middle: 볼린저 밴드 중간 (20일 이동평균)
- bb_lower: 볼린저 밴드 하단
- atr: ATR(14) - Average True Range (변동성)
- sma_20: 20일 단순이동평균
- sma_50: 50일 단순이동평균
- sma_200: 200일 단순이동평균
- volatility: 20일 변동성 (표준편차)

### 3. 수급 데이터
- foreign_net: 외국인 순매수량 (양수=매수, 음수=매도)
- institution_net: 기관 순매수량
- individual_net: 개인 순매수량
- foreign_own: 외국인 보유 지분율 (%)
- short_ratio: 공매도 비율 (%)
- margin_ratio: 신용거래 비율 (%)

## 🔧 사용 가능한 연산자

### Time-Series 연산자 (시계열)
- ts_delta(x, period): 현재값 - N일 전 값 (예: ts_delta(close, 20) = 20일 가격 변화)
- ts_mean(x, window): N일 이동평균 (예: ts_mean(volume, 10) = 10일 평균 거래량)
- ts_std(x, window): N일 이동 표준편차 (변동성)
- ts_rank(x, window): N일 기준 순위 0~1 (예: ts_rank(close, 20) = 20일 중 현재 가격 순위)
- ts_corr(x, y, window): N일 상관계수
- ts_min(x, window), ts_max(x, window): N일 최소/최대값

### Cross-Sectional 연산자 (횡단면)
- zscore_scale(x): Z-score 정규화 (평균=0, 표준편차=1)
- normed_rank(x): 순위 정규화 0~1

## 💡 전략 아이디어 (각각 다른 접근)

1. **RSI 역발상**: RSI < 30 과매도 + 외국인 순매수 증가 → 반등 기대
2. **MACD 골든크로스**: macd > macd_signal 전환 + 거래량 증가 → 상승 모멘텀
3. **볼린저 하단 터치**: close < bb_lower + 기관 순매수 → 저점 매수 기회
4. **이동평균 정배열**: sma_20 > sma_50 > sma_200 + 낮은 변동성 → 안정적 상승
5. **외국인 매집**: ts_delta(foreign_own, 60) > 0 (60일 지분율 증가) → 장기 강세
6. **공매도 커버링**: short_ratio 감소 + 가격 상승 → 공매도 청산 압력
7. **샤프 비율 우수**: ts_mean(returns, 10) / volatility → 위험 대비 수익률
8. **거래량 돌파**: ts_rank(volume, 20) > 0.9 + ts_delta(close, 5) > 0 → 거래량 동반 상승
9. **상대강도**: (close / sma_20) * ts_rank(returns, 20) → 추세 강도
10. **수급 일치**: foreign_net + institution_net (외국인+기관 동시 매수)

## 📝 출력 형식

50개의 매우 다양한 알파를 생성하세요. 단순한 것부터 복잡한 조합까지:

ALPHA_1: AlphaOperators.ts_rank((rsi < 30) * foreign_net, 20)
ALPHA_2: AlphaOperators.ts_rank(macd - macd_signal, 20) * AlphaOperators.ts_rank(AlphaOperators.ts_delta(volume, 5), 20)
ALPHA_3: AlphaOperators.normed_rank((close < bb_lower) * institution_net)
ALPHA_4: AlphaOperators.ts_rank(sma_20 > sma_50, 20) * AlphaOperators.normed_rank(-volatility)
ALPHA_5: AlphaOperators.normed_rank(AlphaOperators.ts_delta(foreign_own, 60))
...
ALPHA_50: [매우 복잡한 조합]

규칙:
- AlphaOperators. 접두사 필수
- 모든 변수명 정확히 사용 (위 목록 참고)
- 복잡한 수식 환영 (2~3개 연산자 조합)
- 한국 증시 특성 고려 (외국인/기관 영향 큼)
"""

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a world-class quantitative researcher."},
            {"role": "user", "content": prompt}
        ],
        temperature=1.0,  # 최대 다양성
        max_tokens=5000  # 50개 알파
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
    
    print(f"✅ {len(alphas)}개 종합 알파 생성")
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
        # 모든 변수 바인딩
        close = data['close']
        open_px = data['open']
        high = data['high']
        low = data['low']
        volume = data['volume']
        returns = data['returns']
        rsi = data['rsi']
        macd = data['macd']
        macd_signal = data['macd_signal']
        macd_hist = data['macd_hist']
        bb_upper = data['bb_upper']
        bb_middle = data['bb_middle']
        bb_lower = data['bb_lower']
        atr = data['atr']
        sma_20 = data['sma_20']
        sma_50 = data['sma_50']
        sma_200 = data['sma_200']
        volatility = data['volatility']
        foreign_net = data['foreign_net']
        institution_net = data['institution_net']
        individual_net = data['individual_net']
        foreign_own = data['foreign_own']
        short_ratio = data['short_ratio']
        margin_ratio = data['margin_ratio']
        
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

def genetic_programming_parallel(seed_alphas, data, generations=15, population_size=200):
    """병렬 GP"""
    num_workers = min(cpu_count(), 8)
    
    print(f"\n🧬 병렬 GP 진화")
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
    print("종합 알파 생성: 모든 지표 + 논문 방식 프롬프트")
    print("=" * 70)
    print()
    
    # 데이터 로드
    data = load_comprehensive_data()
    
    # LLM 알파 생성
    seed_alphas = generate_comprehensive_alphas()
    
    print("\n📊 생성된 초기 알파:")
    for i, alpha in enumerate(seed_alphas, 1):
        print(f"   {i}. {alpha[:80]}...")
    
    # GP 진화 (Large-scale: 50 seeds → 200 population)
    evolved_alphas = genetic_programming_parallel(
        seed_alphas=seed_alphas,
        data=data,
        generations=15,
        population_size=200
    )
    
    # 결과
    print("\n" + "=" * 70)
    print("🏆 진화된 상위 5개 종합 알파")
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
            f"IC: {best_ic:.4f}, Comprehensive (All indicators), 10-day forward"
        ))
        conn.commit()
        print("\n✅ DB 저장 완료")
    finally:
        cur.close()
        conn.close()
    
    print("\n🎉 종합 알파 생성 완료!")

if __name__ == "__main__":
    main()
