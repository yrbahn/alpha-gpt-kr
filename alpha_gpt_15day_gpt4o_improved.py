#!/usr/bin/env python3
"""
Alpha-GPT: 15-day Forward with GPT-4o (v2 — Improved Prompt)
개선된 QuantDeveloper 프롬프트 + ops.xxx() 문법 + 병렬 GP
"""

import sys
import os
import re
import json
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

from alpha_gpt_kr.mining.operators import AlphaOperators as ops
from alpha_gpt_kr.agents.quant_developer import QuantDeveloper

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
    """500개 종목 데이터 로드 (시가총액 상위)"""
    print("📊 데이터 로드 중... (시총 상위 500종목, 2년)")
    
    conn = get_db_connection()
    
    # 시가총액 상위 500개
    query_stocks = """
        SELECT 
            s.id,
            s.ticker,
            s.name,
            s.market_cap
        FROM stocks s
        WHERE s.is_active = true
        AND s.market_cap IS NOT NULL
        AND EXISTS (
            SELECT 1 FROM price_data p 
            WHERE p.stock_id = s.id 
            AND p.date >= CURRENT_DATE - INTERVAL '730 days'
            LIMIT 1
        )
        ORDER BY s.market_cap DESC
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

def generate_seed_alphas_gpt4o(num_seeds=20):
    """GPT-4o + 개선된 QuantDeveloper 프롬프트로 시드 알파 생성"""
    print(f"\n🤖 GPT-4o로 초기 알파 {num_seeds}개 생성 중 (개선된 프롬프트)...")

    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # QuantDeveloper의 개선된 프롬프트 재사용
    system_prompt = QuantDeveloper.SYSTEM_PROMPT

    # 15일 보유에 특화된 user prompt
    prompt = f"""### Task
Generate {num_seeds} diverse, high-performance alpha expressions optimized for **15-day forward returns** in the Korean stock market.

### Trading Idea
15일 보유 전략에 최적화된 중기 알파 팩터. 단기 노이즈를 필터링하고,
15일 후 수익률과 높은 상관관계(IC)를 가지는 시그널을 찾아야 함.
모멘텀, 거래량, 변동성, 추세 강도를 조합하여 다양한 팩터를 생성.

### Available Data Fields
close, volume, returns

### Requirements

**Diversity** — Each alpha MUST belong to a DIFFERENT category:
  1. `momentum_volume` — Momentum confirmed by volume surge
  2. `volatility_adjusted` — Signal adjusted/filtered by volatility
  3. `short_term_reversal` — Mean-reversion exploiting KRX reversal effect
  4. `multi_timeframe` — Combining short + medium + long timeframes
  5. `price_volume_diverge` — Price-volume divergence / smart money
  6. `trend_strength` — Trend strength via regression slope or IR
  7. `tail_risk` — Skewness/kurtosis-based risk signal
  8. `price_position` — Price position relative to recent high/low
  9. `volume_anomaly` — Abnormal volume detection
  10. `composite` — 3+ factor composite signal
  11. `momentum_volume` — Variation with different timeframes
  12. `volatility_adjusted` — Variation with different approach
  13. `short_term_reversal` — Variation with volume filter
  14. `multi_timeframe` — Variation with volatility
  15. `price_volume_diverge` — Variation with trend
  16. `trend_strength` — Variation with volume
  17. `composite` — Different 3+ factor combination
  18. `price_position` — Variation with momentum
  19. `volume_anomaly` — Variation with reversal
  20. `composite` — Most complex combination

**15-Day Holding Optimization**:
- Prefer medium-term lookback windows: 10, 15, 20, 30 days (not too short like 3d, not too long like 60d)
- Combine at least 2 timeframes per alpha
- Volume confirmation is critical for 15-day predictions

**Quality Checklist** — Every alpha must satisfy ALL:
- [ ] Multi-factor: combines 2+ distinct signal types
- [ ] Market-neutral: wrapped with `ops.normed_rank()` or `ops.zscore_scale()`
- [ ] Multi-timeframe: uses 2+ lookback windows
- [ ] No look-ahead bias
- [ ] Complexity 2~4 nesting levels
- [ ] Safe division: use `ops.div()` instead of raw `/`

### Output Format
Return a JSON array:
```json
[
  {{
    "alpha_name": "Alpha_Name",
    "category": "category_name",
    "rationale": "Economic logic explanation",
    "expression": "ops.normed_rank(...)",
    "complexity": 4,
    "operators_used": ["op1", "op2"],
    "timeframes_used": [10, 20]
  }}
]
```

**CRITICAL**:
- You MUST return a JSON object with key "alphas" containing an array of {num_seeds} alpha objects.
- Format: {{"alphas": [{{...}}, {{...}}, ...]}}
- Each object MUST have "expression" field with valid ops.xxx() Python code.
- Generate ALL {num_seeds} alphas. Do NOT return just 1."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=16000,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content
    print(f"   GPT-4o 응답 길이: {len(content)}자")
    print(f"   응답 미리보기: {content[:200]}...")

    # JSON 파싱
    alphas = []
    try:
        data = json.loads(content)
        print(f"   파싱된 타입: {type(data).__name__}")

        # dict → 리스트 추출
        if isinstance(data, dict):
            print(f"   키 목록: {list(data.keys())}")

            # 1순위: dict 자체가 단일 알파인 경우 (expression 키 존재)
            if 'expression' in data or 'expr' in data:
                data = [data]
                print(f"   단일 알파 dict → 리스트로 변환")

            else:
                # 2순위: {"alphas": [{...}, ...]} 형태 — dict 리스트를 가진 키 찾기
                found_list = False
                for key in data:
                    if isinstance(data[key], list) and data[key] and isinstance(data[key][0], dict):
                        data = data[key]
                        print(f"   '{key}' 키에서 {len(data)}개 항목 추출")
                        found_list = True
                        break

                if not found_list:
                    # 3순위: 중첩 dict: {"alpha_1": {...}, "alpha_2": {...}}
                    items = []
                    for key, val in data.items():
                        if isinstance(val, dict) and ('expression' in val or 'expr' in val):
                            items.append(val)
                    if items:
                        data = items
                        print(f"   중첩 dict에서 {len(items)}개 항목 추출")
                    else:
                        print(f"   ⚠️  알 수 없는 dict 구조: {list(data.keys())[:5]}")
                        data = []

        for item in data:
            if isinstance(item, str):
                if 'ops.' in item:
                    alphas.append(item)
                continue
            if not isinstance(item, dict):
                continue
            expr = item.get('expression', item.get('expr', ''))
            if expr and 'ops.' in expr:
                alphas.append(expr)
            elif expr:
                print(f"   ⚠️  ops. 없는 표현식 스킵: {expr[:80]}")

    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠️  JSON 파싱 실패: {e}")
        # 마크다운 코드블록 안의 JSON 추출 시도
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                for item in data:
                    if isinstance(item, dict):
                        expr = item.get('expression', item.get('expr', ''))
                        if expr and 'ops.' in expr:
                            alphas.append(expr)
                print(f"   마크다운 블록에서 {len(alphas)}개 복구")
            except Exception:
                pass

    # 폴백: 개선된 복합 팩터
    if len(alphas) < 10:
        print(f"⚠️  {len(alphas)}개만 파싱됨, 폴백 추가")
        fallback = [
            "ops.normed_rank(ops.cwise_mul(ops.ts_delta_ratio(close, 15), ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20))))",
            "ops.normed_rank(ops.div(ops.neg(ops.ts_zscore_scale(close, 10)), ops.ts_std(returns, 20)))",
            "ops.normed_rank(ops.neg(ops.ts_corr(ops.ts_delta(close, 5), ops.ts_delta(volume, 5), 20)))",
            "ops.normed_rank(ops.minus(ops.ts_ir(returns, 5), ops.ts_ir(returns, 20)))",
            "ops.normed_rank(ops.cwise_mul(ops.ts_maxmin_scale(close, 20), ops.normed_rank(ops.ts_mean(volume, 5))))",
            "ops.normed_rank(ops.cwise_mul(ops.relu(ops.ts_linear_reg(close, 20)), ops.relu(ops.ts_skew(returns, 20))))",
            "ops.normed_rank(ops.cwise_mul(ops.cwise_mul(ops.greater(ops.ts_delta_ratio(volume, 5), 0.5), ops.less(ops.ts_delta_ratio(close, 5), 0)), ops.neg(ops.normed_rank(ops.ts_std(returns, 20)))))",
            "ops.normed_rank(ops.cwise_mul(ops.ts_delta_ratio(close, 10), ops.div(ops.ts_mean(volume, 10), ops.ts_mean(volume, 30))))",
            "ops.normed_rank(ops.minus(ops.ts_linear_reg(close, 10), ops.ts_linear_reg(close, 30)))",
            "ops.normed_rank(ops.div(ops.ts_max_diff(close, 20), ops.ts_std(close, 20)))",
            "ops.normed_rank(ops.cwise_mul(ops.ts_delta_ratio(close, 20), ops.neg(ops.ts_skew(returns, 15))))",
            "ops.normed_rank(ops.div(ops.ts_min_diff(close, 15), ops.ts_std(returns, 15)))",
        ]
        alphas = alphas + [f for f in fallback if f not in alphas]

    print(f"✅ {len(alphas)}개 초기 알파 생성")
    for i, a in enumerate(alphas[:5], 1):
        print(f"   {i}. {a[:80]}...")

    return alphas[:num_seeds]

# 전역 데이터
_global_data = None

def set_global_data(data):
    global _global_data
    _global_data = data

def _compute_raw_ic(alpha_expr, data):
    """알파의 raw IC 계산 (train 또는 test 데이터)"""
    close = data['close']
    volume = data['volume']
    returns = data['returns']

    forward_return_15d = close.shift(-15) / close - 1
    alpha_values = eval(alpha_expr)

    if not isinstance(alpha_values, pd.DataFrame):
        return -999.0

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

    return float(np.mean(ic_list))

def _multi_factor_bonus(alpha_expr):
    """다중 팩터 구조 보너스"""
    bonus = 0.0
    # 거래량 사용 보너스
    if 'volume' in alpha_expr:
        bonus += 0.003
    # 다중 타임프레임 보너스 (윈도우 차이 ≥ 2배)
    windows = [int(w) for w in re.findall(r',\s*(\d+)\)', alpha_expr)]
    if len(windows) >= 2:
        if max(windows) >= min(windows) * 2:
            bonus += 0.002
    # 복잡도 페널티
    depth = alpha_expr.count('(')
    if depth < 3:
        bonus -= 0.002
    if depth > 8:
        bonus -= 0.003
    return bonus

def evaluate_alpha_worker(alpha_expr):
    """병렬 처리용 알파 평가 — train IC + 다중팩터 보너스"""
    global _global_data
    data = _global_data

    try:
        raw_ic = _compute_raw_ic(alpha_expr, data)
        if raw_ic <= -999.0:
            return (alpha_expr, -999.0)
        bonus = _multi_factor_bonus(alpha_expr)
        return (alpha_expr, raw_ic + bonus)
    except Exception:
        return (alpha_expr, -999.0)

def evaluate_alpha_oos(alpha_expr, test_data):
    """Out-of-sample IC 계산 (보너스 없이 순수 IC)"""
    try:
        return _compute_raw_ic(alpha_expr, test_data)
    except Exception:
        return -999.0

# 연산자 교환 그룹 (같은 시그니처끼리만 교체)
OPERATOR_SWAP_GROUPS = [
    ['ts_mean', 'ts_std', 'ts_median', 'ts_ema', 'ts_linear_reg', 'ts_decayed_linear'],
    ['ts_zscore_scale', 'ts_maxmin_scale', 'ts_rank'],
    ['ts_delta', 'ts_delta_ratio'],
    ['ts_skew', 'ts_kurt', 'ts_ir'],
    ['ts_min', 'ts_max'],
    ['ts_argmin', 'ts_argmax'],
    ['ts_max_diff', 'ts_min_diff'],
    ['normed_rank', 'zscore_scale'],
    ['cwise_mul', 'add', 'minus'],
]

OPERAND_POOL = ['close', 'volume', 'returns']

def mutate_alpha(alpha_expr):
    """알파 변이 — 3가지 타입: 윈도우(50%), 연산자(30%), 피연산자(20%)"""
    try:
        mutation_type = random.choices(
            ['window', 'operator', 'operand'],
            weights=[0.5, 0.3, 0.2]
        )[0]

        if mutation_type == 'window':
            return _mutate_window(alpha_expr)
        elif mutation_type == 'operator':
            return _mutate_operator(alpha_expr)
        else:
            return _mutate_operand(alpha_expr)
    except Exception:
        return None

def _mutate_window(alpha_expr):
    """윈도우 파라미터 변경 (범위 5~50)"""
    matches = list(re.finditer(r'(ts_\w+|shift)\([^,]+,\s*(\d+)\)', alpha_expr))
    if not matches:
        return None
    match = random.choice(matches)
    old_window = int(match.group(2))
    new_window = max(5, min(50, old_window + random.choice([-7, -5, -3, -2, 2, 3, 5, 7, 10])))
    if new_window == old_window:
        new_window = max(5, old_window + random.choice([-10, 10]))
    start, end = match.span(2)
    return alpha_expr[:start] + str(new_window) + alpha_expr[end:]

def _mutate_operator(alpha_expr):
    """연산자 교체 — 같은 시그니처 그룹 내에서만"""
    # 현재 표현식에서 연산자 추출
    op_matches = list(re.finditer(r'ops\.(\w+)\(', alpha_expr))
    if not op_matches:
        return None

    # 교환 가능한 연산자만 필터
    swappable = []
    for m in op_matches:
        op_name = m.group(1)
        for group in OPERATOR_SWAP_GROUPS:
            if op_name in group:
                swappable.append((m, op_name, group))
                break

    if not swappable:
        return _mutate_window(alpha_expr)  # 교환 불가면 윈도우 변이

    match, old_op, group = random.choice(swappable)
    candidates = [op for op in group if op != old_op]
    if not candidates:
        return _mutate_window(alpha_expr)

    new_op = random.choice(candidates)
    start, end = match.span(1)
    return alpha_expr[:start] + new_op + alpha_expr[end:]

def _mutate_operand(alpha_expr):
    """피연산자 교체 — close/volume/returns 간 교환"""
    present = [op for op in OPERAND_POOL if op in alpha_expr]
    if not present:
        return _mutate_window(alpha_expr)

    old_operand = random.choice(present)
    candidates = [op for op in OPERAND_POOL if op != old_operand]
    new_operand = random.choice(candidates)

    # 첫 번째 등장만 교체 (전체 교체 방지)
    return alpha_expr.replace(old_operand, new_operand, 1)


def crossover_alphas(alpha1, alpha2):
    """알파 교차 — 두 알파의 윈도우 파라미터를 교환"""
    try:
        matches1 = list(re.finditer(r'(ts_\w+|shift)\(([^,]+),\s*(\d+)\)', alpha1))
        matches2 = list(re.finditer(r'(ts_\w+|shift)\(([^,]+),\s*(\d+)\)', alpha2))

        if not matches1 or not matches2:
            return None

        # 같은 연산자가 있으면 우선 교차
        ops1 = {m.group(1): m for m in matches1}
        ops2 = {m.group(1): m for m in matches2}
        common_ops = set(ops1.keys()) & set(ops2.keys())

        if common_ops:
            op = random.choice(list(common_ops))
            m1, m2 = ops1[op], ops2[op]
        else:
            m1 = random.choice(matches1)
            m2 = random.choice(matches2)

        # alpha1의 윈도우를 alpha2의 값으로 교체
        win2 = m2.group(3)
        start, end = m1.span(3)
        return alpha1[:start] + win2 + alpha1[end:]
    except Exception:
        return None

def _get_alpha_structure(alpha_expr):
    """알파의 구조 시그니처 (윈도우 제거) — 다양성 비교용"""
    return re.sub(r',\s*\d+\)', ', N)', alpha_expr)

def _select_diverse_top_n(results, n=5):
    """IC 상위에서 구조가 다른 Top-N 선택"""
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    selected = []
    seen_structures = set()

    for alpha, ic in sorted_results:
        if ic <= -999.0:
            continue
        structure = _get_alpha_structure(alpha)
        if structure not in seen_structures:
            selected.append((alpha, ic))
            seen_structures.add(structure)
            if len(selected) >= n:
                break

    return selected

def genetic_programming(seed_alphas, data, generations=40, population_size=150):
    """개선된 병렬 GP — 구조적 변이 + 다양성 보존 + 조기종료"""

    print(f"\n🧬 병렬 GP 시작 (개선됨)")
    print(f"   Seed: {len(seed_alphas)}개, 세대: {generations}, 개체수: {population_size}, 워커: 4")

    population = seed_alphas[:population_size]
    while len(population) < population_size:
        parent = random.choice(seed_alphas)
        mutated = mutate_alpha(parent)
        if mutated:
            population.append(mutated)

    set_global_data(data)
    best_ever = (None, -999.0)
    stagnation_count = 0
    all_results_history = []  # 모든 세대의 결과 보관

    elite_count = max(5, population_size // 10)  # 10% 엘리트
    parent_pool_size = 30

    for gen in range(1, generations + 1):
        print(f"\n  세대 {gen}/{generations}")

        with Pool(4, initializer=set_global_data, initargs=(data,)) as pool:
            results = pool.map(evaluate_alpha_worker, population)

        fitness_scores = sorted(results, key=lambda x: x[1], reverse=True)
        all_results_history.extend([(a, ic) for a, ic in fitness_scores if ic > -999.0])

        best_ic = fitness_scores[0][1]
        print(f"    최고 IC: {best_ic:.4f}")

        if best_ic > best_ever[1]:
            best_ever = fitness_scores[0]
            stagnation_count = 0
            print(f"    🏆 신기록!")
        else:
            stagnation_count += 1

        # 조기종료: 5세대 연속 무개선
        if stagnation_count >= 5:
            print(f"    ⏹️  5세대 무개선 → 조기종료")
            break

        # 다음 세대 구성
        next_population = []

        # 엘리트 보존 (10%)
        for alpha, _ in fitness_scores[:elite_count]:
            next_population.append(alpha)

        # 나머지: 교차(60%) + 변이(40%)
        parent_pool = [a for a, ic in fitness_scores[:parent_pool_size]]

        while len(next_population) < population_size:
            if random.random() < 0.6:
                # 교차
                parent1 = random.choice(parent_pool)
                parent2 = random.choice(parent_pool)
                child = crossover_alphas(parent1, parent2)
                if child:
                    next_population.append(child)
                else:
                    next_population.append(parent1)
            else:
                # 변이 (구조적 변이 포함)
                parent = random.choice(parent_pool)
                mutated = mutate_alpha(parent)
                if mutated:
                    next_population.append(mutated)
                else:
                    next_population.append(parent)

        population = next_population[:population_size]

        del results, fitness_scores, next_population
        gc.collect()

    # Top-5 다양한 알파 선택
    top_diverse = _select_diverse_top_n(all_results_history, n=5)

    return best_ever, top_diverse

def main():
    print("=" * 80)
    print("Alpha-GPT: 15-day Forward with GPT-4o (v3 — Enhanced GP)")
    print("=" * 80)
    print()

    # 1. 전체 데이터 로드
    full_data = load_market_data()

    # 2. Train/Test 분할 (70/30)
    close = full_data['close']
    split_idx = int(len(close) * 0.7)
    split_date = close.index[split_idx]
    print(f"\n📐 Train/Test 분할: {split_idx}일 train / {len(close) - split_idx}일 test")
    print(f"   Train: ~{close.index[0]} ~ {close.index[split_idx-1]}")
    print(f"   Test:  ~{split_date} ~ {close.index[-1]}")

    train_data = {
        'close': full_data['close'].iloc[:split_idx],
        'volume': full_data['volume'].iloc[:split_idx],
        'returns': full_data['returns'].iloc[:split_idx],
    }
    test_data = {
        'close': full_data['close'].iloc[split_idx:],
        'volume': full_data['volume'].iloc[split_idx:],
        'returns': full_data['returns'].iloc[split_idx:],
    }

    # 3. GPT-4o 시드 생성
    seed_alphas = generate_seed_alphas_gpt4o()

    # 4. GP 진화 (train 데이터로)
    (best_alpha, best_ic), top_diverse = genetic_programming(
        seed_alphas,
        train_data,
        generations=40,
        population_size=150
    )

    # 5. Top-5 OOS 검증
    print("\n" + "=" * 80)
    print("🏆 TOP 5 ALPHAS (Train IC + Test IC)")
    print("=" * 80)

    validated_alphas = []
    for i, (alpha, train_ic_with_bonus) in enumerate(top_diverse, 1):
        # 순수 train IC (보너스 제거)
        train_ic = _compute_raw_ic(alpha, train_data)
        # OOS test IC
        test_ic = evaluate_alpha_oos(alpha, test_data)
        # 팩터 분류
        factors = []
        if any(kw in alpha for kw in ['close', 'open_price', 'high', 'low']):
            factors.append('price')
        if 'volume' in alpha:
            factors.append('volume')
        if 'returns' in alpha:
            factors.append('returns')
        factor_str = '+'.join(factors) if factors else 'unknown'

        status = "✅" if test_ic > 0.015 else "⚠️"
        print(f"\n  #{i} {status}")
        print(f"     Train IC: {train_ic:.4f}  |  Test IC: {test_ic:.4f}  [{factor_str}]")
        print(f"     {alpha[:100]}{'...' if len(alpha) > 100 else ''}")

        validated_alphas.append({
            'expr': alpha,
            'train_ic': train_ic,
            'test_ic': test_ic,
            'factors': factor_str,
        })

    # 6. 최종 Best 선정 (test IC가 양수인 것 중 train IC 최고)
    valid_alphas = [a for a in validated_alphas if a['test_ic'] > 0]
    if valid_alphas:
        final_best = max(valid_alphas, key=lambda x: x['train_ic'])
    else:
        final_best = validated_alphas[0] if validated_alphas else {'expr': best_alpha, 'train_ic': best_ic, 'test_ic': -999, 'factors': '?'}

    print("\n" + "=" * 80)
    print("🥇 FINAL BEST (OOS-validated)")
    print("=" * 80)
    print(f"Train IC: {final_best['train_ic']:.4f}")
    print(f"Test IC:  {final_best['test_ic']:.4f}")
    print(f"Factors:  {final_best['factors']}")
    print(f"Expression: {final_best['expr']}")

    # 7. DB 저장 (Top-5 전부)
    print("\n💾 데이터베이스에 저장 중...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        for a in validated_alphas:
            cursor.execute("""
                INSERT INTO alpha_formulas (formula, ic_score, description, created_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (formula) DO UPDATE
                SET ic_score = EXCLUDED.ic_score, updated_at = NOW()
            """, (
                a['expr'],
                float(a['train_ic']),
                f"15d fwd, train IC={a['train_ic']:.4f}, test IC={a['test_ic']:.4f}, factors={a['factors']}, v3-enhanced"
            ))

        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ {len(validated_alphas)}개 알파 저장 완료!")
    except Exception as e:
        print(f"⚠️  DB 저장 실패: {e}")

    print("\n🎉 완료!")

if __name__ == "__main__":
    main()
