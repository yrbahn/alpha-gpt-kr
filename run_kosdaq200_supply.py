#!/usr/bin/env python3
"""
KOSDAQ 200 + 수급 데이터 알파 실험
외국인/기관 수급 데이터를 추가하여 IC 향상 테스트
"""

import sys
# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import psycopg2

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.data.postgres_loader import PostgresDataLoader
from alpha_gpt_kr.mining.operators import AlphaOperators as ops
from alpha_gpt_kr.backtest.engine import BacktestEngine

load_dotenv()

# ============================================================
# 설정
# ============================================================
UNIVERSE_SIZE = 200
MARKET = 'KOSDAQ'  # index_membership 필터
FORWARD_DAYS = 20  # 월간 리밸런싱
MAX_LOOKBACK = 120
PURGE_GAP = 20
N_FOLDS = 4
POP_SIZE = 150
GENERATIONS = 70
ELITE_SIZE = 10
TOURNAMENT_SIZE = 5

# 제외 종목
EXCLUDE_TICKERS = ['042700', '005690', '058470']

# ============================================================
# 기본 피처 (수급 추가!)
# ============================================================
BASE_FEATURES = ['close', 'open', 'high', 'low', 'volume']
SUPPLY_FEATURES = ['foreign_net', 'inst_net', 'foreign_net_ratio']  # 수급!

# ============================================================
# 파생 피처
# ============================================================
def compute_derived_features(data):
    """파생 피처 계산 (수급 포함)"""
    close = data['close']
    open_ = data['open']
    high = data['high']
    low = data['low']
    volume = data['volume']
    
    derived = {}
    
    # 기존 피처
    derived['returns'] = close.pct_change()
    derived['log_returns'] = np.log(close / close.shift(1))
    derived['gap'] = (open_ - close.shift(1)) / close.shift(1)
    derived['intraday_return'] = (close - open_) / open_
    derived['high_low_range'] = (high - low) / close
    derived['upper_shadow'] = (high - np.maximum(open_, close)) / close
    derived['lower_shadow'] = (np.minimum(open_, close) - low) / close
    derived['body_ratio'] = abs(close - open_) / (high - low + 1e-8)
    
    # 거래량 관련
    vol_ma20 = volume.rolling(20).mean()
    derived['vol_ratio'] = volume / vol_ma20
    derived['vol_change'] = volume.pct_change()
    
    # 변동성
    derived['volatility_20'] = derived['returns'].rolling(20).std()
    derived['atr'] = (high - low).rolling(14).mean()
    derived['atr_ratio'] = derived['atr'] / close
    
    # 수급 피처 (핵심!)
    if 'foreign_net' in data:
        foreign_net = data['foreign_net']
        derived['foreign_net'] = foreign_net
        derived['foreign_net_ratio'] = foreign_net / (volume * close + 1e-8)
        derived['foreign_net_ma5'] = foreign_net.rolling(5).mean()
        derived['foreign_net_ma20'] = foreign_net.rolling(20).mean()
        derived['foreign_momentum'] = derived['foreign_net_ma5'] / (derived['foreign_net_ma20'] + 1e-8)
        print("  ✅ 외국인 수급 피처 추가!", flush=True)
    
    if 'inst_net' in data:
        inst_net = data['inst_net']
        derived['inst_net'] = inst_net
        derived['inst_net_ratio'] = inst_net / (volume * close + 1e-8)
        derived['inst_net_ma5'] = inst_net.rolling(5).mean()
        derived['inst_momentum'] = derived['inst_net_ma5'] / (inst_net.rolling(20).mean() + 1e-8)
        print("  ✅ 기관 수급 피처 추가!", flush=True)
    
    if 'foreign_ownership' in data:
        derived['foreign_ownership'] = data['foreign_ownership']
        derived['foreign_own_change'] = data['foreign_ownership'].diff()
        print("  ✅ 외국인 지분율 피처 추가!", flush=True)
    
    return derived


def get_kosdaq_top200():
    """KOSDAQ 시가총액 상위 200개"""
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )
    
    query = f"""
        SELECT ticker, name, market_cap
        FROM stocks
        WHERE is_active = true
          AND index_membership = '{MARKET}'
          AND market_cap IS NOT NULL
          AND ticker NOT IN ({','.join([f"'{t}'" for t in EXCLUDE_TICKERS])})
        ORDER BY market_cap DESC
        LIMIT {UNIVERSE_SIZE}
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    return df['ticker'].tolist()


def calculate_forward_returns(close, days=20):
    """N일 후 수익률"""
    return close.shift(-days) / close - 1


def evaluate_alpha(alpha_values, forward_returns):
    """알파 IC 계산"""
    ics = []
    for date in alpha_values.index:
        if date not in forward_returns.index:
            continue
        alpha_row = alpha_values.loc[date].dropna()
        ret_row = forward_returns.loc[date].dropna()
        common = alpha_row.index.intersection(ret_row.index)
        if len(common) < 30:
            continue
        ic = alpha_row[common].corr(ret_row[common], method='spearman')
        if not np.isnan(ic):
            ics.append(ic)
    return np.mean(ics) if ics else -1.0


def cross_validate(alpha_expr, data, forward_returns, n_folds=4):
    """시계열 CV"""
    dates = data['close'].index
    fold_size = len(dates) // n_folds
    
    results = []
    for i in range(n_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_folds - 1 else len(dates)
        
        train_dates = list(dates[:max(0, test_start - PURGE_GAP)]) + list(dates[test_end:])
        test_dates = dates[test_start:test_end]
        
        if len(train_dates) < 60 or len(test_dates) < 20:
            continue
        
        # 알파 계산
        try:
            alpha_values = eval(alpha_expr)
        except:
            return None
        
        train_ic = evaluate_alpha(alpha_values.loc[train_dates], forward_returns.loc[train_dates])
        test_ic = evaluate_alpha(alpha_values.loc[test_dates], forward_returns.loc[test_dates])
        
        results.append({
            'fold': i + 1,
            'train_ic': train_ic,
            'test_ic': test_ic
        })
    
    return results


# ============================================================
# GP 관련
# ============================================================
import random
from typing import List

class Individual:
    def __init__(self, expr: str, fitness: float = -np.inf):
        self.expression = expr
        self.fitness = fitness
        self.cv_results = None

def generate_random_alpha(features: List[str]) -> str:
    """랜덤 알파 생성"""
    ts_ops = ['ts_mean', 'ts_std', 'ts_rank', 'ts_delta', 'ts_sum', 
              'ts_min', 'ts_max', 'ts_corr', 'ts_ema', 'ts_decayed_linear']
    cs_ops = ['normed_rank', 'zscore_scale', 'neg', 'abs_val', 'sign']
    combine_ops = ['add', 'minus', 'cwise_mul', 'div']
    
    # 단순 구조
    f1, f2 = random.sample(features, 2)
    window = random.choice([5, 10, 15, 20, 30, 40, 60])
    
    templates = [
        f"ops.normed_rank(ops.ts_delta({f1}, {window}))",
        f"ops.normed_rank(ops.ts_mean({f1}, {window}))",
        f"ops.normed_rank(ops.ts_std({f1}, {window}))",
        f"ops.normed_rank(ops.ts_corr({f1}, {f2}, {window}))",
        f"ops.zscore_scale(ops.ts_rank({f1}, {window}))",
        f"ops.normed_rank(ops.minus(ops.ts_rank({f1}, {window}), ops.ts_rank({f2}, {window})))",
        f"ops.normed_rank(ops.ts_regression_residual({f1}, {f2}, {window}))",
        # 수급 특화
        f"ops.normed_rank(ops.ts_sum(foreign_net_ratio, {window}))",
        f"ops.normed_rank(ops.ts_corr(returns, foreign_net_ratio, {window}))",
        f"ops.zscore_scale(ops.ts_delta(foreign_net_ma5, {window}))",
    ]
    
    return random.choice(templates)


def mutate(ind: Individual, features: List[str]) -> Individual:
    """변이"""
    expr = ind.expression
    
    mutation_type = random.choice(['window', 'operator', 'feature', 'structure'])
    
    if mutation_type == 'window':
        import re
        windows = re.findall(r', (\d+)\)', expr)
        if windows:
            old_w = random.choice(windows)
            new_w = str(random.choice([5, 10, 15, 20, 30, 40, 60, 90, 120]))
            expr = expr.replace(f', {old_w})', f', {new_w})', 1)
    
    elif mutation_type == 'operator':
        ts_ops = ['ts_mean', 'ts_std', 'ts_rank', 'ts_delta', 'ts_sum', 'ts_ema']
        for op in ts_ops:
            if op in expr:
                new_op = random.choice(ts_ops)
                expr = expr.replace(op, new_op, 1)
                break
    
    elif mutation_type == 'feature':
        for f in features:
            if f in expr and random.random() < 0.3:
                new_f = random.choice(features)
                expr = expr.replace(f, new_f, 1)
                break
    
    else:  # structure
        expr = generate_random_alpha(features)
    
    return Individual(expr)


def crossover(ind1: Individual, ind2: Individual) -> Individual:
    """교배"""
    # 간단한 부분 교환
    parts1 = ind1.expression.split('ops.')
    parts2 = ind2.expression.split('ops.')
    
    if len(parts1) > 2 and len(parts2) > 2:
        idx = random.randint(1, min(len(parts1), len(parts2)) - 1)
        new_parts = parts1[:idx] + parts2[idx:]
        new_expr = 'ops.'.join(new_parts)
        return Individual(new_expr)
    
    return Individual(ind1.expression if random.random() < 0.5 else ind2.expression)


def run_gp(data, forward_returns, features, pop_size=100, generations=50):
    """GP 진화 실행"""
    print(f"\n{'='*60}", flush=True)
    print(f"🧬 GP 진화 시작: pop={pop_size}, gen={generations}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # 초기 개체군
    population = []
    
    # 수급 특화 시드
    supply_seeds = [
        "ops.normed_rank(ops.ts_sum(foreign_net_ratio, 20))",
        "ops.normed_rank(ops.ts_corr(returns, foreign_net_ratio, 20))",
        "ops.zscore_scale(ops.ts_delta(foreign_net_ma5, 10))",
        "ops.normed_rank(ops.neg(ops.ts_corr(returns, vol_ratio, 20)))",
        "ops.normed_rank(ops.ts_regression_residual(returns, foreign_net_ratio, 30))",
        "ops.normed_rank(ops.add(ops.ts_rank(foreign_momentum, 20), ops.ts_rank(returns, 20)))",
    ]
    
    for seed in supply_seeds:
        population.append(Individual(seed))
    
    # 랜덤 생성으로 채우기
    while len(population) < pop_size:
        population.append(Individual(generate_random_alpha(features)))
    
    best_ever = None
    best_ic = -np.inf
    stagnation = 0
    
    for gen in range(generations):
        # 평가
        for ind in population:
            if ind.fitness == -np.inf:
                try:
                    alpha_values = eval(ind.expression)
                    ind.fitness = evaluate_alpha(alpha_values, forward_returns)
                except:
                    ind.fitness = -1.0
        
        # 정렬
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        current_best = population[0]
        median_fit = np.median([p.fitness for p in population[:pop_size//2]])
        
        if current_best.fitness > best_ic:
            best_ic = current_best.fitness
            best_ever = current_best
            stagnation = 0
            print(f"  세대 {gen+1:2d}/{generations}: 🏆 최고 IC={best_ic:.4f} | 중앙값={median_fit:.4f}", flush=True)
        else:
            stagnation += 1
            if gen % 10 == 0:
                print(f"  세대 {gen+1:2d}/{generations}: 최고 IC={best_ic:.4f} | 중앙값={median_fit:.4f} | 정체={stagnation}", flush=True)
        
        # 이민 (정체 시)
        if stagnation >= 10 and stagnation % 10 == 0:
            print(f"    🌍 이민: 50개 새 개체 주입", flush=True)
            for _ in range(50):
                population.append(Individual(generate_random_alpha(features)))
        
        # 다음 세대
        next_gen = population[:ELITE_SIZE]  # 엘리트
        
        while len(next_gen) < pop_size:
            if random.random() < 0.7:  # 교배
                p1 = random.choice(population[:pop_size//3])
                p2 = random.choice(population[:pop_size//3])
                child = crossover(p1, p2)
            else:  # 변이
                parent = random.choice(population[:pop_size//2])
                child = mutate(parent, features)
            next_gen.append(child)
        
        population = next_gen[:pop_size]
    
    return best_ever, population[:20]


def main():
    print("="*70, flush=True)
    print("🚀 KOSDAQ 200 + 수급 데이터 알파 실험", flush=True)
    print("="*70, flush=True)
    start_time = datetime.now()
    
    # 1. 유니버스 로드
    print("\n📊 1. KOSDAQ 200 종목 로드...", flush=True)
    tickers = get_kosdaq_top200()
    print(f"  ✅ {len(tickers)}개 종목", flush=True)
    
    # 2. 데이터 로드 (수급 포함!)
    print("\n📊 2. 데이터 로드 (수급 포함)...", flush=True)
    loader = PostgresDataLoader()
    
    data = loader.load_data(
        universe=tickers,
        start_date="2023-01-01",
        end_date="2026-02-14",
        include_technical=False,
        include_supply_demand=True  # 🔥 수급 데이터!
    )
    
    print(f"  기간: {data['close'].index[0].date()} ~ {data['close'].index[-1].date()}", flush=True)
    print(f"  일수: {len(data['close'])}", flush=True)
    print(f"  필드: {list(data.keys())}", flush=True)
    
    # 3. 파생 피처 계산
    print("\n📊 3. 파생 피처 계산...", flush=True)
    derived = compute_derived_features(data)
    
    # 전역 변수로 등록
    for name, df in data.items():
        globals()[name] = df
    for name, df in derived.items():
        globals()[name] = df
    
    all_features = list(data.keys()) + list(derived.keys())
    print(f"  총 피처: {len(all_features)}개", flush=True)
    print(f"  수급 피처: {[f for f in all_features if 'foreign' in f or 'inst' in f]}", flush=True)
    
    # 4. Forward returns
    print("\n📊 4. Forward returns 계산...", flush=True)
    forward_returns = calculate_forward_returns(data['close'], FORWARD_DAYS)
    
    # 5. GP 진화
    print("\n📊 5. GP 진화...", flush=True)
    best_alpha, top_alphas = run_gp(
        data, forward_returns, all_features,
        pop_size=POP_SIZE, generations=GENERATIONS
    )
    
    # 6. CV 검증
    print("\n" + "="*60, flush=True)
    print("📊 Cross-Validation (Top 20)", flush=True)
    print("="*60, flush=True)
    
    cv_results = []
    for i, ind in enumerate(top_alphas):
        try:
            cv = cross_validate(ind.expression, data, forward_returns, N_FOLDS)
            if cv:
                test_ics = [r['test_ic'] for r in cv]
                mean_test = np.mean(test_ics)
                test_ir = mean_test / (np.std(test_ics) + 1e-8)
                pos_folds = sum(1 for ic in test_ics if ic > 0)
                
                cv_results.append({
                    'rank': i + 1,
                    'expr': ind.expression,
                    'train_ic': np.mean([r['train_ic'] for r in cv]),
                    'test_ic': mean_test,
                    'test_ir': test_ir,
                    'pos_folds': pos_folds,
                    'fold_ics': test_ics
                })
        except:
            pass
    
    # 정렬
    cv_results.sort(key=lambda x: x['test_ic'], reverse=True)
    
    print(f"\n{'#':>3} {'Train IC':>10} {'Test IC':>10} {'IR':>8} {'Folds':>6}", flush=True)
    print("-" * 50, flush=True)
    for r in cv_results[:10]:
        folds_str = f"{r['pos_folds']}/4"
        print(f"{r['rank']:>3} {r['train_ic']:>10.4f} {r['test_ic']:>10.4f} {r['test_ir']:>8.2f} {folds_str:>6}", flush=True)
    
    # 7. 최종 결과
    print("\n" + "="*60, flush=True)
    print("🏆 FINAL BEST (CV-validated)", flush=True)
    print("="*60, flush=True)
    
    if cv_results:
        best = cv_results[0]
        print(f"Test IC:  {best['test_ic']:.4f}", flush=True)
        print(f"Test IR:  {best['test_ir']:.2f}", flush=True)
        print(f"Folds:    {best['pos_folds']}/4 {[f'{ic:+.3f}' for ic in best['fold_ics']]}", flush=True)
        print(f"Expression: {best['expr'][:100]}...", flush=True)
        
        # JSON 저장
        with open('best_alpha_supply.json', 'w') as f:
            json.dump({
                'test_ic': best['test_ic'],
                'test_ir': best['test_ir'],
                'pos_folds': best['pos_folds'],
                'expression': best['expr'],
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        print("\n💾 best_alpha_supply.json 저장 완료", flush=True)
    
    elapsed = datetime.now() - start_time
    print(f"\n⏱️  소요 시간: {elapsed}", flush=True)
    print("\n🎉 완료!", flush=True)


if __name__ == "__main__":
    main()
