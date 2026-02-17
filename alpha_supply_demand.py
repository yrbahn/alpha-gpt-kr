#!/usr/bin/env python3
"""
수급 중심 알파 생성기
외국인/기관/개인 순매수 + 공매도 지표를 반드시 포함하는 알파 탐색
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2
from multiprocessing import Pool
import random

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.mining.operators import AlphaOperators as ops

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
    """수급 데이터 포함 로드"""
    print("📊 데이터 로드 중... (수급 중심)")
    
    conn = get_db_connection()
    
    # 시가총액 상위 500개
    stocks_df = pd.read_sql("""
        SELECT s.id, s.ticker, s.name, s.market_cap
        FROM stocks s
        WHERE s.is_active = true AND s.market_cap IS NOT NULL
        AND EXISTS (SELECT 1 FROM price_data p WHERE p.stock_id = s.id 
                    AND p.date >= CURRENT_DATE - INTERVAL '730 days' LIMIT 1)
        ORDER BY s.market_cap DESC
        LIMIT 500
    """, conn)
    
    stock_ids = stocks_df['id'].tolist()
    stock_id_list = ', '.join(map(str, stock_ids))
    
    # 가격 데이터
    price_df = pd.read_sql(f"""
        SELECT s.ticker, p.date, p.close, p.volume
        FROM price_data p
        JOIN stocks s ON p.stock_id = s.id
        WHERE p.stock_id IN ({stock_id_list})
        AND p.date >= CURRENT_DATE - INTERVAL '730 days'
        ORDER BY s.ticker, p.date
    """, conn)
    
    close = price_df.pivot(index='date', columns='ticker', values='close')
    volume = price_df.pivot(index='date', columns='ticker', values='volume')
    returns = close.pct_change()
    
    # 수급 데이터
    flow_df = pd.read_sql(f"""
        SELECT s.ticker, sd.date,
               sd.foreign_net_buy, sd.institution_net_buy,
               sd.individual_net_buy, sd.foreign_ownership,
               sd.short_volume
        FROM supply_demand_data sd
        JOIN stocks s ON sd.stock_id = s.id
        WHERE sd.stock_id IN ({stock_id_list})
        AND sd.date >= CURRENT_DATE - INTERVAL '730 days'
        ORDER BY s.ticker, sd.date
    """, conn)
    
    conn.close()
    
    # 수급 피벗
    foreign_net = flow_df.pivot(index='date', columns='ticker', values='foreign_net_buy')
    inst_net = flow_df.pivot(index='date', columns='ticker', values='institution_net_buy')
    indiv_net = flow_df.pivot(index='date', columns='ticker', values='individual_net_buy')
    foreign_own = flow_df.pivot(index='date', columns='ticker', values='foreign_ownership')
    short_vol = flow_df.pivot(index='date', columns='ticker', values='short_volume')
    
    # 인덱스 맞추기
    common_idx = close.index.intersection(foreign_net.index)
    common_cols = close.columns.intersection(foreign_net.columns)
    
    close = close.loc[common_idx, common_cols]
    volume = volume.loc[common_idx, common_cols]
    returns = returns.loc[common_idx, common_cols]
    foreign_net = foreign_net.loc[common_idx, common_cols]
    inst_net = inst_net.loc[common_idx, common_cols]
    indiv_net = indiv_net.loc[common_idx, common_cols]
    foreign_own = foreign_own.reindex(index=common_idx, columns=common_cols)
    short_vol = short_vol.reindex(index=common_idx, columns=common_cols)
    
    # 수급 비율 계산
    safe_volume = volume.replace(0, np.nan)
    foreign_net_ratio = (foreign_net / safe_volume).clip(-1, 1).fillna(0)
    inst_net_ratio = (inst_net / safe_volume).clip(-1, 1).fillna(0)
    indiv_net_ratio = (indiv_net / safe_volume).clip(-1, 1).fillna(0)
    foreign_ownership_pct = (foreign_own / 100).clip(0, 1).fillna(0)
    short_ratio = (short_vol / safe_volume).clip(0, 1).fillna(0)
    
    print(f"✅ {len(close.columns)}개 종목, {len(close)}일 데이터")
    print(f"   수급 변수: foreign_net_ratio, inst_net_ratio, indiv_net_ratio, foreign_ownership_pct, short_ratio")
    
    return {
        'close': close,
        'volume': volume,
        'returns': returns,
        'foreign_net_ratio': foreign_net_ratio,
        'inst_net_ratio': inst_net_ratio,
        'indiv_net_ratio': indiv_net_ratio,
        'foreign_ownership_pct': foreign_ownership_pct,
        'short_ratio': short_ratio,
    }

# 수급 중심 시드 알파
SUPPLY_DEMAND_SEEDS = [
    # 외국인 누적 매수
    "ops.normed_rank(ops.ts_sum(foreign_net_ratio, 20))",
    "ops.normed_rank(ops.ts_sum(foreign_net_ratio, 10))",
    "ops.normed_rank(ops.ts_mean(foreign_net_ratio, 15))",
    
    # 기관 누적 매수
    "ops.normed_rank(ops.ts_sum(inst_net_ratio, 20))",
    "ops.normed_rank(ops.ts_mean(inst_net_ratio, 15))",
    
    # 외국인 + 기관 복합
    "ops.normed_rank(ops.add(ops.ts_sum(foreign_net_ratio, 15), ops.ts_sum(inst_net_ratio, 15)))",
    
    # 개인 역매매 (개인 매도 = 기관/외국인 매수)
    "ops.normed_rank(ops.neg(ops.ts_sum(indiv_net_ratio, 20)))",
    
    # 외국인 보유비율 변화
    "ops.normed_rank(ops.ts_delta(foreign_ownership_pct, 20))",
    "ops.normed_rank(ops.ts_delta(foreign_ownership_pct, 10))",
    
    # 공매도 역전략 (공매도 급증 후 반등)
    "ops.normed_rank(ops.neg(ops.ts_delta(short_ratio, 5)))",
    "ops.normed_rank(ops.neg(ops.ts_mean(short_ratio, 10)))",
    
    # 수급-가격 괴리 (외국인 매수 but 가격 하락 = 매집)
    "ops.normed_rank(ops.cwise_mul(ops.ts_sum(foreign_net_ratio, 10), ops.neg(ops.ts_delta(close, 10))))",
    
    # 수급 모멘텀 (외국인 가속화)
    "ops.normed_rank(ops.minus(ops.ts_mean(foreign_net_ratio, 5), ops.ts_mean(foreign_net_ratio, 20)))",
    
    # 기관-외국인 동조
    "ops.normed_rank(ops.ts_corr(foreign_net_ratio, inst_net_ratio, 20))",
    
    # 외국인 매수 + 보유비율 상승
    "ops.normed_rank(ops.cwise_mul(ops.ts_sum(foreign_net_ratio, 15), ops.ts_delta(foreign_ownership_pct, 15)))",
    
    # 수급 복합 (외국인 + 기관 - 개인)
    "ops.normed_rank(ops.add(ops.add(ops.ts_sum(foreign_net_ratio, 10), ops.ts_sum(inst_net_ratio, 10)), ops.neg(ops.ts_sum(indiv_net_ratio, 10))))",
    
    # 수급 강도 (외국인 매수 / 변동성)
    "ops.normed_rank(ops.div(ops.ts_sum(foreign_net_ratio, 10), ops.ts_std(returns, 20)))",
    
    # 공매도 청산 신호
    "ops.normed_rank(ops.cwise_mul(ops.neg(ops.ts_delta(short_ratio, 5)), ops.ts_sum(foreign_net_ratio, 5)))",
    
    # 외국인 순매수 지속성
    "ops.normed_rank(ops.ts_ir(foreign_net_ratio, 20))",
    
    # 기관 vs 개인 (스마트머니)
    "ops.normed_rank(ops.minus(ops.ts_sum(inst_net_ratio, 15), ops.ts_sum(indiv_net_ratio, 15)))",
]

_global_data = None

def set_global_data(data):
    global _global_data
    _global_data = data

def evaluate_alpha(alpha_expr):
    """알파 평가 (20일 forward IC)"""
    global _global_data
    data = _global_data
    
    try:
        close = data['close']
        volume = data['volume']
        returns = data['returns']
        foreign_net_ratio = data['foreign_net_ratio']
        inst_net_ratio = data['inst_net_ratio']
        indiv_net_ratio = data['indiv_net_ratio']
        foreign_ownership_pct = data['foreign_ownership_pct']
        short_ratio = data['short_ratio']
        
        forward_return = close.shift(-20) / close - 1
        alpha_values = eval(alpha_expr)
        
        if not isinstance(alpha_values, pd.DataFrame):
            return (alpha_expr, -999.0)
        
        ic_list = []
        for date in alpha_values.index[:-20]:
            alpha_cs = alpha_values.loc[date]
            returns_cs = forward_return.loc[date]
            valid = alpha_cs.notna() & returns_cs.notna()
            
            if valid.sum() > 30:
                ic = alpha_cs[valid].corr(returns_cs[valid])
                if not np.isnan(ic):
                    ic_list.append(ic)
        
        if len(ic_list) < 10:
            return (alpha_expr, -999.0)
        
        return (alpha_expr, float(np.mean(ic_list)))
    except Exception as e:
        return (alpha_expr, -999.0)

def mutate_supply_alpha(alpha_expr):
    """수급 알파 변이"""
    try:
        # 윈도우 변이
        import re
        matches = list(re.finditer(r'(ts_\w+)\([^,]+,\s*(\d+)\)', alpha_expr))
        if matches:
            match = random.choice(matches)
            old_window = int(match.group(2))
            new_window = max(5, min(30, old_window + random.choice([-5, -3, 3, 5])))
            start, end = match.span(2)
            return alpha_expr[:start] + str(new_window) + alpha_expr[end:]
        
        # 수급 변수 교체
        supply_vars = ['foreign_net_ratio', 'inst_net_ratio', 'indiv_net_ratio', 'short_ratio']
        for old_var in supply_vars:
            if old_var in alpha_expr:
                new_var = random.choice([v for v in supply_vars if v != old_var])
                return alpha_expr.replace(old_var, new_var, 1)
        
        return None
    except:
        return None

def main():
    print("=" * 80)
    print("📊 수급 중심 알파 생성기")
    print("=" * 80)
    
    # 데이터 로드
    data = load_market_data()
    set_global_data(data)
    
    # 시드 평가
    print(f"\n🌱 시드 알파 {len(SUPPLY_DEMAND_SEEDS)}개 평가 중...")
    
    results = []
    for alpha in SUPPLY_DEMAND_SEEDS:
        expr, ic = evaluate_alpha(alpha)
        if ic > -999:
            results.append((expr, ic))
            print(f"   IC {ic:+.4f}: {alpha[:60]}...")
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n✅ 유효한 시드: {len(results)}개")
    
    # GP 진화 (간단 버전)
    print("\n🧬 GP 진화 시작 (수급 전용, 20세대)...")
    
    population = [expr for expr, _ in results]
    best_ever = results[0] if results else (None, -999)
    
    for gen in range(20):
        # 변이
        new_alphas = []
        for alpha in population[:10]:
            for _ in range(3):
                mutated = mutate_supply_alpha(alpha)
                if mutated and mutated not in population:
                    new_alphas.append(mutated)
        
        # 평가
        all_alphas = population + new_alphas
        with Pool(4, initializer=set_global_data, initargs=(data,)) as pool:
            eval_results = pool.map(evaluate_alpha, all_alphas)
        
        # 정렬
        valid_results = [(e, ic) for e, ic in eval_results if ic > -999]
        valid_results.sort(key=lambda x: x[1], reverse=True)
        
        if valid_results and valid_results[0][1] > best_ever[1]:
            best_ever = valid_results[0]
            print(f"  세대 {gen+1}: IC {best_ever[1]:.4f} 🏆")
        else:
            print(f"  세대 {gen+1}: IC {best_ever[1]:.4f}")
        
        population = [e for e, _ in valid_results[:20]]
    
    # 최종 결과
    print("\n" + "=" * 80)
    print("🏆 최적 수급 알파")
    print("=" * 80)
    print(f"IC: {best_ever[1]:.4f}")
    print(f"Expression: {best_ever[0]}")
    
    # DB 저장
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alpha_formulas (formula, ic_score, description, created_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (formula) DO UPDATE SET ic_score = EXCLUDED.ic_score
        """, (best_ever[0], best_ever[1], f"20d fwd, supply_demand alpha, IC={best_ever[1]:.4f}"))
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ DB 저장 완료!")
    except Exception as e:
        print(f"⚠️ DB 저장 실패: {e}")
    
    print("\n🎉 완료!")

if __name__ == "__main__":
    main()
