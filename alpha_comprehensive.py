#!/usr/bin/env python3
"""
종합 알파 생성기: 가격 + 거래량 + 수급 + 재무 모두 포함
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

def load_all_data():
    """모든 데이터 로드: 가격 + 거래량 + 수급 + 재무"""
    print("📊 종합 데이터 로드 중...")
    
    conn = get_db_connection()
    
    # 시가총액 상위 500개
    stocks_df = pd.read_sql("""
        SELECT s.id, s.ticker, s.name, s.market_cap
        FROM stocks s
        WHERE s.is_active = true AND s.market_cap IS NOT NULL
        AND EXISTS (SELECT 1 FROM price_data p WHERE p.stock_id = s.id 
                    AND p.date >= CURRENT_DATE - INTERVAL '500 days' LIMIT 1)
        ORDER BY s.market_cap DESC
        LIMIT 500
    """, conn)
    
    stock_ids = stocks_df['id'].tolist()
    stock_id_list = ', '.join(map(str, stock_ids))
    ticker_map = dict(zip(stocks_df['id'], stocks_df['ticker']))
    
    # 1. 가격 데이터
    price_df = pd.read_sql(f"""
        SELECT s.ticker, p.date, p.open, p.high, p.low, p.close, p.volume
        FROM price_data p
        JOIN stocks s ON p.stock_id = s.id
        WHERE p.stock_id IN ({stock_id_list})
        AND p.date >= CURRENT_DATE - INTERVAL '500 days'
        ORDER BY s.ticker, p.date
    """, conn)
    
    close = price_df.pivot(index='date', columns='ticker', values='close')
    volume = price_df.pivot(index='date', columns='ticker', values='volume')
    high = price_df.pivot(index='date', columns='ticker', values='high')
    low = price_df.pivot(index='date', columns='ticker', values='low')
    returns = close.pct_change()
    
    # 2. 수급 데이터
    flow_df = pd.read_sql(f"""
        SELECT s.ticker, sd.date,
               sd.foreign_net_buy, sd.institution_net_buy,
               sd.individual_net_buy, sd.foreign_ownership
        FROM supply_demand_data sd
        JOIN stocks s ON sd.stock_id = s.id
        WHERE sd.stock_id IN ({stock_id_list})
        AND sd.date >= CURRENT_DATE - INTERVAL '500 days'
        ORDER BY s.ticker, sd.date
    """, conn)
    
    foreign_net = flow_df.pivot(index='date', columns='ticker', values='foreign_net_buy')
    inst_net = flow_df.pivot(index='date', columns='ticker', values='institution_net_buy')
    foreign_own = flow_df.pivot(index='date', columns='ticker', values='foreign_ownership')
    
    # 3. 재무 데이터 (최근 분기)
    fin_df = pd.read_sql(f"""
        SELECT DISTINCT ON (fs.stock_id)
            s.ticker,
            (fs.raw_data->>'roe')::float as roe,
            (fs.raw_data->>'roa')::float as roa,
            (fs.raw_data->>'operating_margin')::float as op_margin,
            (fs.raw_data->>'net_margin')::float as net_margin,
            (fs.raw_data->>'debt_ratio')::float as debt_ratio,
            (fs.raw_data->>'current_ratio')::float as current_ratio,
            fs.revenue,
            fs.net_income,
            fs.period_end
        FROM financial_statements fs
        JOIN stocks s ON fs.stock_id = s.id
        WHERE fs.stock_id IN ({stock_id_list})
        AND fs.raw_data IS NOT NULL
        ORDER BY fs.stock_id, fs.period_end DESC
    """, conn)
    
    conn.close()
    
    # 인덱스 맞추기
    common_idx = close.index.intersection(foreign_net.index)
    common_cols = close.columns.intersection(foreign_net.columns)
    
    close = close.loc[common_idx, common_cols]
    volume = volume.loc[common_idx, common_cols]
    high = high.loc[common_idx, common_cols]
    low = low.loc[common_idx, common_cols]
    returns = returns.loc[common_idx, common_cols]
    foreign_net = foreign_net.loc[common_idx, common_cols]
    inst_net = inst_net.loc[common_idx, common_cols]
    foreign_own = foreign_own.reindex(index=common_idx, columns=common_cols)
    
    # 파생 지표
    safe_volume = volume.replace(0, np.nan)
    foreign_net_ratio = (foreign_net / safe_volume).clip(-1, 1).fillna(0)
    inst_net_ratio = (inst_net / safe_volume).clip(-1, 1).fillna(0)
    foreign_ownership_pct = (foreign_own / 100).clip(0, 1).fillna(0)
    
    # 기술적 지표
    vwap = (high + low + close) / 3
    atr = (high - low) / close
    amihud = (returns.abs() / (close * volume).replace(0, np.nan)).fillna(0)
    vol_ratio = volume / volume.rolling(20, min_periods=5).mean()
    
    # 재무 지표를 DataFrame으로 확장 (모든 날짜에 동일 값)
    fin_dict = {}
    for col in ['roe', 'roa', 'op_margin', 'net_margin', 'debt_ratio']:
        fin_series = fin_df.set_index('ticker')[col]
        fin_df_expanded = pd.DataFrame(
            np.tile(fin_series.reindex(common_cols).values, (len(common_idx), 1)),
            index=common_idx,
            columns=common_cols
        )
        fin_dict[col] = fin_df_expanded.fillna(0)
    
    print(f"✅ {len(common_cols)}개 종목, {len(common_idx)}일 데이터")
    print(f"   가격: close, volume, returns, vwap, atr, amihud")
    print(f"   수급: foreign_net_ratio, inst_net_ratio, foreign_ownership_pct")
    print(f"   재무: roe, roa, op_margin, net_margin, debt_ratio")
    
    return {
        'close': close,
        'volume': volume,
        'returns': returns,
        'vwap': vwap,
        'atr': atr,
        'amihud': amihud,
        'vol_ratio': vol_ratio,
        'foreign_net_ratio': foreign_net_ratio,
        'inst_net_ratio': inst_net_ratio,
        'foreign_ownership_pct': foreign_ownership_pct,
        'roe': fin_dict['roe'],
        'roa': fin_dict['roa'],
        'op_margin': fin_dict['op_margin'],
        'net_margin': fin_dict['net_margin'],
        'debt_ratio': fin_dict['debt_ratio'],
    }

# 종합 시드 알파 (가격 + 거래량 + 수급 + 재무)
COMPREHENSIVE_SEEDS = [
    # === 가격 + 수급 ===
    # 외국인 매수 + 가격 모멘텀
    "ops.normed_rank(ops.cwise_mul(ops.ts_sum(foreign_net_ratio, 15), ops.ts_delta_ratio(close, 15)))",
    
    # 외국인 보유비율 증가 + 상승 추세
    "ops.normed_rank(ops.cwise_mul(ops.ts_delta(foreign_ownership_pct, 20), ops.ts_delta(close, 20)))",
    
    # === 수급 + 재무 ===
    # ROE 상위 + 외국인 매수
    "ops.normed_rank(ops.cwise_mul(ops.normed_rank(roe), ops.ts_sum(foreign_net_ratio, 20)))",
    
    # 영업이익률 + 기관 매수
    "ops.normed_rank(ops.cwise_mul(ops.normed_rank(op_margin), ops.ts_sum(inst_net_ratio, 15)))",
    
    # === 가격 + 재무 ===
    # ROE + 모멘텀
    "ops.normed_rank(ops.cwise_mul(ops.normed_rank(roe), ops.ts_delta_ratio(close, 20)))",
    
    # 저부채 + 가격 상승
    "ops.normed_rank(ops.cwise_mul(ops.neg(ops.normed_rank(debt_ratio)), ops.ts_delta(close, 15)))",
    
    # === 거래량 + 수급 ===
    # 거래량 급증 + 외국인 매수
    "ops.normed_rank(ops.cwise_mul(ops.ts_delta_ratio(volume, 5), ops.ts_sum(foreign_net_ratio, 5)))",
    
    # 유동성 + 기관 매수
    "ops.normed_rank(ops.cwise_mul(ops.neg(amihud), ops.ts_sum(inst_net_ratio, 10)))",
    
    # === 종합 (3개 이상 결합) ===
    # ROE + 외국인 매수 + 모멘텀
    "ops.normed_rank(ops.cwise_mul(ops.cwise_mul(ops.normed_rank(roe), ops.ts_sum(foreign_net_ratio, 15)), ops.ts_delta_ratio(close, 15)))",
    
    # 영업이익률 + 외국인 보유비율 증가 + 거래량 안정
    "ops.normed_rank(ops.cwise_mul(ops.cwise_mul(ops.normed_rank(op_margin), ops.ts_delta(foreign_ownership_pct, 20)), ops.neg(ops.ts_std(vol_ratio, 20))))",
    
    # ROA + 기관 매수 + 저변동성
    "ops.normed_rank(ops.cwise_mul(ops.cwise_mul(ops.normed_rank(roa), ops.ts_sum(inst_net_ratio, 15)), ops.neg(ops.ts_std(returns, 20))))",
    
    # 순이익률 + 외국인 매수 + 상승 추세
    "ops.normed_rank(ops.cwise_mul(ops.cwise_mul(ops.normed_rank(net_margin), ops.ts_sum(foreign_net_ratio, 10)), ops.ts_delta(close, 10)))",
    
    # 저부채 + 외국인 보유비율 + 거래량 증가
    "ops.normed_rank(ops.cwise_mul(ops.cwise_mul(ops.neg(ops.normed_rank(debt_ratio)), ops.normed_rank(foreign_ownership_pct)), ops.ts_delta_ratio(volume, 10)))",
    
    # ROE + 외국인 + 기관 동시 매수
    "ops.normed_rank(ops.cwise_mul(ops.normed_rank(roe), ops.add(ops.ts_sum(foreign_net_ratio, 10), ops.ts_sum(inst_net_ratio, 10))))",
    
    # 영업이익률 + 모멘텀 + 수급
    "ops.normed_rank(ops.add(ops.cwise_mul(ops.normed_rank(op_margin), ops.ts_delta_ratio(close, 20)), ops.ts_sum(foreign_net_ratio, 20)))",
    
    # === 복합 점수 ===
    # (ROE + ROA) / 2 + 외국인 매수
    "ops.normed_rank(ops.cwise_mul(ops.add(ops.normed_rank(roe), ops.normed_rank(roa)), ops.ts_sum(foreign_net_ratio, 15)))",
    
    # 재무 건전성 + 수급 + 모멘텀
    "ops.normed_rank(ops.add(ops.add(ops.cwise_mul(ops.normed_rank(op_margin), ops.ts_sum(foreign_net_ratio, 10)), ops.normed_rank(ops.neg(debt_ratio))), ops.ts_delta_ratio(close, 10)))",
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
        vwap = data['vwap']
        atr = data['atr']
        amihud = data['amihud']
        vol_ratio = data['vol_ratio']
        foreign_net_ratio = data['foreign_net_ratio']
        inst_net_ratio = data['inst_net_ratio']
        foreign_ownership_pct = data['foreign_ownership_pct']
        roe = data['roe']
        roa = data['roa']
        op_margin = data['op_margin']
        net_margin = data['net_margin']
        debt_ratio = data['debt_ratio']
        
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

def mutate_alpha(alpha_expr):
    """알파 변이"""
    import re
    try:
        mutation_type = random.choice(['window', 'variable'])
        
        if mutation_type == 'window':
            matches = list(re.finditer(r'(ts_\w+)\([^,]+,\s*(\d+)\)', alpha_expr))
            if matches:
                match = random.choice(matches)
                old_window = int(match.group(2))
                new_window = max(5, min(30, old_window + random.choice([-5, -3, 3, 5])))
                start, end = match.span(2)
                return alpha_expr[:start] + str(new_window) + alpha_expr[end:]
        else:
            # 변수 교체
            var_groups = [
                ['foreign_net_ratio', 'inst_net_ratio'],
                ['roe', 'roa', 'op_margin', 'net_margin'],
                ['close', 'vwap'],
            ]
            for group in var_groups:
                for old_var in group:
                    if old_var in alpha_expr:
                        new_var = random.choice([v for v in group if v != old_var])
                        return alpha_expr.replace(old_var, new_var, 1)
        
        return None
    except:
        return None

def main():
    print("=" * 80)
    print("📊 종합 알파 생성기 (가격 + 거래량 + 수급 + 재무)")
    print("=" * 80)
    
    # 데이터 로드
    data = load_all_data()
    set_global_data(data)
    
    # 시드 평가
    print(f"\n🌱 시드 알파 {len(COMPREHENSIVE_SEEDS)}개 평가 중...")
    
    results = []
    for alpha in COMPREHENSIVE_SEEDS:
        expr, ic = evaluate_alpha(alpha)
        if ic > -999:
            results.append((expr, ic))
            sign = "+" if ic > 0 else ""
            print(f"   IC {sign}{ic:.4f}: {alpha[:70]}...")
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n✅ 유효한 시드: {len(results)}개")
    print(f"🏆 최고 시드 IC: {results[0][1]:.4f}")
    
    # GP 진화
    print("\n🧬 GP 진화 시작 (종합, 30세대)...")
    
    population = [expr for expr, _ in results]
    best_ever = results[0] if results else (None, -999)
    
    for gen in range(30):
        # 변이
        new_alphas = []
        for alpha in population[:15]:
            for _ in range(2):
                mutated = mutate_alpha(alpha)
                if mutated and mutated not in population:
                    new_alphas.append(mutated)
        
        # 평가
        all_alphas = list(set(population + new_alphas))
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
        
        population = [e for e, _ in valid_results[:25]]
    
    # 최종 결과
    print("\n" + "=" * 80)
    print("🏆 최적 종합 알파")
    print("=" * 80)
    print(f"IC: {best_ever[1]:.4f}")
    print(f"\nExpression:")
    print(f"  {best_ever[0]}")
    
    # 어떤 변수 포함?
    vars_used = []
    if 'close' in best_ever[0] or 'vwap' in best_ever[0]: vars_used.append('가격')
    if 'volume' in best_ever[0] or 'amihud' in best_ever[0]: vars_used.append('거래량')
    if 'foreign' in best_ever[0] or 'inst' in best_ever[0]: vars_used.append('수급')
    if any(f in best_ever[0] for f in ['roe', 'roa', 'margin', 'debt']): vars_used.append('재무')
    
    print(f"\n포함된 지표: {' + '.join(vars_used)}")
    
    # DB 저장
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO alpha_formulas (formula, ic_score, description, created_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (formula) DO UPDATE SET ic_score = EXCLUDED.ic_score
        """, (best_ever[0], best_ever[1], f"20d fwd, comprehensive alpha (price+volume+flow+financial), IC={best_ever[1]:.4f}"))
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ DB 저장 완료!")
    except Exception as e:
        print(f"⚠️ DB 저장 실패: {e}")
    
    print("\n🎉 완료!")

if __name__ == "__main__":
    main()
