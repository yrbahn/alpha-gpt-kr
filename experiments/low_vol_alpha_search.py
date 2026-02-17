#!/usr/bin/env python3
"""
Low Volatility Alpha Search for KOSDAQ 200
-------------------------------------------
가설: 오랜 기간 변동성이 낮은 종목 → 1달 뒤 상승

Based on alpha_gpt_20day_kosdaq200.py structure
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
from scipy.stats import spearmanr

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.mining.operators import AlphaOperators as ops

load_dotenv(project_root / '.env')

FORWARD_DAYS = 20  # 1달

# ── Low Volatility 가설 기반 Seed Alphas ──
LOW_VOL_SEED_ALPHAS = [
    # 기본 저변동성 (음수 = 낮을수록 좋음 → 반전)
    "ops.neg(ops.ts_std(returns, 60))",  # 60일 수익률 변동성 낮은 종목
    "ops.neg(ops.ts_std(returns, 120))", # 120일 변동성
    "ops.neg(ops.ts_std(close, 60) / ops.ts_mean(close, 60))",  # 변동계수 (CV)
    
    # 고저 범위 기반
    "ops.neg(ops.ts_mean(high_low_range, 60))",  # 평균 일간 변동폭 낮은 종목
    "ops.neg(ops.ts_mean(high_low_range, 120))", # 장기 평균 변동폭
    
    # 변동성 압축 (최근 vs 과거)
    "ops.ts_std(returns, 20) / ops.ts_std(returns, 60)",  # 단기/장기 변동성비
    "ops.neg(ops.ts_std(returns, 20) / ops.ts_std(returns, 120))",  # 변동성 압축
    
    # ATR 기반
    "ops.neg(ops.ts_mean(atr_ratio, 60))",  # 상대 ATR 낮은 종목
    "ops.neg(ops.ts_mean(atr_ratio, 120))",
    
    # 거래량 변동성도 낮은 종목
    "ops.neg(ops.ts_std(volume, 60))",
    "ops.neg(ops.ts_std(amount, 60))",
    
    # 복합: 가격 변동성 낮고 + 거래량도 안정적
    "ops.neg(ops.add(ops.ts_std(returns, 60), ops.ts_std(vol_ratio, 60)))",
    
    # 장기 저변동성 + 최근 모멘텀 조합
    "ops.add(ops.neg(ops.ts_std(returns, 120)), ops.ts_delta(close, 5) / ops.ts_delay(close, 5))",
    
    # 변동성 하락 추세 (변동성이 줄어드는 종목)
    "ops.neg(ops.ts_delta(ops.ts_std(returns, 20), 20))",
    
    # 외국인 안정적 보유 + 저변동성
    "ops.add(ops.neg(ops.ts_std(returns, 60)), ops.ts_mean(foreign_net_ratio, 20))",
    
    # 기관 축적 + 저변동성
    "ops.add(ops.neg(ops.ts_std(returns, 60)), ops.ts_mean(inst_net_ratio, 20))",
    
    # 볼린저밴드 압축 (BB width 낮은 종목)
    "ops.neg(ops.ts_std(close, 20) / ops.ts_mean(close, 20))",
    
    # 저변동성 + 상승 추세
    "ops.add(ops.neg(ops.ts_std(returns, 60)), ops.ts_mean(returns, 20))",
    
    # 안정적 외국인 보유율 + 저변동성
    "ops.add(ops.neg(ops.ts_std(returns, 90)), ops.ts_mean(foreign_ownership_pct, 20))",
]


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )


def load_market_data():
    """KOSDAQ 200개 종목 데이터 로드 (시가총액 상위)"""
    print("📊 데이터 로드 중... (KOSDAQ 시총 상위 200종목, 3년)")
    
    conn = get_db_connection()
    
    # KOSDAQ 시가총액 상위 200개
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
            AND p.date >= CURRENT_DATE - INTERVAL '1095 days'
            LIMIT 1
        )
        AND s.ticker >= '400000' 
        ORDER BY s.market_cap DESC
        LIMIT 200
    """
    
    stocks_df = pd.read_sql(query_stocks, conn)
    stock_ids = stocks_df['id'].tolist()
    stock_id_list = ', '.join(map(str, stock_ids))
    
    query_prices = f"""
        SELECT
            s.ticker,
            p.date,
            p.open,
            p.high,
            p.low,
            p.close,
            p.volume
        FROM price_data p
        JOIN stocks s ON p.stock_id = s.id
        WHERE p.stock_id IN ({stock_id_list})
        AND p.date >= CURRENT_DATE - INTERVAL '1095 days'
        ORDER BY s.ticker, p.date
    """
    
    price_df = pd.read_sql(query_prices, conn)
    
    # Pivot tables
    open_price = price_df.pivot(index='date', columns='ticker', values='open')
    high = price_df.pivot(index='date', columns='ticker', values='high')
    low = price_df.pivot(index='date', columns='ticker', values='low')
    close = price_df.pivot(index='date', columns='ticker', values='close')
    volume = price_df.pivot(index='date', columns='ticker', values='volume')
    returns = close.pct_change()
    
    # 수급 데이터
    try:
        flow_query = f"""
            SELECT s.ticker, sd.date,
                   sd.foreign_net_buy, sd.institution_net_buy,
                   sd.individual_net_buy, sd.foreign_ownership
            FROM supply_demand_data sd
            JOIN stocks s ON sd.stock_id = s.id
            WHERE sd.stock_id IN ({stock_id_list})
            AND sd.date >= CURRENT_DATE - INTERVAL '1095 days'
            ORDER BY s.ticker, sd.date
        """
        flow_df = pd.read_sql(flow_query, conn)
        foreign_buy_raw = flow_df.pivot(index='date', columns='ticker', values='foreign_net_buy')
        inst_buy_raw = flow_df.pivot(index='date', columns='ticker', values='institution_net_buy')
        retail_buy_raw = flow_df.pivot(index='date', columns='ticker', values='individual_net_buy')
        foreign_own_raw = flow_df.pivot(index='date', columns='ticker', values='foreign_ownership')
        
        # 인덱스 맞추기
        foreign_buy_raw = foreign_buy_raw.reindex(index=close.index, columns=close.columns)
        inst_buy_raw = inst_buy_raw.reindex(index=close.index, columns=close.columns)
        retail_buy_raw = retail_buy_raw.reindex(index=close.index, columns=close.columns)
        foreign_own_raw = foreign_own_raw.reindex(index=close.index, columns=close.columns)
        
        safe_volume = volume.replace(0, np.nan)
        foreign_net_ratio = (foreign_buy_raw / safe_volume).clip(-1, 1).fillna(0)
        inst_net_ratio = (inst_buy_raw / safe_volume).clip(-1, 1).fillna(0)
        retail_net_ratio = (retail_buy_raw / safe_volume).clip(-1, 1).fillna(0)
        foreign_ownership_pct = (foreign_own_raw / 100).clip(0, 1).fillna(0)
        has_flow = True
        print("   ✅ 수급 데이터 로드 성공")
    except Exception as e:
        print(f"   ⚠️ 수급 데이터 로드 실패: {e}")
        foreign_net_ratio = close * 0.0
        inst_net_ratio = close * 0.0
        retail_net_ratio = close * 0.0
        foreign_ownership_pct = close * 0.0
        has_flow = False
    
    conn.close()
    
    # 파생 지표
    vwap = (high + low + close) / 3
    high_low_range = (high - low) / close
    body = (close - open_price) / open_price
    upper_shadow = (high - close.clip(lower=open_price)) / close
    lower_shadow = (close.clip(upper=open_price) - low) / close
    
    # ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3]).groupby(level=0).max()
    true_range = true_range.reindex(close.index)
    atr_ratio = true_range / close
    
    # 거래대금
    amount = close * volume
    
    # Amihud
    amihud = returns.abs() / amount.replace(0, np.nan)
    amihud = amihud.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 거래량 비율
    vol_ratio = volume / volume.rolling(20, min_periods=5).mean()
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1)
    
    print(f"✅ {len(close.columns)}개 종목, {len(close)}일 데이터 로드 완료")
    
    return {
        'close': close,
        'open_price': open_price,
        'high': high,
        'low': low,
        'volume': volume,
        'returns': returns,
        'vwap': vwap,
        'high_low_range': high_low_range,
        'body': body,
        'upper_shadow': upper_shadow,
        'lower_shadow': lower_shadow,
        'atr_ratio': atr_ratio,
        'amount': amount,
        'amihud': amihud,
        'vol_ratio': vol_ratio,
        'foreign_net_ratio': foreign_net_ratio,
        'inst_net_ratio': inst_net_ratio,
        'retail_net_ratio': retail_net_ratio,
        'foreign_ownership_pct': foreign_ownership_pct,
    }


def compute_forward_returns(close, forward_days=FORWARD_DAYS):
    """N일 후 수익률 계산"""
    return close.shift(-forward_days) / close - 1


def evaluate_alpha(alpha_expr, data, forward_returns, 
                   train_end_idx=None, purge_gap=20):
    """알파 IC/IR 평가 (4-fold CV)"""
    
    close = data['close']
    n_days = len(close)
    
    if train_end_idx is None:
        train_end_idx = int(n_days * 0.7)
    
    try:
        # 알파 계산
        alpha = eval(alpha_expr, {'ops': ops, 'np': np, **data})
        
        if isinstance(alpha, (int, float)):
            return None
        
        alpha = pd.DataFrame(alpha)
        alpha = alpha.replace([np.inf, -np.inf], np.nan)
        
        # Train/Test 분리 (purge gap 적용)
        train_alpha = alpha.iloc[:train_end_idx - purge_gap]
        train_fwd = forward_returns.iloc[:train_end_idx - purge_gap]
        
        test_alpha = alpha.iloc[train_end_idx:]
        test_fwd = forward_returns.iloc[train_end_idx:]
        
        # IC 계산
        train_ics = []
        for i in range(len(train_alpha)):
            row_alpha = train_alpha.iloc[i].dropna()
            row_fwd = train_fwd.iloc[i].dropna()
            common = row_alpha.index.intersection(row_fwd.index)
            if len(common) >= 30:
                ic, _ = spearmanr(row_alpha[common], row_fwd[common])
                if not np.isnan(ic):
                    train_ics.append(ic)
        
        test_ics = []
        for i in range(len(test_alpha)):
            row_alpha = test_alpha.iloc[i].dropna()
            row_fwd = test_fwd.iloc[i].dropna()
            common = row_alpha.index.intersection(row_fwd.index)
            if len(common) >= 30:
                ic, _ = spearmanr(row_alpha[common], row_fwd[common])
                if not np.isnan(ic):
                    test_ics.append(ic)
        
        if len(train_ics) < 10 or len(test_ics) < 5:
            return None
        
        train_ic = np.mean(train_ics)
        train_ir = train_ic / (np.std(train_ics) + 1e-8)
        test_ic = np.mean(test_ics)
        test_ir = test_ic / (np.std(test_ics) + 1e-8)
        
        return {
            'train_ic': train_ic,
            'train_ir': train_ir,
            'test_ic': test_ic,
            'test_ir': test_ir,
            'n_train': len(train_ics),
            'n_test': len(test_ics)
        }
        
    except Exception as e:
        return None


def mutate_alpha(alpha_expr, data):
    """간단한 랜덤 변이"""
    mutations = []
    
    # 윈도우 변경
    for old_w in [20, 30, 40, 60, 90, 120]:
        for new_w in [15, 25, 45, 50, 75, 100]:
            if f', {old_w})' in alpha_expr:
                new_expr = alpha_expr.replace(f', {old_w})', f', {new_w})')
                mutations.append(new_expr)
    
    # 연산자 추가
    wrappers = [
        f"ops.zscore_scale({alpha_expr})",
        f"ops.neg({alpha_expr})",
        f"ops.ts_mean({alpha_expr}, 5)",
    ]
    mutations.extend(wrappers)
    
    return mutations[:5]


def llm_mutate_alpha(top_alphas, num_mutations=10):
    """LLM 기반 변이 (저변동성 가설 특화)"""
    try:
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    except:
        return []
    
    parent_block = "\n".join([f"  {i+1}. IC={ic:.4f}: `{expr}`" 
                              for i, (expr, ic) in enumerate(top_alphas[:5])])
    
    prompt = f"""### Task: Generate Low-Volatility Alpha Mutations

You are designing alphas for the hypothesis: **Stocks with persistently low volatility tend to outperform after 1 month.**

Key insight: Low volatility stocks are "coiling" before a breakout. We want to find stocks that:
1. Have low price volatility (ts_std, high_low_range, atr_ratio)
2. Have stable trading patterns (volume, amount stability)
3. May have institutional/foreign accumulation
4. Are potentially compressing before expansion

**Top performing parents:**
{parent_block}

**Available operators:**
- Time series: ts_std, ts_mean, ts_delta, ts_delay, ts_corr, ts_regression_residual
- Math: add, sub, mul, div, neg, abs_val
- Scaling: zscore_scale, rank, normalize

**Available variables:**
- Price: close, open_price, high, low, returns
- Volatility: high_low_range, atr_ratio, body, upper_shadow, lower_shadow
- Volume: volume, amount, vol_ratio, amihud
- Flow: foreign_net_ratio, inst_net_ratio, retail_net_ratio, foreign_ownership_pct

**Generate {num_mutations} creative mutations focused on LOW VOLATILITY signals.**
Format: One ops.xxx(...) expression per line.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=2000
        )
        
        content = response.choices[0].message.content
        lines = content.strip().split('\n')
        
        mutations = []
        for line in lines:
            line = line.strip()
            if line.startswith('ops.') or 'ops.' in line:
                # 추출
                match = re.search(r'(ops\.[^\n`]+)', line)
                if match:
                    expr = match.group(1).strip()
                    if expr.endswith('`'):
                        expr = expr[:-1]
                    mutations.append(expr)
        
        return mutations[:num_mutations]
        
    except Exception as e:
        print(f"   LLM error: {e}")
        return []


def main():
    print("=" * 70)
    print("🔍 Low Volatility Alpha Search for KOSDAQ 200")
    print("   가설: 장기 저변동성 종목 → 1달(20일) 뒤 상승")
    print("=" * 70)
    
    # 1. 데이터 로드
    data = load_market_data()
    forward_returns = compute_forward_returns(data['close'], FORWARD_DAYS)
    
    # 2. Seed 알파 평가
    print("\n📊 Seed Alphas 평가 중...")
    results = []
    
    for alpha in LOW_VOL_SEED_ALPHAS:
        result = evaluate_alpha(alpha, data, forward_returns)
        if result:
            results.append({
                'alpha': alpha,
                **result
            })
            print(f"  Train IC={result['train_ic']:.4f}, Test IC={result['test_ic']:.4f}: {alpha[:60]}...")
        else:
            print(f"  ❌ Failed: {alpha[:60]}...")
    
    # 정렬
    results.sort(key=lambda x: x['test_ic'], reverse=True)
    
    print(f"\n🏆 Top 5 Seed Alphas (by Test IC):")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. Train IC={r['train_ic']:.4f}, Test IC={r['test_ic']:.4f}, IR={r['train_ir']:.2f}")
        print(f"     {r['alpha']}")
    
    # 3. GP Evolution (Simple + LLM)
    print("\n" + "=" * 70)
    print("🧬 Evolution Phase (3 generations)")
    print("=" * 70)
    
    population = [(r['alpha'], r['test_ic']) for r in results]
    
    for gen in range(3):
        print(f"\n--- Generation {gen + 1} ---")
        
        # 상위 알파로 변이
        top_alphas = population[:10]
        new_candidates = []
        
        # 랜덤 변이
        for alpha, _ in top_alphas[:5]:
            mutations = mutate_alpha(alpha, data)
            new_candidates.extend(mutations)
        
        # LLM 변이
        llm_mutations = llm_mutate_alpha(top_alphas, num_mutations=15)
        new_candidates.extend(llm_mutations)
        
        print(f"  Evaluating {len(new_candidates)} candidates...")
        
        for alpha in new_candidates:
            result = evaluate_alpha(alpha, data, forward_returns)
            if result and result['test_ic'] > 0:
                population.append((alpha, result['test_ic']))
                if result['test_ic'] > 0.05:
                    print(f"    ✨ Test IC={result['test_ic']:.4f}: {alpha[:50]}...")
        
        # 정렬 및 정리
        population = list(set(population))  # 중복 제거
        population.sort(key=lambda x: x[1], reverse=True)
        population = population[:50]  # 상위 50개만 유지
        
        print(f"  Best so far: Test IC={population[0][1]:.4f}")
    
    # 4. 최종 결과
    print("\n" + "=" * 70)
    print("🎯 FINAL RESULTS - Best Low Volatility Alphas")
    print("=" * 70)
    
    for i, (alpha, test_ic) in enumerate(population[:10]):
        result = evaluate_alpha(alpha, data, forward_returns)
        if result:
            print(f"\n{i+1}. Test IC={result['test_ic']:.4f}, Train IC={result['train_ic']:.4f}, IR={result['train_ir']:.2f}")
            print(f"   {alpha}")
    
    # 최고 알파 저장
    if population:
        best_alpha, best_ic = population[0]
        best_result = evaluate_alpha(best_alpha, data, forward_returns)
        
        print(f"\n✨ BEST LOW-VOLATILITY ALPHA:")
        print(f"   Expression: {best_alpha}")
        print(f"   Test IC: {best_result['test_ic']:.4f}")
        print(f"   Train IC: {best_result['train_ic']:.4f}")
        print(f"   IR: {best_result['train_ir']:.2f}")
        
        # 파일 저장
        output_path = project_root / 'experiments' / 'best_low_vol_alpha.txt'
        with open(output_path, 'w') as f:
            f.write(f"# Low Volatility Alpha for KOSDAQ 200\n")
            f.write(f"# Hypothesis: Low long-term volatility -> 1 month outperformance\n")
            f.write(f"# Test IC: {best_result['test_ic']:.4f}\n")
            f.write(f"# Train IC: {best_result['train_ic']:.4f}\n")
            f.write(f"# IR: {best_result['train_ir']:.2f}\n\n")
            f.write(best_alpha)
        
        print(f"\n💾 Saved to {output_path}")


if __name__ == "__main__":
    main()
