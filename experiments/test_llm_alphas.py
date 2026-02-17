#!/usr/bin/env python3
"""Test 30 LLM-generated alpha formulas on KOSDAQ 200."""

import sys
sys.path.insert(0, '/Users/yrbahn/.openclaw/workspace/alpha-gpt-kr')

import os
import numpy as np
import pandas as pd
import psycopg2
from dotenv import load_dotenv
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

def load_kosdaq200_data():
    """KOSDAQ 200 데이터 로드"""
    print("📊 데이터 로드 중... (KOSDAQ 시총 상위 200종목)")
    
    conn = get_db_connection()
    
    # 시가총액 상위 200개 (KOSDAQ = ticker >= '400000')
    query_stocks = """
        SELECT s.id, s.ticker, s.name, s.market_cap
        FROM stocks s
        WHERE s.is_active = true
        AND s.market_cap IS NOT NULL
        AND EXISTS (
            SELECT 1 FROM price_data p 
            WHERE p.stock_id = s.id 
            AND p.date >= '2019-01-01'
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
        SELECT s.ticker, p.date, p.open, p.high, p.low, p.close, p.volume
        FROM price_data p
        JOIN stocks s ON p.stock_id = s.id
        WHERE p.stock_id IN ({stock_id_list})
        AND p.date >= '2019-01-01'
        ORDER BY s.ticker, p.date
    """
    
    price_df = pd.read_sql(query_prices, conn)
    conn.close()
    
    close = price_df.pivot(index='date', columns='ticker', values='close')
    high = price_df.pivot(index='date', columns='ticker', values='high')
    low = price_df.pivot(index='date', columns='ticker', values='low')
    volume = price_df.pivot(index='date', columns='ticker', values='volume')
    amount = close * volume
    returns = close.pct_change()
    
    return {
        'close': close,
        'high': high,
        'low': low,
        'volume': volume,
        'amount': amount,
        'returns': returns
    }

# Load data
data = load_kosdaq200_data()
print(f"Loaded {len(data['returns'].columns)} stocks, {len(data['returns'])} days")

# Forward returns for 20-day rebalancing
forward_ret = data['close'].pct_change(20).shift(-20)

# LLM-generated alphas
llm_alphas = [
    ("LLM01_ATR100_Mom25", "ops.add(ops.zscore_scale(ops.neg(ops.ts_mean(ops.ts_std(atr_ratio, 100), 20))), ops.zscore_scale(ops.ts_delta(close, 25)))"),
    ("LLM02_VolStd50_PriceStrength", "ops.add(ops.zscore_scale(ops.neg(ops.ts_std(volume, 50))), ops.zscore_scale(ops.div(close, ops.ts_min(close, 40))))"),
    ("LLM03_HLRange150_RetRank", "ops.add(ops.zscore_scale(ops.neg(ops.ts_mean(high_low_range, 150))), ops.zscore_scale(ops.ts_rank(returns, 30)))"),
    ("LLM04_VolCloseCorr_AmtStrength", "ops.add(ops.zscore_scale(ops.neg(ops.ts_corr(volume, close, 90))), ops.zscore_scale(ops.div(amount, ops.ts_max(amount, 60))))"),
    ("LLM05_ATR120_MinRet", "ops.add(ops.zscore_scale(ops.neg(ops.ts_std(atr_ratio, 120))), ops.zscore_scale(ops.ts_min(returns, 45)))"),
    ("LLM06_CloseStd70_HLRank", "ops.add(ops.zscore_scale(ops.neg(ops.ts_mean(ops.ts_std(close, 70), 25))), ops.zscore_scale(ops.ts_rank(high_low_range, 50)))"),
    ("LLM07_VolStd45_PriceDelta", "ops.add(ops.zscore_scale(ops.neg(ops.ts_std(volume, 45))), ops.zscore_scale(ops.div(close, ops.ts_delta(close, 35))))"),
    ("LLM08_VolRetCorr_NormRank", "ops.add(ops.zscore_scale(ops.neg(ops.ts_corr(volume, returns, 60))), ops.zscore_scale(ops.normed_rank(returns)))"),
    ("LLM09_HLRange80_AmtMin", "ops.add(ops.zscore_scale(ops.neg(ops.ts_mean(high_low_range, 80))), ops.zscore_scale(ops.div(amount, ops.ts_min(amount, 95))))"),
    ("LLM10_ATR135_Mom40", "ops.add(ops.zscore_scale(ops.neg(ops.ts_std(atr_ratio, 135))), ops.zscore_scale(ops.ts_delta(close, 40)))"),
    ("LLM11_HLStd85_VolCorr", "ops.add(ops.zscore_scale(ops.neg(ops.ts_mean(ops.ts_std(high_low_range, 85), 30))), ops.zscore_scale(ops.ts_corr(volume, close, 100)))"),
    ("LLM12_VolStd90_PriceMax", "ops.add(ops.zscore_scale(ops.neg(ops.ts_std(volume, 90))), ops.zscore_scale(ops.div(close, ops.ts_max(close, 55))))"),
    ("LLM13_RetMean50_RetRank", "ops.add(ops.zscore_scale(ops.neg(ops.ts_mean(returns, 50))), ops.zscore_scale(ops.ts_rank(returns, 20)))"),
    ("LLM14_AmtHLCorr_AmtDelta", "ops.add(ops.zscore_scale(ops.neg(ops.ts_corr(amount, high_low_range, 75))), ops.zscore_scale(ops.div(amount, ops.ts_delta(amount, 110))))"),
    ("LLM15_ATR125_MinRet65", "ops.add(ops.zscore_scale(ops.neg(ops.ts_std(atr_ratio, 125))), ops.zscore_scale(ops.ts_min(returns, 65)))"),
    ("LLM16_VolStd60_AmtRank", "ops.add(ops.zscore_scale(ops.neg(ops.ts_mean(ops.ts_std(volume, 60), 35))), ops.zscore_scale(ops.ts_rank(amount, 130)))"),
    ("LLM17_VolStd55_PriceDelta80", "ops.add(ops.zscore_scale(ops.neg(ops.ts_std(volume, 55))), ops.zscore_scale(ops.div(close, ops.ts_delta(close, 80))))"),
    ("LLM18_CloseRetCorr_NormRank", "ops.add(ops.zscore_scale(ops.neg(ops.ts_corr(close, returns, 110))), ops.zscore_scale(ops.normed_rank(returns)))"),
    ("LLM19_HLRange95_AmtMin70", "ops.add(ops.zscore_scale(ops.neg(ops.ts_mean(high_low_range, 95))), ops.zscore_scale(ops.div(amount, ops.ts_min(amount, 70))))"),
    ("LLM20_ATR140_Mom50", "ops.add(ops.zscore_scale(ops.neg(ops.ts_std(atr_ratio, 140))), ops.zscore_scale(ops.ts_delta(close, 50)))"),
    ("LLM21_LowStd65_VolCorr", "ops.add(ops.zscore_scale(ops.neg(ops.ts_mean(ops.ts_std(low, 65), 15))), ops.zscore_scale(ops.ts_corr(volume, close, 85)))"),
    ("LLM22_AmtStd100_PriceMax", "ops.add(ops.zscore_scale(ops.neg(ops.ts_std(amount, 100))), ops.zscore_scale(ops.div(close, ops.ts_max(close, 40))))"),
    ("LLM23_RetMean70_RetRank25", "ops.add(ops.zscore_scale(ops.neg(ops.ts_mean(returns, 70))), ops.zscore_scale(ops.ts_rank(returns, 25)))"),
    ("LLM24_AmtCloseCorr_AmtDelta", "ops.add(ops.zscore_scale(ops.neg(ops.ts_corr(amount, close, 120))), ops.zscore_scale(ops.div(amount, ops.ts_delta(amount, 130))))"),
    ("LLM25_ATR115_MinRet40", "ops.add(ops.zscore_scale(ops.neg(ops.ts_std(atr_ratio, 115))), ops.zscore_scale(ops.ts_min(returns, 40)))"),
    ("LLM26_VolStd75_AmtRank90", "ops.add(ops.zscore_scale(ops.neg(ops.ts_mean(ops.ts_std(volume, 75), 20))), ops.zscore_scale(ops.ts_rank(amount, 90)))"),
    ("LLM27_VolStd85_PriceDelta75", "ops.add(ops.zscore_scale(ops.neg(ops.ts_std(volume, 85))), ops.zscore_scale(ops.div(close, ops.ts_delta(close, 75))))"),
    ("LLM28_CloseRetCorr95_NormRank", "ops.add(ops.zscore_scale(ops.neg(ops.ts_corr(close, returns, 95))), ops.zscore_scale(ops.normed_rank(returns)))"),
    ("LLM29_HLRange125_AmtMin130", "ops.add(ops.zscore_scale(ops.neg(ops.ts_mean(high_low_range, 125))), ops.zscore_scale(ops.div(amount, ops.ts_min(amount, 130))))"),
    ("LLM30_ATR105_Mom60", "ops.add(ops.zscore_scale(ops.neg(ops.ts_std(atr_ratio, 105))), ops.zscore_scale(ops.ts_delta(close, 60)))"),
]

# 4-fold CV dates
folds = [
    ('2020-01-01', '2021-06-30', '2021-07-21', '2022-12-31'),
    ('2020-01-01', '2022-06-30', '2022-07-21', '2023-12-31'),
    ('2020-01-01', '2023-06-30', '2023-07-21', '2024-12-31'),
    ('2020-01-01', '2024-06-30', '2024-07-21', '2025-12-31'),
]

def calc_ic(alpha_vals, fwd_ret, start, end):
    """Calculate IC for a date range."""
    # Convert index to string for comparison
    idx_str = pd.to_datetime(alpha_vals.index).strftime('%Y-%m-%d')
    mask = (idx_str >= start) & (idx_str <= end)
    a = alpha_vals.loc[mask]
    r = fwd_ret.loc[mask]
    
    ics = []
    for dt in a.index:
        if dt not in r.index:
            continue
        av = a.loc[dt].dropna()
        rv = r.loc[dt].dropna()
        common = av.index.intersection(rv.index)
        if len(common) < 20:
            continue
        ic = av[common].corr(rv[common])
        if not np.isnan(ic):
            ics.append(ic)
    return np.mean(ics) if ics else 0, np.std(ics) if ics else 1

results = []

for name, formula in llm_alphas:
    print(f"\nTesting {name}...")
    
    try:
        # Build local namespace
        atr_ratio = data['high'] / data['low'] - 1
        high_low_range = (data['high'] - data['low']) / data['close']
        
        local_ns = {
            'ops': ops,
            'close': data['close'],
            'volume': data['volume'],
            'amount': data['amount'],
            'high': data['high'],
            'low': data['low'],
            'returns': data['returns'],
            'atr_ratio': atr_ratio,
            'high_low_range': high_low_range,
        }
        
        alpha_vals = eval(formula, {"__builtins__": {}}, local_ns)
        
        # Calculate CV metrics
        test_ics = []
        for train_start, train_end, test_start, test_end in folds:
            test_ic, test_std = calc_ic(alpha_vals, forward_ret, test_start, test_end)
            test_ics.append(test_ic)
        
        avg_test_ic = np.mean(test_ics)
        std_test_ic = np.std(test_ics)
        ir = avg_test_ic / std_test_ic if std_test_ic > 0 else 0
        
        results.append({
            'name': name,
            'test_ic': avg_test_ic,
            'test_std': std_test_ic,
            'ir': ir,
            'formula': formula[:60] + '...'
        })
        print(f"  Test IC: {avg_test_ic:.4f}, IR: {ir:.2f}")
        
    except Exception as e:
        print(f"  Error: {e}")
        results.append({
            'name': name,
            'test_ic': 0,
            'test_std': 0,
            'ir': 0,
            'formula': f"ERROR: {str(e)[:40]}"
        })

# Sort by test IC
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('test_ic', ascending=False)

print("\n" + "="*70)
print("TOP 10 LLM-GENERATED ALPHAS (by Test IC)")
print("="*70)
for i, row in results_df.head(10).iterrows():
    print(f"{row['name']:30s} | IC: {row['test_ic']:+.4f} | IR: {row['ir']:.2f}")

print("\n" + "="*70)
print("COMPARISON WITH CURRENT BEST")
print("="*70)
print("Current best (LV3+모멘텀): IC = 0.1198, IR = 1.03")
print(f"Best LLM alpha: {results_df.iloc[0]['name']} IC = {results_df.iloc[0]['test_ic']:.4f}")

# Save results
results_df.to_csv('/Users/yrbahn/.openclaw/workspace/alpha-gpt-kr/experiments/llm_alpha_results.csv', index=False)
print("\nResults saved to experiments/llm_alpha_results.csv")
