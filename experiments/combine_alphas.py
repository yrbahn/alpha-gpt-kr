#!/usr/bin/env python3
"""
Combine Best Alpha + Low Volatility Alpha
------------------------------------------
기존 최고 알파 + 저변동성 알파 결합 실험
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2
from scipy.stats import spearmanr

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.mining.operators import AlphaOperators as ops

load_dotenv(project_root / '.env')

FORWARD_DAYS = 20


def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )


def load_market_data():
    """KOSDAQ 200개 종목 데이터 로드"""
    print("📊 데이터 로드 중...")
    
    conn = get_db_connection()
    
    query_stocks = """
        SELECT s.id, s.ticker, s.name, s.market_cap
        FROM stocks s
        WHERE s.is_active = true AND s.market_cap IS NOT NULL
        AND EXISTS (SELECT 1 FROM price_data p WHERE p.stock_id = s.id AND p.date >= CURRENT_DATE - INTERVAL '1095 days' LIMIT 1)
        AND s.ticker >= '400000' 
        ORDER BY s.market_cap DESC LIMIT 200
    """
    
    stocks_df = pd.read_sql(query_stocks, conn)
    stock_ids = stocks_df['id'].tolist()
    stock_id_list = ', '.join(map(str, stock_ids))
    
    query_prices = f"""
        SELECT s.ticker, p.date, p.open, p.high, p.low, p.close, p.volume
        FROM price_data p JOIN stocks s ON p.stock_id = s.id
        WHERE p.stock_id IN ({stock_id_list}) AND p.date >= CURRENT_DATE - INTERVAL '1095 days'
        ORDER BY s.ticker, p.date
    """
    
    price_df = pd.read_sql(query_prices, conn)
    
    open_price = price_df.pivot(index='date', columns='ticker', values='open')
    high = price_df.pivot(index='date', columns='ticker', values='high')
    low = price_df.pivot(index='date', columns='ticker', values='low')
    close = price_df.pivot(index='date', columns='ticker', values='close')
    volume = price_df.pivot(index='date', columns='ticker', values='volume')
    returns = close.pct_change()
    
    # 수급 데이터
    try:
        flow_query = f"""
            SELECT s.ticker, sd.date, sd.foreign_net_buy, sd.institution_net_buy,
                   sd.individual_net_buy, sd.foreign_ownership
            FROM supply_demand_data sd JOIN stocks s ON sd.stock_id = s.id
            WHERE sd.stock_id IN ({stock_id_list}) AND sd.date >= CURRENT_DATE - INTERVAL '1095 days'
        """
        flow_df = pd.read_sql(flow_query, conn)
        foreign_own_raw = flow_df.pivot(index='date', columns='ticker', values='foreign_ownership').reindex(index=close.index, columns=close.columns)
        foreign_ownership_pct = (foreign_own_raw / 100).clip(0, 1).fillna(0)
        print("   ✅ 수급 데이터 로드")
    except:
        foreign_ownership_pct = close * 0.0
    
    conn.close()
    
    # 파생 지표
    high_low_range = (high - low) / close
    body = (close - open_price) / open_price
    upper_shadow = (high - close.clip(lower=open_price)) / close
    amount = close * volume
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3]).groupby(level=0).max().reindex(close.index)
    atr_ratio = true_range / close
    
    print(f"✅ {len(close.columns)}개 종목 로드 완료")
    
    return {
        'close': close, 'open_price': open_price, 'high': high, 'low': low,
        'volume': volume, 'returns': returns,
        'high_low_range': high_low_range, 'body': body,
        'upper_shadow': upper_shadow, 'amount': amount,
        'atr_ratio': atr_ratio,
        'foreign_ownership_pct': foreign_ownership_pct,
    }


def compute_best_alpha(data):
    """기존 최고 알파 계산"""
    upper_shadow = data['upper_shadow']
    amount = data['amount']
    high_low_range = data['high_low_range']
    body = data['body']
    foreign_ownership_pct = data['foreign_ownership_pct']
    
    alpha = ops.zscore_scale(ops.add(
        ops.add(
            ops.cwise_mul(
                ops.ts_std(upper_shadow, 20),
                ops.neg(ops.ts_std(amount, 30))
            ),
            ops.ts_regression_residual(upper_shadow, high_low_range, 50)
        ),
        ops.ts_corr(foreign_ownership_pct, body, 8)
    ))
    
    return alpha


def compute_low_vol_alphas(data):
    """저변동성 알파들 계산"""
    atr_ratio = data['atr_ratio']
    volume = data['volume']
    high_low_range = data['high_low_range']
    
    return {
        'atr_vol': ops.neg(ops.ts_mean(ops.ts_std(atr_ratio, 60), 15)),
        'volume_std': ops.neg(ops.ts_std(volume, 75)),
        'hl_range': ops.neg(ops.ts_mean(high_low_range, 120)),
    }


def evaluate_alpha(alpha, forward_returns, purge_gap=20):
    """알파 IC/IR 평가"""
    n_days = len(alpha)
    train_end_idx = int(n_days * 0.7)
    
    try:
        alpha = pd.DataFrame(alpha).replace([np.inf, -np.inf], np.nan)
        
        train_alpha = alpha.iloc[:train_end_idx - purge_gap]
        train_fwd = forward_returns.iloc[:train_end_idx - purge_gap]
        test_alpha = alpha.iloc[train_end_idx:]
        test_fwd = forward_returns.iloc[train_end_idx:]
        
        def calc_ics(a, f):
            ics = []
            for i in range(len(a)):
                row_a = a.iloc[i].dropna()
                row_f = f.iloc[i].dropna()
                common = row_a.index.intersection(row_f.index)
                if len(common) >= 30:
                    ic, _ = spearmanr(row_a[common], row_f[common])
                    if not np.isnan(ic):
                        ics.append(ic)
            return ics
        
        train_ics = calc_ics(train_alpha, train_fwd)
        test_ics = calc_ics(test_alpha, test_fwd)
        
        if len(train_ics) < 10 or len(test_ics) < 5:
            return None
        
        return {
            'train_ic': np.mean(train_ics),
            'train_ir': np.mean(train_ics) / (np.std(train_ics) + 1e-8),
            'test_ic': np.mean(test_ics),
            'test_ir': np.mean(test_ics) / (np.std(test_ics) + 1e-8),
        }
        
    except Exception as e:
        print(f"   Error: {e}")
        return None


def main():
    print("=" * 70)
    print("🔗 Alpha Combination Experiment")
    print("   기존 최고 알파 + 저변동성 알파 결합")
    print("=" * 70)
    
    # 데이터 로드
    data = load_market_data()
    forward_returns = data['close'].shift(-FORWARD_DAYS) / data['close'] - 1
    
    # 기존 알파 계산
    print("\n📊 알파 계산 중...")
    best_alpha = compute_best_alpha(data)
    low_vol_alphas = compute_low_vol_alphas(data)
    
    # 기존 알파 평가
    print("\n📊 기존 알파 평가...")
    best_result = evaluate_alpha(best_alpha, forward_returns)
    if best_result:
        print(f"   기존 알파: Train IC={best_result['train_ic']:.4f}, Test IC={best_result['test_ic']:.4f}, IR={best_result['train_ir']:.2f}")
    else:
        print("   ❌ 기존 알파 평가 실패")
        return
    
    # 저변동성 알파 평가
    print("\n📊 저변동성 알파 평가...")
    low_vol_results = {}
    for name, alpha in low_vol_alphas.items():
        result = evaluate_alpha(alpha, forward_returns)
        if result:
            low_vol_results[name] = result
            print(f"   {name}: Train IC={result['train_ic']:.4f}, Test IC={result['test_ic']:.4f}")
    
    # 결합 실험
    print("\n" + "=" * 70)
    print("🧪 결합 실험")
    print("=" * 70)
    
    combinations = []
    
    for name, low_vol_alpha in low_vol_alphas.items():
        # 1. 단순 합 (zscore 후)
        combined = ops.add(
            ops.zscore_scale(best_alpha),
            ops.zscore_scale(low_vol_alpha)
        )
        result = evaluate_alpha(combined, forward_returns)
        if result:
            combinations.append({
                'name': f'zscore_add_{name}',
                'desc': f'zscore(기존) + zscore({name})',
                'alpha': combined,
                **result
            })
            print(f"  zscore(기존) + zscore({name}): Test IC={result['test_ic']:.4f}")
        
        # 2. 기존 가중치 2배
        combined = ops.add(
            ops.cwise_mul(ops.zscore_scale(best_alpha), 2),
            ops.zscore_scale(low_vol_alpha)
        )
        result = evaluate_alpha(combined, forward_returns)
        if result:
            combinations.append({
                'name': f'weighted_2to1_{name}',
                'desc': f'2×zscore(기존) + zscore({name})',
                'alpha': combined,
                **result
            })
            print(f"  2×zscore(기존) + zscore({name}): Test IC={result['test_ic']:.4f}")
        
        # 3. 저변동성 가중치 2배
        combined = ops.add(
            ops.zscore_scale(best_alpha),
            ops.cwise_mul(ops.zscore_scale(low_vol_alpha), 2)
        )
        result = evaluate_alpha(combined, forward_returns)
        if result:
            combinations.append({
                'name': f'weighted_1to2_{name}',
                'desc': f'zscore(기존) + 2×zscore({name})',
                'alpha': combined,
                **result
            })
            print(f"  zscore(기존) + 2×zscore({name}): Test IC={result['test_ic']:.4f}")
        
        # 4. 곱 (rank)
        combined = ops.cwise_mul(
            ops.normed_rank(best_alpha),
            ops.normed_rank(low_vol_alpha)
        )
        result = evaluate_alpha(combined, forward_returns)
        if result:
            combinations.append({
                'name': f'mul_rank_{name}',
                'desc': f'rank(기존) × rank({name})',
                'alpha': combined,
                **result
            })
            print(f"  rank(기존) × rank({name}): Test IC={result['test_ic']:.4f}")
    
    # 3개 저변동성 합산 + 기존
    all_low_vol = ops.add(
        ops.add(
            ops.zscore_scale(low_vol_alphas['atr_vol']),
            ops.zscore_scale(low_vol_alphas['volume_std'])
        ),
        ops.zscore_scale(low_vol_alphas['hl_range'])
    )
    combined = ops.add(ops.zscore_scale(best_alpha), all_low_vol)
    result = evaluate_alpha(combined, forward_returns)
    if result:
        combinations.append({
            'name': 'best_plus_all_lowvol',
            'desc': '기존 + (3개 저변동성 합)',
            'alpha': combined,
            **result
        })
        print(f"  기존 + (3개 저변동성 합): Test IC={result['test_ic']:.4f}")
    
    # 정렬
    combinations.sort(key=lambda x: x['test_ic'], reverse=True)
    
    # 최종 결과
    print("\n" + "=" * 70)
    print("🏆 FINAL RESULTS - Best Combinations")
    print("=" * 70)
    
    print(f"\n📌 기준선: 기존 알파 Test IC = {best_result['test_ic']:.4f}")
    
    for i, r in enumerate(combinations[:10]):
        improvement = (r['test_ic'] - best_result['test_ic']) / best_result['test_ic'] * 100
        marker = "⬆️" if improvement > 0 else "⬇️"
        print(f"\n{i+1}. {r['desc']}")
        print(f"   Test IC={r['test_ic']:.4f}, Train IC={r['train_ic']:.4f}, IR={r['train_ir']:.2f}")
        print(f"   {marker} {improvement:+.1f}% vs 기존")
    
    # 최고 결합 찾기
    if combinations:
        best_combo = combinations[0]
        if best_combo['test_ic'] > best_result['test_ic']:
            print(f"\n✨ BEST COMBINATION FOUND!")
            print(f"   {best_combo['desc']}")
            print(f"   Test IC: {best_combo['test_ic']:.4f}")
            improvement = (best_combo['test_ic'] - best_result['test_ic']) / best_result['test_ic'] * 100
            print(f"   기존 대비 +{improvement:.1f}%")
        else:
            print(f"\n⚠️ 기존 알파가 여전히 최고 (Test IC={best_result['test_ic']:.4f})")
            print(f"   결합 최고: {best_combo['desc']} (Test IC={best_combo['test_ic']:.4f})")


if __name__ == "__main__":
    main()
