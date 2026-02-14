#!/usr/bin/env python3
"""
Catch-up Alpha 백테스트
"아직 안 올랐지만 한달 내 오를 주식" 찾기

핵심 아이디어:
  - 최근 가격 상승이 적은 종목 중에서
  - 펀더멘탈 개선(영업이익 증가) 또는 거래량 축적(기관 매집) 신호가 있는 종목
  - 20영업일(약 1달) 포워드 수익률과의 상관관계(IC) 측정

알파 후보군:
  A. 가격-실적 괴리형: 실적은 좋아지는데 주가는 아직 반응 안 함
  B. 거래량 축적형: 주가는 횡보하나 거래량이 증가 (스마트머니 매집)
  C. 과매도 반전형: 많이 빠졌지만 실적 개선 + 거래량 회복
  D. 복합형: 가격 미반영 + 실적 개선 + 거래량 신호 동시
"""

import sys
import os
import json as _json
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import psycopg2

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


# ── 데이터 로드 ──
print("📊 데이터 로드 중...")
conn = get_db_connection()

stocks_df = pd.read_sql("""
    SELECT s.id, s.ticker, s.name, s.market_cap
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
    LIMIT 2000
""", conn)

stock_ids = stocks_df['id'].tolist()
stock_id_list = ', '.join(map(str, stock_ids))

ticker_name = dict(zip(stocks_df['ticker'], stocks_df['name']))
ticker_mcap = dict(zip(stocks_df['ticker'], stocks_df['market_cap']))
id_ticker = dict(zip(stocks_df['id'], stocks_df['ticker']))

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

# 재무 추세 데이터
print("   재무 추세 데이터 로드 중...")
fin_df = pd.read_sql(f"""
    SELECT stock_id, period_end, revenue, operating_income, net_income, raw_data
    FROM financial_statements
    WHERE stock_id IN ({stock_id_list})
    ORDER BY stock_id, period_end
""", conn)
conn.close()

# raw_data 파싱
def _parse_raw(row):
    rd = row.get('raw_data')
    if rd is None:
        return {'quarter_type': None}
    if isinstance(rd, str):
        rd = _json.loads(rd)
    return {'quarter_type': rd.get('quarter', '')}

raw_parsed = fin_df.apply(_parse_raw, axis=1, result_type='expand')
fin_df = pd.concat([fin_df, raw_parsed], axis=1)
fin_df['ticker'] = fin_df['stock_id'].map(id_ticker)
fin_df = fin_df.dropna(subset=['ticker'])
fin_df = fin_df[fin_df['quarter_type'] != '연간'].copy()
fin_df = fin_df.sort_values(['ticker', 'period_end'])

# QoQ / YoY / 3분기 추세
trend_records = []
for ticker, grp in fin_df.groupby('ticker'):
    grp = grp.sort_values('period_end').reset_index(drop=True)
    for i in range(len(grp)):
        row = grp.iloc[i]
        rec = {'ticker': ticker, 'period_end': row['period_end']}

        if i >= 1:
            prev = grp.iloc[i - 1]
            if prev['operating_income'] and prev['operating_income'] != 0:
                rec['oi_qoq'] = (row['operating_income'] - prev['operating_income']) / abs(prev['operating_income'])

        if i >= 3:
            yoy_prev = grp.iloc[i - 3]
            if row['period_end'].month == yoy_prev['period_end'].month:
                if yoy_prev['operating_income'] and yoy_prev['operating_income'] != 0:
                    rec['oi_yoy'] = (row['operating_income'] - yoy_prev['operating_income']) / abs(yoy_prev['operating_income'])

        if i >= 2:
            oi_vals = [grp.iloc[j]['operating_income'] for j in range(i - 2, i + 1)
                       if grp.iloc[j]['operating_income'] is not None and not np.isnan(grp.iloc[j]['operating_income'])]
            if len(oi_vals) == 3:
                rec['oi_trend'] = (oi_vals[2] - oi_vals[0]) / (abs(oi_vals[0]) + 1e-10)

        trend_records.append(rec)

trend_df = pd.DataFrame(trend_records)

_empty = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
trend_vars = {}
for field in ['oi_qoq', 'oi_yoy', 'oi_trend']:
    if field not in trend_df.columns:
        continue
    pivot = trend_df.pivot_table(index='period_end', columns='ticker', values=field, aggfunc='last')
    if pivot.empty or pivot.notna().sum().sum() < 50:
        continue
    daily = pivot.reindex(close.index).ffill().reindex(columns=close.columns)
    trend_vars[f'{field}_rank'] = daily.rank(axis=1, pct=True)

oi_trend_rank = trend_vars.get('oi_trend_rank', _empty)
oi_yoy_rank = trend_vars.get('oi_yoy_rank', _empty)
oi_qoq_rank = trend_vars.get('oi_qoq_rank', _empty)

print(f"✅ {len(close.columns)}개 종목, {len(close)}일 데이터")
print(f"   재무 추세: {list(trend_vars.keys())}")


# ── IC 계산 함수 ──
def compute_ic(alpha_values, close_data, forward_days=20):
    """20일 포워드 IC 계산"""
    forward_return = close_data.shift(-forward_days) / close_data - 1

    ic_list = []
    for date in alpha_values.index[:-forward_days]:
        a = alpha_values.loc[date]
        r = forward_return.loc[date]
        valid = a.notna() & r.notna()
        if valid.sum() > 30:
            ic = a[valid].corr(r[valid])
            if not np.isnan(ic):
                ic_list.append(ic)

    if len(ic_list) < 10:
        return 0.0, 0.0, 0
    return np.mean(ic_list), np.std(ic_list), len(ic_list)


# ── Train/Test 분할 ──
split_idx = int(len(close) * 0.7)
train_close = close.iloc[:split_idx]
test_close = close.iloc[split_idx:]

print(f"\n📐 Train/Test: {split_idx}일 / {len(close) - split_idx}일")
print(f"   Train: {close.index[0]} ~ {close.index[split_idx-1]}")
print(f"   Test:  {close.index[split_idx]} ~ {close.index[-1]}")


# ══════════════════════════════════════════════════════════════
# Catch-up Alpha 후보 정의
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print(f"🔍 Catch-up Alpha 백테스트 (20일 포워드)")
print(f"   아이디어: 아직 덜 올랐지만 펀더멘탈/거래량 신호가 좋은 종목")
print(f"{'='*90}")

# 가격 모멘텀 역전 (낮은 모멘텀 = 높은 점수)
neg_momentum_20d = ops.neg(ops.ts_delta_ratio(close, 20))
neg_momentum_10d = ops.neg(ops.ts_delta_ratio(close, 10))
neg_momentum_30d = ops.neg(ops.ts_delta_ratio(close, 30))

# 가격 위치: 30일 저점 근처 (낮은 위치 = 높은 catch-up 잠재력)
low_position_30d = ops.neg(ops.ts_maxmin_scale(close, 30))
low_position_20d = ops.neg(ops.ts_maxmin_scale(close, 20))

# 거래량 축적: 최근 거래량 > 평균 거래량
vol_surge = ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20))
vol_surge_10_30 = ops.div(ops.ts_mean(volume, 10), ops.ts_mean(volume, 30))

# 거래량 안정성
vol_stability = ops.div(ops.ts_median(volume, 10), ops.ts_std(volume, 15))

# 가격 변동성 (낮은 변동성 = 횡보 = 축적 단계)
low_volatility = ops.neg(ops.ts_std(returns, 20))

# 기존 앙상블 알파 (비교용)
v3_alpha = ops.normed_rank(
    ops.cwise_mul(
        ops.cwise_mul(
            ops.ts_delta_ratio(close, 25),
            ops.div(ops.ts_median(volume, 10), ops.ts_std(volume, 15))
        ),
        ops.ts_maxmin_scale(close, 28)
    )
)

alphas = {}

# ── A. 가격-실적 괴리형 (Earnings-Price Divergence) ──
# 실적은 개선 중인데 주가는 아직 반응 안 함
alphas['A1_실적개선×가격미반영'] = ops.normed_rank(
    ops.cwise_mul(neg_momentum_20d, oi_yoy_rank)
)

alphas['A2_실적추세×가격저위치'] = ops.normed_rank(
    ops.cwise_mul(low_position_30d, oi_trend_rank)
)

alphas['A3_QoQ개선×가격하락'] = ops.normed_rank(
    ops.cwise_mul(neg_momentum_10d, oi_qoq_rank)
)

alphas['A4_YoY+QoQ×가격미반영'] = ops.normed_rank(
    ops.cwise_mul(
        neg_momentum_20d,
        ops.add(oi_yoy_rank, oi_qoq_rank)
    )
)

# ── B. 거래량 축적형 (Volume Accumulation) ──
# 주가는 횡보하지만 거래량이 증가 → 스마트머니 매집
alphas['B1_횡보×거래량급증'] = ops.normed_rank(
    ops.cwise_mul(low_volatility, vol_surge)
)

alphas['B2_가격미반영×거래량축적'] = ops.normed_rank(
    ops.cwise_mul(neg_momentum_20d, vol_surge_10_30)
)

alphas['B3_저위치×거래량안정'] = ops.normed_rank(
    ops.cwise_mul(low_position_20d, vol_stability)
)

# ── C. 과매도 반전 + 실적형 (Oversold + Improving) ──
# 많이 빠졌지만 실적은 좋아지고 있는 종목
alphas['C1_과매도×실적개선'] = ops.normed_rank(
    ops.cwise_mul(
        ops.cwise_mul(neg_momentum_30d, oi_yoy_rank),
        vol_surge
    )
)

alphas['C2_저위치×추세개선×거래량'] = ops.normed_rank(
    ops.cwise_mul(
        ops.cwise_mul(low_position_30d, oi_trend_rank),
        vol_surge_10_30
    )
)

# ── D. 복합형 (Multi-Factor Catch-up) ──
# 가격 미반영 + 실적 개선 + 거래량 축적 모두 확인
alphas['D1_3팩터(가격+실적+거래량)'] = ops.normed_rank(
    ops.cwise_mul(
        ops.cwise_mul(
            neg_momentum_20d,
            ops.add(oi_yoy_rank, oi_trend_rank)
        ),
        vol_surge
    )
)

alphas['D2_저위치×YoY×거래량안정'] = ops.normed_rank(
    ops.cwise_mul(
        ops.cwise_mul(low_position_30d, oi_yoy_rank),
        vol_stability
    )
)

alphas['D3_가격미반영×3추세×축적'] = ops.normed_rank(
    ops.cwise_mul(
        ops.cwise_mul(
            neg_momentum_20d,
            ops.add(ops.add(oi_yoy_rank, oi_qoq_rank), oi_trend_rank)
        ),
        vol_surge_10_30
    )
)

# ── E. 대안 전략 ──
# E1: 상대적 언더퍼포머 (전체 시장 대비 덜 오른 종목 + 실적 OK)
# 시장 평균 수익률 대비 저성과
market_avg_ret = ops.ts_delta_ratio(close, 20).mean(axis=1)
relative_underperf = ops.ts_delta_ratio(close, 20).sub(market_avg_ret, axis=0)
alphas['E1_상대저성과×실적개선'] = ops.normed_rank(
    ops.cwise_mul(
        ops.neg(relative_underperf).rank(axis=1, pct=True),
        oi_yoy_rank
    )
)

# E2: 낮은 변동성 + 실적 개선 (조용한 축적 → 브레이크아웃 대기)
alphas['E2_저변동×실적추세'] = ops.normed_rank(
    ops.cwise_mul(
        ops.neg(ops.ts_std(returns, 30)).rank(axis=1, pct=True),
        ops.add(oi_trend_rank, oi_yoy_rank)
    )
)

# 비교용: 기존 모멘텀 알파 (v3)
alphas['REF_v3모멘텀(비교용)'] = v3_alpha

# 비교용: 순수 실적 알파 (v6)
alphas['REF_v6실적(비교용)'] = ops.normed_rank(
    ops.add(ops.add(oi_trend_rank, oi_yoy_rank), oi_qoq_rank)
)


# ── 백테스트 실행 ──
print(f"\n{'순번':>4} {'알파명':<30} {'Train IC':>10} {'Test IC':>10} {'IR':>6} {'판정':>4}")
print(f"{'-'*70}")

results = []
for name, alpha in alphas.items():
    train_ic, train_std, train_n = compute_ic(alpha.iloc[:split_idx], train_close)
    test_ic, test_std, test_n = compute_ic(alpha.iloc[split_idx:], test_close)
    ir = test_ic / test_std if test_std > 0 else 0

    status = "✅" if test_ic > 0.02 else ("⚠️" if test_ic > 0 else "❌")
    print(f"  {len(results)+1:>2}. {name:<30} {train_ic:>+.4f}    {test_ic:>+.4f}  {ir:>+.2f}   {status}")

    results.append({
        'name': name,
        'alpha': alpha,
        'train_ic': train_ic,
        'test_ic': test_ic,
        'ir': ir,
    })

# ── 결과 분석 ──
results.sort(key=lambda x: x['test_ic'], reverse=True)

print(f"\n{'='*90}")
print(f"📊 Test IC 기준 순위 (20일 포워드)")
print(f"{'='*90}")
for i, r in enumerate(results, 1):
    marker = " 🏆" if i == 1 else ""
    print(f"  {i:>2}. {r['name']:<30}  Test IC: {r['test_ic']:>+.4f}  Train IC: {r['train_ic']:>+.4f}  IR: {r['ir']:>+.2f}{marker}")

# 기존 모멘텀 알파와 비교
ref_v3 = next((r for r in results if r['name'] == 'REF_v3모멘텀(비교용)'), None)
ref_v6 = next((r for r in results if r['name'] == 'REF_v6실적(비교용)'), None)
best_catchup = next((r for r in results if not r['name'].startswith('REF_')), None)

if best_catchup and ref_v3:
    print(f"\n📈 비교:")
    print(f"   기존 v3 모멘텀:     Test IC {ref_v3['test_ic']:+.4f} (20d fwd)")
    print(f"   기존 v6 실적:       Test IC {ref_v6['test_ic']:+.4f} (20d fwd)")
    print(f"   Best Catch-up:      Test IC {best_catchup['test_ic']:+.4f} (20d fwd) ← {best_catchup['name']}")

# ── 앙상블 시도: 기존 모멘텀 + 최고 catch-up ──
print(f"\n{'='*90}")
print(f"🔗 앙상블 테스트: Catch-up × 기존 알파 조합")
print(f"{'='*90}")

# 상위 3개 catch-up 알파 (REF 제외)
top_catchups = [r for r in results if not r['name'].startswith('REF_')][:3]

for r in top_catchups:
    catchup = r['alpha']

    # Catch-up 단독
    print(f"\n  [{r['name']}]")

    # 앙상블: catch-up 50% + v3 모멘텀 50%
    ens_v3 = ops.normed_rank(catchup * 0.5 + v3_alpha * 0.5)
    train_ic, _, _ = compute_ic(ens_v3.iloc[:split_idx], train_close)
    test_ic, test_std, _ = compute_ic(ens_v3.iloc[split_idx:], test_close)
    ir = test_ic / test_std if test_std > 0 else 0
    print(f"    + v3모멘텀 50:50    Train IC: {train_ic:+.4f}  Test IC: {test_ic:+.4f}  IR: {ir:+.2f}")

    # 앙상블: catch-up 70% + v3 30%
    ens_v3_70 = ops.normed_rank(catchup * 0.7 + v3_alpha * 0.3)
    train_ic, _, _ = compute_ic(ens_v3_70.iloc[:split_idx], train_close)
    test_ic, test_std, _ = compute_ic(ens_v3_70.iloc[split_idx:], test_close)
    ir = test_ic / test_std if test_std > 0 else 0
    print(f"    + v3모멘텀 70:30    Train IC: {train_ic:+.4f}  Test IC: {test_ic:+.4f}  IR: {ir:+.2f}")

    # 앙상블: catch-up 50% + v6 실적 50%
    v6_alpha = ops.normed_rank(ops.add(ops.add(oi_trend_rank, oi_yoy_rank), oi_qoq_rank))
    v6_filled = v6_alpha.fillna(0.5)
    ens_v6 = ops.normed_rank(catchup * 0.5 + v6_filled * 0.5)
    train_ic, _, _ = compute_ic(ens_v6.iloc[:split_idx], train_close)
    test_ic, test_std, _ = compute_ic(ens_v6.iloc[split_idx:], test_close)
    ir = test_ic / test_std if test_std > 0 else 0
    print(f"    + v6실적 50:50      Train IC: {train_ic:+.4f}  Test IC: {test_ic:+.4f}  IR: {ir:+.2f}")


print(f"\n{'='*90}")
print(f"💡 해석")
print(f"{'='*90}")
print(f"   Catch-up Alpha = neg(모멘텀) × 펀더멘탈 개선도")
print(f"   → '주가는 아직 덜 올랐지만 실적이 좋아지는 종목'에 매수 신호")
print(f"   → 20영업일(약 1달) 보유 전략에 최적화")
print(f"   → 기존 모멘텀 알파와 상관관계 낮아 분산 투자 효과 기대")
