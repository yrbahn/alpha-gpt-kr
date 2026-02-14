#!/usr/bin/env python3
"""
Catch-up Alpha v2: "곧 오를 종목" 재설계
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

v1 교훈: 순수 역모멘텀(덜 오른 것을 사라)은 한국시장에서 작동 안 함 (IC < 0)
       → 모멘텀이 강하게 지속됨

v2 접근: "아직 덜 올랐지만 곧 오를" = 다음 3가지 패턴
  1. 초기 모멘텀: 최근 단기(5일)는 올랐지만 중기(30일)는 아직 → 모멘텀 초기 단계
  2. 횡보 후 돌파: 변동성 낮았는데(횡보) 최근 급등 시작 → 브레이크아웃
  3. 거래량 선행: 가격보다 거래량이 먼저 증가 → 스마트머니 선매수
  4. 모멘텀 가속: 최근 모멘텀이 이전 대비 가속 → 추세 강화 초기

+ 실적 개선 필터를 결합하여 "진짜 오를 이유가 있는" 종목만 선별
+ 20영업일(1달) 포워드 IC 백테스트
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

# 밸류에이션 필터
print("   밸류에이션 필터 계산 중...")
valuation = {}
for ticker, grp in fin_df.groupby('ticker'):
    grp = grp.sort_values('period_end')
    recent = grp.tail(4)
    if len(recent) < 2:
        continue
    trailing_oi = recent['operating_income'].sum()
    trailing_ni = recent['net_income'].sum()
    mcap = ticker_mcap.get(ticker, 0)
    per = mcap / trailing_ni if mcap and mcap > 0 and trailing_ni and trailing_ni > 0 else np.nan
    valuation[ticker] = {
        'trailing_oi': trailing_oi,
        'trailing_ni': trailing_ni,
        'per': per,
    }

FILTER_PER_MAX = 50
filtered_out = set()
for ticker, v in valuation.items():
    reasons = []
    if v['trailing_ni'] is None or v['trailing_ni'] <= 0:
        reasons.append("순이익 적자")
    elif v['per'] is not None and v['per'] > FILTER_PER_MAX:
        reasons.append(f"PER {v['per']:.1f}x")
    if v['trailing_oi'] is not None and v['trailing_oi'] <= 0:
        reasons.append("영업적자")
    if reasons:
        filtered_out.add(ticker)
no_data_tickers = set(close.columns) - set(valuation.keys())
exclude_tickers = filtered_out | no_data_tickers

print(f"✅ {len(close.columns)}개 종목, {len(close)}일 데이터")
print(f"   PER ≤ {FILTER_PER_MAX}x 필터: {len(filtered_out)}개 제외, 재무없음: {len(no_data_tickers)}개")


# ── IC 계산 ──
def compute_ic(alpha_values, close_data, forward_days=20):
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


# Train/Test 분할
split_idx = int(len(close) * 0.7)
train_close = close.iloc[:split_idx]
test_close = close.iloc[split_idx:]

print(f"\n📐 Train/Test: {split_idx}일 / {len(close) - split_idx}일")
print(f"   Train: {close.index[0]} ~ {close.index[split_idx-1]}")
print(f"   Test:  {close.index[split_idx]} ~ {close.index[-1]}")


# ══════════════════════════════════════════════════════════════
# v2 Catch-up Alpha: "막 올라가기 시작하는" 패턴
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print(f"🚀 Catch-up v2: '곧 오를 종목' 알파 백테스트 (20일 포워드)")
print(f"   v1 교훈: 순수 역모멘텀 ❌ → 초기모멘텀/돌파/가속 패턴으로 재설계")
print(f"{'='*90}")

alphas = {}

# ═══ 1. 초기 모멘텀 (Early Momentum) ═══
# 단기(5d)는 올랐지만 중기(30d)는 아직 → 모멘텀 시작 단계
mom_5d = ops.ts_delta_ratio(close, 5)
mom_10d = ops.ts_delta_ratio(close, 10)
mom_20d = ops.ts_delta_ratio(close, 20)
mom_30d = ops.ts_delta_ratio(close, 30)

# 단기 모멘텀 - 장기 모멘텀 = "최근에 시작된" 모멘텀
alphas['1a_초기모멘텀(5d-30d)'] = ops.normed_rank(
    ops.minus(
        ops.ts_delta_ratio(close, 5),
        ops.ts_delta_ratio(close, 30)
    )
)

# 단기 상승 + 장기 아직 낮은 위치 → "바닥에서 막 올라오는"
alphas['1b_바닥탈출(5d↑+30d저위치)'] = ops.normed_rank(
    ops.cwise_mul(
        ops.ts_delta_ratio(close, 5),
        ops.neg(ops.ts_maxmin_scale(close, 30))
    )
)

# 초기 모멘텀 + 실적 개선
alphas['1c_초기모멘텀×실적'] = ops.normed_rank(
    ops.cwise_mul(
        ops.minus(ops.ts_delta_ratio(close, 5), ops.ts_delta_ratio(close, 25)),
        oi_yoy_rank
    )
)

# ═══ 2. 횡보 후 돌파 (Breakout from Consolidation) ═══
# 과거 변동성 낮았는데(횡보) + 최근 상승 시작
past_vol = ops.ts_std(returns, 30)  # 과거 30일 변동성
recent_move = ops.ts_delta_ratio(close, 5)  # 최근 5일 움직임

alphas['2a_횡보후돌파'] = ops.normed_rank(
    ops.cwise_mul(
        ops.neg(past_vol),  # 변동성 낮았던 (횡보)
        recent_move          # × 최근 상승
    )
)

# 횡보 후 돌파 + 거래량 확인
alphas['2b_돌파×거래량폭증'] = ops.normed_rank(
    ops.cwise_mul(
        ops.cwise_mul(
            ops.neg(ops.ts_std(returns, 20)),
            ops.ts_delta_ratio(close, 5)
        ),
        ops.div(ops.ts_mean(volume, 3), ops.ts_mean(volume, 20))  # 3일 거래량 / 20일 평균
    )
)

# 횡보 후 돌파 + 실적
alphas['2c_돌파×실적개선'] = ops.normed_rank(
    ops.cwise_mul(
        ops.cwise_mul(
            ops.neg(ops.ts_std(returns, 25)),
            ops.ts_delta_ratio(close, 7)
        ),
        oi_yoy_rank
    )
)

# ═══ 3. 거래량 선행 (Volume Leads Price) ═══
# 거래량이 먼저 증가 → 가격 아직 안 올랐지만 곧 오를 신호
vol_momentum = ops.ts_delta_ratio(volume, 10)  # 거래량 증가율
price_momentum = ops.ts_delta_ratio(close, 10)  # 가격 변화율

# 거래량↑ 가격 아직 → 괴리
alphas['3a_거래량선행'] = ops.normed_rank(
    ops.minus(
        ops.normed_rank(ops.ts_delta_ratio(volume, 10)),
        ops.normed_rank(ops.ts_delta_ratio(close, 10))
    )
)

# 거래량 최근 급증 + 가격 아직 저위치
alphas['3b_거래량급증×가격저위치'] = ops.normed_rank(
    ops.cwise_mul(
        ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 30)),
        ops.neg(ops.ts_maxmin_scale(close, 20))
    )
)

# 거래량 선행 + 실적 확인
alphas['3c_거래량선행×실적'] = ops.normed_rank(
    ops.cwise_mul(
        ops.minus(
            ops.normed_rank(ops.ts_delta_ratio(volume, 10)),
            ops.normed_rank(ops.ts_delta_ratio(close, 10))
        ),
        oi_yoy_rank
    )
)

# ═══ 4. 모멘텀 가속 (Momentum Acceleration) ═══
# 최근 모멘텀이 이전보다 강해지는 → 추세 초기/가속 단계
alphas['4a_모멘텀가속'] = ops.normed_rank(
    ops.minus(
        ops.ts_delta_ratio(close, 10),   # 최근 10일 수익률
        ops.ts_delta_ratio(close.shift(10), 10)  # 10~20일전 수익률
    )
)

# 가속 + 거래량 동반
alphas['4b_가속×거래량동반'] = ops.normed_rank(
    ops.cwise_mul(
        ops.minus(
            ops.ts_delta_ratio(close, 10),
            ops.ts_delta_ratio(close.shift(10), 10)
        ),
        ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20))
    )
)

# 가속 + 실적
alphas['4c_가속×실적개선'] = ops.normed_rank(
    ops.cwise_mul(
        ops.minus(
            ops.ts_delta_ratio(close, 10),
            ops.ts_delta_ratio(close.shift(10), 10)
        ),
        oi_yoy_rank
    )
)

# ═══ 5. 복합형 (Best Combinations) ═══
# 초기 모멘텀 + 거래량 확인 + 실적
alphas['5a_초기모멘텀×거래량×실적'] = ops.normed_rank(
    ops.cwise_mul(
        ops.cwise_mul(
            ops.minus(ops.ts_delta_ratio(close, 5), ops.ts_delta_ratio(close, 25)),
            ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20))
        ),
        oi_yoy_rank
    )
)

# 돌파 + 거래량 + 실적 3팩터
alphas['5b_돌파×거래량×실적'] = ops.normed_rank(
    ops.cwise_mul(
        ops.cwise_mul(
            ops.cwise_mul(
                ops.neg(ops.ts_std(returns, 20)),
                ops.ts_delta_ratio(close, 5)
            ),
            ops.div(ops.ts_mean(volume, 3), ops.ts_mean(volume, 20))
        ),
        oi_yoy_rank
    )
)

# 모멘텀 가속 + 거래량 + 실적
alphas['5c_가속×거래량×실적'] = ops.normed_rank(
    ops.cwise_mul(
        ops.cwise_mul(
            ops.minus(ops.ts_delta_ratio(close, 10), ops.ts_delta_ratio(close.shift(10), 10)),
            ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20))
        ),
        oi_yoy_rank
    )
)

# ═══ 6. 기존 v3 개선형 (20일 최적화) ═══
# v3를 20일에 맞춰 재조정
alphas['6a_v3_20d최적화'] = ops.normed_rank(
    ops.cwise_mul(
        ops.cwise_mul(
            ops.ts_delta_ratio(close, 20),
            ops.div(ops.ts_median(volume, 10), ops.ts_std(volume, 15))
        ),
        ops.ts_maxmin_scale(close, 25)
    )
)

# v3 + 실적 (20일용)
alphas['6b_v3_20d×실적'] = ops.normed_rank(
    ops.cwise_mul(
        ops.cwise_mul(
            ops.cwise_mul(
                ops.ts_delta_ratio(close, 20),
                ops.div(ops.ts_median(volume, 10), ops.ts_std(volume, 15))
            ),
            ops.ts_maxmin_scale(close, 25)
        ),
        oi_yoy_rank
    )
)

# ═══ 비교용 ═══
alphas['REF_v3모멘텀'] = ops.normed_rank(
    ops.cwise_mul(
        ops.cwise_mul(
            ops.ts_delta_ratio(close, 25),
            ops.div(ops.ts_median(volume, 10), ops.ts_std(volume, 15))
        ),
        ops.ts_maxmin_scale(close, 28)
    )
)

alphas['REF_v6실적'] = ops.normed_rank(
    ops.add(ops.add(oi_trend_rank, oi_yoy_rank), oi_qoq_rank)
)

alphas['REF_앙상블v3v6'] = ops.normed_rank(
    alphas['REF_v3모멘텀'] * 0.5 +
    ops.normed_rank(ops.add(ops.add(oi_trend_rank, oi_yoy_rank), oi_qoq_rank)).fillna(0.5) * 0.5
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
    marker = " 🏆" if i == 1 else (" ⭐" if i <= 3 else "")
    ref = " [REF]" if r['name'].startswith('REF_') else ""
    print(f"  {i:>2}. {r['name']:<30}  Test IC: {r['test_ic']:>+.4f}  Train: {r['train_ic']:>+.4f}  IR: {r['ir']:>+.2f}{marker}{ref}")

# ── 상위 알파로 종목 선정 ──
# REF 제외한 best catch-up 알파 찾기
non_ref = [r for r in results if not r['name'].startswith('REF_') and r['test_ic'] > 0]

if non_ref:
    best = non_ref[0]
    print(f"\n{'='*90}")
    print(f"🏆 최적 Catch-up 알파: {best['name']}")
    print(f"   Train IC: {best['train_ic']:+.4f} / Test IC: {best['test_ic']:+.4f} / IR: {best['ir']:+.2f}")
    print(f"{'='*90}")

    # 최신 날짜 기준 종목 선정
    latest_date = close.index[-1]
    alpha_scores = best['alpha'].loc[latest_date].dropna().sort_values(ascending=False)

    # 밸류에이션 필터
    filtered_scores = alpha_scores[~alpha_scores.index.isin(exclude_tickers)]

    print(f"\n📋 Catch-up 상위 10종목 (기준일: {latest_date})")
    print(f"   필터: PER ≤ {FILTER_PER_MAX}x, 영업흑자, 순이익흑자")
    print(f"   유니버스: {len(alpha_scores)}종목 → 필터 통과 {len(filtered_scores)}종목")
    print(f"\n{'순위':>4} {'종목코드':<10} {'종목명':<16} {'Catch-up':>10} {'20d수익률':>10} {'PER':>6} {'NI(억)':>8}")
    print(f"{'-'*75}")

    for i, (ticker, score) in enumerate(filtered_scores.head(10).items(), 1):
        name = ticker_name.get(ticker, '?')
        price = close.loc[latest_date, ticker]
        # 최근 20일 수익률
        if len(close) > 20:
            ret_20d = (close.loc[latest_date, ticker] / close.iloc[-20][ticker] - 1) * 100
        else:
            ret_20d = 0
        v = valuation.get(ticker, {})
        per = v.get('per', np.nan)
        ni = v.get('trailing_ni', 0)
        ni_억 = ni / 1e8 if ni else 0
        per_s = f"{per:.1f}x" if per and not np.isnan(per) else "  -"
        print(f"  {i:>2}. {ticker:<10} {name:<16} {score:.4f}    {ret_20d:>+.1f}%  {per_s:>6} {ni_억:>7,.0f}억")

    # 앙상블 시도: best catch-up + v3 + v6
    print(f"\n{'='*90}")
    print(f"🔗 앙상블: Catch-up + 기존 알파")
    print(f"{'='*90}")

    v3 = alphas['REF_v3모멘텀']
    v6 = alphas['REF_v6실적'].fillna(0.5)
    catchup = best['alpha']

    weights_to_test = [
        ('Catch 33% + v3 33% + v6 33%', 0.33, 0.33, 0.34),
        ('Catch 50% + v3 25% + v6 25%', 0.50, 0.25, 0.25),
        ('Catch 25% + v3 25% + v6 50%', 0.25, 0.25, 0.50),
        ('Catch 20% + v3 40% + v6 40%', 0.20, 0.40, 0.40),
    ]

    print(f"\n{'구성':<35} {'Train IC':>10} {'Test IC':>10} {'IR':>6}")
    print(f"{'-'*65}")

    for label, wc, w3, w6 in weights_to_test:
        ens = ops.normed_rank(catchup * wc + v3 * w3 + v6 * w6)
        train_ic, _, _ = compute_ic(ens.iloc[:split_idx], train_close)
        test_ic, test_std, _ = compute_ic(ens.iloc[split_idx:], test_close)
        ir = test_ic / test_std if test_std > 0 else 0
        print(f"  {label:<35} {train_ic:>+.4f}    {test_ic:>+.4f}  {ir:>+.2f}")

else:
    print(f"\n⚠️  양의 Test IC를 가진 Catch-up 알파가 없습니다.")
    print(f"   → 현재 시장은 강한 모멘텀 장세로, '덜 오른 것을 사는' 전략 자체가 불리합니다.")
    print(f"   → 기존 v3(모멘텀)+v6(실적) 앙상블을 유지하는 것이 최선입니다.")

    # 대안: 비교적 나은 것들 표시
    print(f"\n   비교적 IC가 높은 알파 (REF 제외):")
    non_ref_all = [r for r in results if not r['name'].startswith('REF_')]
    for r in non_ref_all[:5]:
        print(f"     {r['name']:<30}  Test IC: {r['test_ic']:>+.4f}")

print(f"\n{'='*90}")
print(f"💡 결론 및 해석")
print(f"{'='*90}")
print(f"   v1 역모멘텀: '덜 오른 것을 사라' → IC < 0 (실패)")
print(f"   v2 초기모멘텀: '막 올라가기 시작하는 것' → 위 결과 확인")
print(f"   v2 돌파형: '횡보 후 돌파 시작' → 위 결과 확인")
print(f"   v2 가속형: '모멘텀 가속되는 것' → 위 결과 확인")
print(f"   → 20영업일(약 1달) 리밸런싱 전략에 적합한 알파 선정")
