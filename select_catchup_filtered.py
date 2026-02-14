#!/usr/bin/env python3
"""
Catch-up 종목 선정 (이미 많이 오른 종목 제외)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

문제: 기존 알파는 모멘텀 비중이 높아 "이미 크게 오른" 종목을 계속 뽑음
해결:
  1) 장기 수익률(60d, 90d)로 "이미 많이 오른" 종목 제외
  2) 모멘텀 가속(최근 10d vs 이전 10d) + 실적 개선 알파 사용
  3) v6 재무추세 비중 확대

전략: "3개월간 상대적으로 덜 올랐지만, 최근 가속 + 실적 개선" 종목
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

# 추세 변수
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


# ══════════════════════════════════════════════════════════════
# 장기 수익률 계산
# ══════════════════════════════════════════════════════════════
latest_date = close.index[-1]

# 각 기간별 수익률
periods = {
    '20d': 20,
    '40d': 40,
    '60d': 60,
    '90d': 90,
    '120d': 120,
}
ret_by_period = {}
for label, days in periods.items():
    if len(close) > days:
        ret_by_period[label] = (close.iloc[-1] / close.iloc[-days] - 1) * 100
    else:
        ret_by_period[label] = pd.Series(0, index=close.columns)

# 모멘텀 가속도: 최근 10일 수익률 - 이전 10일 수익률
mom_recent_10d = (close.iloc[-1] / close.iloc[-10] - 1) * 100
mom_prev_10d = (close.iloc[-10] / close.iloc[-20] - 1) * 100
mom_accel = mom_recent_10d - mom_prev_10d

# 52주(250일) 최고가 대비 현재 위치
if len(close) > 250:
    high_250d = close.iloc[-250:].max()
    low_250d = close.iloc[-250:].min()
    position_52w = (close.iloc[-1] - low_250d) / (high_250d - low_250d + 1e-10) * 100
else:
    position_52w = pd.Series(50, index=close.columns)


# ══════════════════════════════════════════════════════════════
# 먼저 기존 상위 종목들의 장기 수익률 확인
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*110}")
print(f"📊 기존 상위 종목들의 장기 수익률 점검 (기준일: {latest_date})")
print(f"{'='*110}")

# 기존 앙상블 (select_top5.py와 동일)
v3_alpha = ops.normed_rank(
    ops.cwise_mul(
        ops.cwise_mul(
            ops.ts_delta_ratio(close, 25),
            ops.div(ops.ts_median(volume, 10), ops.ts_std(volume, 15))
        ),
        ops.ts_maxmin_scale(close, 28)
    )
)
v6_alpha = ops.normed_rank(
    ops.add(ops.add(oi_trend_rank, oi_yoy_rank), oi_qoq_rank)
)
v6_filled = v6_alpha.fillna(0.5)
old_ensemble = ops.normed_rank(v3_alpha * 0.5 + v6_filled * 0.5)
old_scores = old_ensemble.loc[latest_date].dropna().sort_values(ascending=False)
old_filtered = old_scores[~old_scores.index.isin(exclude_tickers)]

print(f"\n{'순위':>4} {'종목코드':<10} {'종목명':<14} {'현재가':>10} {'20d':>8} {'60d':>8} {'90d':>8} {'120d':>8} {'52W위치':>8} {'판정':<8}")
print(f"{'-'*100}")

for i, (ticker, score) in enumerate(old_filtered.head(10).items(), 1):
    name = ticker_name.get(ticker, '?')
    price = close.loc[latest_date, ticker]
    r20 = ret_by_period['20d'].get(ticker, 0)
    r60 = ret_by_period['60d'].get(ticker, 0)
    r90 = ret_by_period['90d'].get(ticker, 0)
    r120 = ret_by_period['120d'].get(ticker, 0)
    pos = position_52w.get(ticker, 50)

    # 판정: 60일 수익률 > 30% 또는 52주 위치 > 80%면 "과열"
    if r60 > 30 or pos > 80:
        verdict = "⚠️ 과열"
    elif r60 > 15:
        verdict = "🔶 상승중"
    else:
        verdict = "✅ 초기"

    print(f"  {i:>2}. {ticker:<10} {name:<14} {price:>10,.0f}원 {r20:>+6.1f}% {r60:>+6.1f}% {r90:>+6.1f}% {r120:>+6.1f}% {pos:>6.0f}%  {verdict}")


# ══════════════════════════════════════════════════════════════
# 새 전략: 가속×실적 알파 + "과열 종목 제외" 필터
# ══════════════════════════════════════════════════════════════

# 과열 필터: 60일 수익률 30% 이상 또는 52주 고점 80% 이상 제외
OVERHEAT_60D_MAX = 30  # 60일 수익률 상한
OVERHEAT_52W_MAX = 80  # 52주 위치 상한

overheated = set()
for ticker in close.columns:
    r60 = ret_by_period['60d'].get(ticker, 0)
    pos = position_52w.get(ticker, 50)
    if r60 > OVERHEAT_60D_MAX or pos > OVERHEAT_52W_MAX:
        overheated.add(ticker)

all_exclude = exclude_tickers | overheated

print(f"\n{'='*110}")
print(f"🔥 과열 필터 적용")
print(f"   60일 수익률 > {OVERHEAT_60D_MAX}% 또는 52주 위치 > {OVERHEAT_52W_MAX}% → 제외")
print(f"   과열 종목: {len(overheated)}개 제외")
print(f"   PER+흑자 필터: {len(exclude_tickers)}개 제외")
print(f"   최종 유니버스: {len(close.columns) - len(all_exclude)}종목")
print(f"{'='*110}")


# ── 알파 1: 모멘텀 가속 × 실적 개선 (5c) ──
# 백테스트 결과 Test IC +0.0280
accel_alpha = ops.normed_rank(
    ops.cwise_mul(
        ops.cwise_mul(
            ops.minus(ops.ts_delta_ratio(close, 10), ops.ts_delta_ratio(close.shift(10), 10)),
            ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20))
        ),
        oi_yoy_rank
    )
)

# ── 알파 2: v6 재무추세 (가장 강력한 단일 알파) ──
# 백테스트 결과 Test IC +0.0592 (20d fwd)
# 이미 계산됨: v6_alpha

# ── 알파 3: v3 가격 모멘텀 (가속과 구분) ──
# 이미 계산됨: v3_alpha

# ── Catch-up 앙상블: 가속 30% + v6 50% + v3 20% ──
# 가속 비중을 높이고 순수 모멘텀(v3) 비중 축소
accel_filled = accel_alpha.fillna(0.5)
catchup_ensemble = ops.normed_rank(
    accel_filled * 0.30 + v6_filled * 0.50 + v3_alpha * 0.20
)

catchup_scores = catchup_ensemble.loc[latest_date].dropna().sort_values(ascending=False)
catchup_filtered = catchup_scores[~catchup_scores.index.isin(all_exclude)]


# ══════════════════════════════════════════════════════════════
# 결과 출력
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*120}")
print(f"🎯 Catch-up 종목 선정: '아직 덜 올랐지만 곧 오를' 종목")
print(f"   알파: 모멘텀가속 30% + 재무추세 50% + 가격모멘텀 20%")
print(f"   필터: PER ≤ {FILTER_PER_MAX}x + 흑자 + 60d수익률 ≤ {OVERHEAT_60D_MAX}% + 52W위치 ≤ {OVERHEAT_52W_MAX}%")
print(f"   유니버스: {len(catchup_scores)}종목 → {len(catchup_filtered)}종목 통과")
print(f"{'='*120}")
print(f"{'순위':>4} {'종목코드':<10} {'종목명':<14} {'점수':>7} {'가속':>6} {'v6':>6} {'v3':>6} {'현재가':>10} {'20d':>7} {'60d':>7} {'90d':>7} {'52W':>6} {'PER':>6} {'NI(억)':>8}")
print(f"{'-'*120}")

for i, (ticker, score) in enumerate(catchup_filtered.head(15).items(), 1):
    name = ticker_name.get(ticker, '?')
    acc_s = accel_alpha.loc[latest_date, ticker] if ticker in accel_alpha.columns else np.nan
    v6_s = v6_alpha.loc[latest_date, ticker] if ticker in v6_alpha.columns else np.nan
    v3_s = v3_alpha.loc[latest_date, ticker] if ticker in v3_alpha.columns else np.nan
    price = close.loc[latest_date, ticker]
    r20 = ret_by_period['20d'].get(ticker, 0)
    r60 = ret_by_period['60d'].get(ticker, 0)
    r90 = ret_by_period['90d'].get(ticker, 0)
    pos = position_52w.get(ticker, 50)
    v = valuation.get(ticker, {})
    per = v.get('per', np.nan)
    ni = v.get('trailing_ni', 0)
    ni_억 = ni / 1e8 if ni else 0
    per_s = f"{per:.1f}x" if per and not np.isnan(per) else "  -"
    acc_str = f"{acc_s:.3f}" if not np.isnan(acc_s) else "  -  "
    v6_str = f"{v6_s:.3f}" if not np.isnan(v6_s) else "  -  "

    print(f"  {i:>2}. {ticker:<10} {name:<14} {score:.4f} {acc_str} {v6_str} {v3_s:.3f} {price:>10,.0f}원 {r20:>+5.1f}% {r60:>+5.1f}% {r90:>+5.1f}% {pos:>4.0f}% {per_s:>6} {ni_억:>7,.0f}억")


# ── 과열 필터로 제외된 상위 종목 (참고용) ──
overheated_top = catchup_scores[catchup_scores.index.isin(overheated) & ~catchup_scores.index.isin(exclude_tickers)].head(5)
if not overheated_top.empty:
    print(f"\n🔥 과열 필터로 제외된 고점수 종목:")
    for ticker, score in overheated_top.items():
        name = ticker_name.get(ticker, '?')
        r60 = ret_by_period['60d'].get(ticker, 0)
        r90 = ret_by_period['90d'].get(ticker, 0)
        pos = position_52w.get(ticker, 50)
        print(f"     {ticker:<10} {name:<14} 점수 {score:.4f}  60d: {r60:+.1f}%  90d: {r90:+.1f}%  52W: {pos:.0f}%")


# ── 하위 종목 ──
print(f"\n{'='*120}")
print(f"📉 하위 5종목 (숏 후보)")
print(f"{'='*120}")
for i, (ticker, score) in enumerate(catchup_filtered.tail(5).items(), 1):
    name = ticker_name.get(ticker, '?')
    price = close.loc[latest_date, ticker]
    r60 = ret_by_period['60d'].get(ticker, 0)
    r90 = ret_by_period['90d'].get(ticker, 0)
    pos = position_52w.get(ticker, 50)
    v = valuation.get(ticker, {})
    per = v.get('per', np.nan)
    per_s = f"{per:.1f}x" if per and not np.isnan(per) else "  -"
    print(f"  {i:>2}. {ticker:<10} {name:<14} 점수 {score:.4f} {price:>10,.0f}원 60d:{r60:>+5.1f}% 90d:{r90:>+5.1f}% 52W:{pos:>4.0f}% {per_s}")


print(f"\n{'='*120}")
print(f"💡 해석")
print(f"{'='*120}")
print(f"   ✅ 선정 기준:")
print(f"      1) PER ≤ {FILTER_PER_MAX}x + 영업·순이익 흑자 (밸류 필터)")
print(f"      2) 60일 수익률 ≤ {OVERHEAT_60D_MAX}% (이미 많이 오른 종목 제외)")
print(f"      3) 52주 고점 대비 {OVERHEAT_52W_MAX}% 이하 위치 (고점 근처 제외)")
print(f"      4) 모멘텀 '가속' 신호 (최근 10일 vs 이전 10일 수익률 차이)")
print(f"      5) 영업이익 추세 개선 (YoY + QoQ + 3Q 추세)")
print(f"   → '60일간 30% 미만 상승 + 실적 개선 + 최근 가속' 종목")
print(f"   → 한달(20영업일) 보유 전략")
