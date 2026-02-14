#!/usr/bin/env python3
"""
3팩터 앙상블 상위 종목 선정 (20일 리밸런싱)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

v7 모멘텀×실적 (25%) + v3 가격 (25%) + v6 재무추세 (50%)
Test IC: +0.0570 / IR: 0.92 (20-day forward)

각 팩터:
  v7: 20일 모멘텀 × 거래량 안정성 × 25일 레인지 × 실적 YoY 개선
  v3: 25일 모멘텀 × 거래량 안정성 × 28일 레인지 (Test IC 0.0374)
  v6: 영업이익 3Q추세 + YoY + QoQ 랭킹 합산 (Test IC 0.0592)

필터: PER ≤ 50x, 영업흑자, 순이익흑자
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

# ── 밸류에이션 필터용 ──
print("   밸류에이션 필터 계산 중...")
valuation = {}
for ticker, grp in fin_df.groupby('ticker'):
    grp = grp.sort_values('period_end')
    recent = grp.tail(4)
    if len(recent) < 2:
        continue
    trailing_rev = recent['revenue'].sum()
    trailing_oi = recent['operating_income'].sum()
    trailing_ni = recent['net_income'].sum()
    mcap = ticker_mcap.get(ticker, 0)
    if mcap and mcap > 0 and trailing_ni and trailing_ni > 0:
        per = mcap / trailing_ni
    else:
        per = np.nan
    valuation[ticker] = {
        'trailing_rev': trailing_rev,
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

tickers_with_valuation = set(valuation.keys())
no_data_tickers = set(close.columns) - tickers_with_valuation
exclude_tickers = filtered_out | no_data_tickers

print(f"   필터 조건: PER ≤ {FILTER_PER_MAX}x, 영업이익 흑자, 순이익 흑자")
print(f"   밸류 필터 제외: {len(filtered_out)}개, 재무데이터 없음: {len(no_data_tickers)}개")

# ── QoQ / YoY / 3분기 추세 ──
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
print(f"   최신 날짜: {close.index[-1]}")
print(f"   재무 추세: {list(trend_vars.keys())}")


# ══════════════════════════════════════════════════════════════
# 3팩터 알파 계산
# ══════════════════════════════════════════════════════════════

# ── v7: 20일 모멘텀 × 거래량 안정성 × 25일 레인지 × 실적 YoY ──
# 가격 모멘텀 + 실적 개선이 동시에 나타나는 종목 (20d 최적화)
v7_alpha = ops.normed_rank(
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

# ── v3: 25일 모멘텀 × 거래량 안정성 × 28일 레인지 ──
v3_alpha = ops.normed_rank(
    ops.cwise_mul(
        ops.cwise_mul(
            ops.ts_delta_ratio(close, 25),
            ops.div(ops.ts_median(volume, 10), ops.ts_std(volume, 15))
        ),
        ops.ts_maxmin_scale(close, 28)
    )
)

# ── v6: 영업이익 추세 + YoY + QoQ 랭킹 합산 ──
v6_alpha = ops.normed_rank(
    ops.add(ops.add(oi_trend_rank, oi_yoy_rank), oi_qoq_rank)
)

# ── 3팩터 앙상블: v7 25% + v3 25% + v6 50% ──
W_V7 = 0.25   # 모멘텀×실적 (catch-up 성격)
W_V3 = 0.25   # 순수 가격 모멘텀
W_V6 = 0.50   # 재무추세 (가장 강력)

v6_filled = v6_alpha.fillna(0.5)
v7_filled = v7_alpha.fillna(0.5)

ensemble = ops.normed_rank(
    v7_filled * W_V7 + v3_alpha * W_V3 + v6_filled * W_V6
)

# ── 최신 날짜 기준 ──
latest_date = ensemble.index[-1]
all_scores = ensemble.loc[latest_date].dropna().sort_values(ascending=False)

# 밸류에이션 필터 적용
filtered_scores = all_scores[~all_scores.index.isin(exclude_tickers)]

# ── 추가 정보: 최근 20일 수익률, 모멘텀 가속도 ──
if len(close) > 20:
    ret_20d = (close.iloc[-1] / close.iloc[-20] - 1) * 100
else:
    ret_20d = pd.Series(0, index=close.columns)

# 모멘텀 가속: 최근 10일 수익률 - 이전 10일 수익률
if len(close) > 20:
    mom_recent = close.iloc[-1] / close.iloc[-10] - 1
    mom_prev = close.iloc[-10] / close.iloc[-20] - 1
    mom_accel = (mom_recent - mom_prev) * 100
else:
    mom_accel = pd.Series(0, index=close.columns)


# ══════════════════════════════════════════════════════════════
# 결과 출력
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*100}")
print(f"🏆 3팩터 앙상블 상위 종목 (기준일: {latest_date})")
print(f"   v7 모멘텀×실적 ({W_V7:.0%}) + v3 가격 ({W_V3:.0%}) + v6 재무추세 ({W_V6:.0%})")
print(f"   Test IC: +0.0570 / IR: 0.92 (20-day forward)")
print(f"   필터: PER ≤ {FILTER_PER_MAX}x, 영업흑자, 순이익흑자")
print(f"   유니버스: {len(all_scores)}종목 → 필터 통과 {len(filtered_scores)}종목")
print(f"{'='*100}")
print(f"{'순위':>4} {'종목코드':<10} {'종목명':<14} {'앙상블':>8} {'v7':>6} {'v3':>6} {'v6':>6} {'현재가':>12} {'20d수익률':>10} {'가속도':>8} {'PER':>6} {'NI(억)':>8}")
print(f"{'-'*110}")

for i, (ticker, score) in enumerate(filtered_scores.head(15).items(), 1):
    name = ticker_name.get(ticker, '?')
    v7_s = v7_alpha.loc[latest_date, ticker] if ticker in v7_alpha.columns else np.nan
    v3_s = v3_alpha.loc[latest_date, ticker] if ticker in v3_alpha.columns else np.nan
    v6_s = v6_alpha.loc[latest_date, ticker] if ticker in v6_alpha.columns else np.nan
    price = close.loc[latest_date, ticker]
    ret = ret_20d.get(ticker, 0)
    accel = mom_accel.get(ticker, 0)
    v = valuation.get(ticker, {})
    per = v.get('per', np.nan)
    ni = v.get('trailing_ni', 0)
    ni_억 = ni / 1e8 if ni else 0
    per_s = f"{per:.1f}x" if per and not np.isnan(per) else "  -"
    v7_str = f"{v7_s:.3f}" if not np.isnan(v7_s) else "  -  "
    v6_str = f"{v6_s:.3f}" if not np.isnan(v6_s) else "  -  "

    # 가속도 표시: 양수면 ↑, 음수면 ↓
    accel_mark = "↑" if accel > 1 else ("↓" if accel < -1 else "→")

    print(f"  {i:>2}. {ticker:<10} {name:<14} {score:.4f} {v7_str}  {v3_s:.3f}  {v6_str} {price:>12,.0f}원 {ret:>+7.1f}%  {accel:>+5.1f}{accel_mark} {per_s:>6} {ni_억:>7,.0f}억")

# ── 필터로 제외된 상위 종목 ──
excluded_top = all_scores[all_scores.index.isin(filtered_out)].head(5)
if not excluded_top.empty:
    print(f"\n⚠️  필터로 제외된 고점수 종목:")
    for ticker, score in excluded_top.items():
        name = ticker_name.get(ticker, '?')
        v = valuation.get(ticker, {})
        per = v.get('per', np.nan)
        ni = v.get('trailing_ni', 0)
        ni_억 = ni / 1e8 if ni else 0
        per_s = f"PER {per:.1f}x" if per and not np.isnan(per) else "PER N/A"
        ni_s = f"NI {ni_억:,.0f}억" if ni and ni > 0 else "순이익적자"
        oi = v.get('trailing_oi', 0)
        oi_s = "" if oi and oi > 0 else " 영업적자"
        print(f"     {ticker:<10} {name:<14} 점수 {score:.4f}  {per_s}  {ni_s}{oi_s}")

# ── 하위 5종목 ──
print(f"\n{'='*100}")
print(f"📉 하위 5종목 (숏 후보, 필터 통과)")
print(f"{'='*100}")
print(f"{'순위':>4} {'종목코드':<10} {'종목명':<14} {'앙상블':>8} {'v7':>6} {'v3':>6} {'v6':>6} {'현재가':>12} {'20d수익률':>10} {'PER':>6} {'NI(억)':>8}")
print(f"{'-'*100}")

for i, (ticker, score) in enumerate(filtered_scores.tail(5).items(), 1):
    name = ticker_name.get(ticker, '?')
    v7_s = v7_alpha.loc[latest_date, ticker] if ticker in v7_alpha.columns else np.nan
    v3_s = v3_alpha.loc[latest_date, ticker] if ticker in v3_alpha.columns else np.nan
    v6_s = v6_alpha.loc[latest_date, ticker] if ticker in v6_alpha.columns else np.nan
    price = close.loc[latest_date, ticker]
    ret = ret_20d.get(ticker, 0)
    v = valuation.get(ticker, {})
    per = v.get('per', np.nan)
    ni = v.get('trailing_ni', 0)
    ni_억 = ni / 1e8 if ni else 0
    per_s = f"{per:.1f}x" if per and not np.isnan(per) else "  -"
    v7_str = f"{v7_s:.3f}" if not np.isnan(v7_s) else "  -  "
    v6_str = f"{v6_s:.3f}" if not np.isnan(v6_s) else "  -  "
    print(f"  {i:>2}. {ticker:<10} {name:<14} {score:.4f} {v7_str}  {v3_s:.3f}  {v6_str} {price:>12,.0f}원 {ret:>+7.1f}%  {per_s:>6} {ni_억:>7,.0f}억")

# ── 15일 vs 20일 비교 ──
print(f"\n{'='*100}")
print(f"📊 15일 vs 20일 전략 비교")
print(f"{'='*100}")

# 15일 앙상블 (기존 select_top5.py와 동일)
ens_15d = ops.normed_rank(v3_alpha * 0.5 + v6_filled * 0.5)
scores_15d = ens_15d.loc[latest_date].dropna().sort_values(ascending=False)
scores_15d = scores_15d[~scores_15d.index.isin(exclude_tickers)]

# 20일 앙상블 (현재)
scores_20d = filtered_scores

# 겹치는 종목 분석
top10_15d = set(scores_15d.head(10).index)
top10_20d = set(scores_20d.head(10).index)
overlap = top10_15d & top10_20d

print(f"   15일 전략 Top10: {len(top10_15d)}종목")
print(f"   20일 전략 Top10: {len(top10_20d)}종목")
print(f"   겹치는 종목: {len(overlap)}개")

if overlap:
    print(f"   → 공통: {', '.join(ticker_name.get(t, t) for t in overlap)}")

only_20d = top10_20d - top10_15d
if only_20d:
    print(f"   → 20일 전략에만: {', '.join(ticker_name.get(t, t) for t in only_20d)}")

only_15d = top10_15d - top10_20d
if only_15d:
    print(f"   → 15일 전략에만: {', '.join(ticker_name.get(t, t) for t in only_15d)}")


print(f"\n{'='*100}")
print(f"💡 해석")
print(f"{'='*100}")
print(f"   v7 모멘텀×실적: 20일 가격 상승 × 거래량 안정 × 레인지 고위치 × 영업이익 YoY 개선")
print(f"   v3 가격모멘텀:  25일 가격 상승 × 거래량 안정 × 28일 레인지 고위치")
print(f"   v6 재무추세:    영업이익 3Q추세 + YoY + QoQ 개선도 랭킹 합산")
print(f"   가속도:         최근 10일 수익률 - 이전 10일 수익률 (↑=가속, ↓=감속)")
print(f"   → 20영업일(약 1달) 보유 전략에 최적화")
print(f"   → v7이 '실적 개선 + 가격 반영 초기' 종목을 포착")
print(f"   → v6 비중 50%로 실적 추세에 가장 큰 가중치")
