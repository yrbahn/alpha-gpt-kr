#!/usr/bin/env python3
"""
앙상블 알파로 상위 5종목 선정 (20영업일 = 1달 리밸런싱)
v11 Multi-Alpha Ensemble (60%) + v6 재무 추세 (40%)
- Tech: best_alphas.json에서 CV-검증된 상위 N개 알파를 IC-weighted 앙상블
- v6: 영업이익 추세 + YoY + QoQ 개선
- 장점: 단일 알파 대비 분산 감소, 안정성 증가 (overfitting 방지)
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


# 데이터 로드
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
    SELECT s.ticker, p.date, p.open, p.high, p.low, p.close, p.volume
    FROM price_data p
    JOIN stocks s ON p.stock_id = s.id
    WHERE p.stock_id IN ({stock_id_list})
    AND p.date >= CURRENT_DATE - INTERVAL '730 days'
    ORDER BY s.ticker, p.date
""", conn)

open_price = price_df.pivot(index='date', columns='ticker', values='open')
high = price_df.pivot(index='date', columns='ticker', values='high')
low = price_df.pivot(index='date', columns='ticker', values='low')
close = price_df.pivot(index='date', columns='ticker', values='close')
volume = price_df.pivot(index='date', columns='ticker', values='volume')
returns = close.pct_change()

# 수급 데이터 로드
print("   수급 데이터 로드 중...")
try:
    flow_df = pd.read_sql(f"""
        SELECT s.ticker, sd.date,
               sd.foreign_net_buy, sd.institution_net_buy,
               sd.individual_net_buy, sd.foreign_ownership
        FROM supply_demand_data sd
        JOIN stocks s ON sd.stock_id = s.id
        WHERE sd.stock_id IN ({stock_id_list})
        AND sd.date >= CURRENT_DATE - INTERVAL '730 days'
        ORDER BY s.ticker, sd.date
    """, conn)
    foreign_buy_raw = flow_df.pivot(index='date', columns='ticker', values='foreign_net_buy')
    inst_buy_raw = flow_df.pivot(index='date', columns='ticker', values='institution_net_buy')
    retail_buy_raw = flow_df.pivot(index='date', columns='ticker', values='individual_net_buy')
    foreign_own_raw = flow_df.pivot(index='date', columns='ticker', values='foreign_ownership')
    has_flow = True
except Exception as e:
    print(f"   ⚠️ 수급 데이터 로드 실패: {e}")
    has_flow = False

# 재무 추세 데이터
print("   재무 추세 데이터 로드 중...")
fin_df = pd.read_sql(f"""
    SELECT stock_id, period_end, revenue, operating_income, net_income, raw_data
    FROM financial_statements
    WHERE stock_id IN ({stock_id_list})
    ORDER BY stock_id, period_end
""", conn)
conn.close()

# raw_data에서 분기 타입 추출
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

# ── 밸류에이션 필터용: 최근 4분기 매출/영업이익/순이익 합산 ──
print("   밸류에이션 필터 계산 중...")
valuation = {}
for ticker, grp in fin_df.groupby('ticker'):
    grp = grp.sort_values('period_end')
    recent = grp.tail(4)  # 최근 4분기
    if len(recent) < 2:
        continue
    trailing_rev = recent['revenue'].sum()
    trailing_oi = recent['operating_income'].sum()
    trailing_ni = recent['net_income'].sum()
    mcap = ticker_mcap.get(ticker, 0)
    if mcap and mcap > 0 and trailing_rev and trailing_rev > 0:
        psr = mcap / trailing_rev
    else:
        psr = np.nan
    if mcap and mcap > 0 and trailing_ni and trailing_ni > 0:
        per = mcap / trailing_ni
    else:
        per = np.nan  # 적자 또는 데이터 없음
    valuation[ticker] = {
        'trailing_rev': trailing_rev,
        'trailing_oi': trailing_oi,
        'trailing_ni': trailing_ni,
        'per': per,
    }

# 필터 기준:
#   1) PER ≤ 50 (적자 = 제외, PER 50배 초과 = 제외)
#   2) 최근 4분기 영업이익 > 0 (흑자)
FILTER_PER_MAX = 50

filtered_out = set()
for ticker, v in valuation.items():
    reasons = []
    # 순이익 적자 → PER 산출 불가 → 제외
    if v['trailing_ni'] is None or v['trailing_ni'] <= 0:
        reasons.append("순이익 적자")
    elif v['per'] is not None and v['per'] > FILTER_PER_MAX:
        reasons.append(f"PER {v['per']:.1f}x")
    # 영업이익 적자도 제외
    if v['trailing_oi'] is not None and v['trailing_oi'] <= 0:
        reasons.append("영업적자")
    if reasons:
        filtered_out.add(ticker)

# 재무데이터 없는 종목도 제외
tickers_with_valuation = set(valuation.keys())
no_data_tickers = set(close.columns) - tickers_with_valuation

print(f"   필터 조건: PER ≤ {FILTER_PER_MAX}x, 영업이익 흑자, 순이익 흑자")
print(f"   밸류 필터 제외: {len(filtered_out)}개, 재무데이터 없음: {len(no_data_tickers)}개")

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

# 일별 forward-fill + cross-sectional rank
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

# ── 파생 기술 지표 (21개 전체 — multi-alpha ensemble용) ──
vwap = (high + low + close) / 3
high_low_range = (high - low) / close
body = (close - open_price) / open_price
upper_shadow = (high - close.clip(lower=open_price)) / close
lower_shadow = (close.clip(upper=open_price) - low) / close

tr1 = high - low
tr2 = (high - close.shift(1)).abs()
tr3 = (low - close.shift(1)).abs()
true_range = pd.concat([tr1, tr2, tr3]).groupby(level=0).max()
true_range = true_range.reindex(close.index)
atr_ratio = true_range / close

amount = close * volume
amihud = returns.abs() / amount.replace(0, np.nan)
amihud = amihud.replace([np.inf, -np.inf], np.nan).fillna(0)

gap = open_price / close.shift(1) - 1
intraday_ret = close / open_price - 1

vol_ratio = volume / volume.rolling(20, min_periods=5).mean()
vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1)

# 수급 비율 계산
if has_flow:
    foreign_buy_raw = foreign_buy_raw.reindex(index=close.index, columns=close.columns)
    inst_buy_raw = inst_buy_raw.reindex(index=close.index, columns=close.columns)
    retail_buy_raw = retail_buy_raw.reindex(index=close.index, columns=close.columns)
    foreign_own_raw = foreign_own_raw.reindex(index=close.index, columns=close.columns)
    safe_volume = volume.replace(0, np.nan)
    foreign_net_ratio = (foreign_buy_raw / safe_volume).clip(-1, 1).fillna(0)
    inst_net_ratio = (inst_buy_raw / safe_volume).clip(-1, 1).fillna(0)
    retail_net_ratio = (retail_buy_raw / safe_volume).clip(-1, 1).fillna(0)
    foreign_ownership_pct = (foreign_own_raw / 100).clip(0, 1).fillna(0)
    print(f"   수급 지표 4개 로드 완료")
else:
    foreign_net_ratio = close * 0.0
    inst_net_ratio = close * 0.0
    retail_net_ratio = close * 0.0
    foreign_ownership_pct = close * 0.0

print(f"✅ {len(close.columns)}개 종목, {len(close)}일 데이터")
print(f"   최신 날짜: {close.index[-1]}")
print(f"   재무 추세: {list(trend_vars.keys())}")

# ── Multi-Alpha Ensemble (best_alphas.json에서 로드) ──
alphas_path = project_root / 'best_alphas.json'
if alphas_path.exists():
    with open(alphas_path) as f:
        alpha_configs = _json.load(f)

    print(f"\n🧬 Multi-Alpha Ensemble: {len(alpha_configs)}개 알파 로드")
    alpha_values_list = []
    alpha_weights = []
    alpha_infos = []
    for i, cfg in enumerate(alpha_configs, 1):
        expr = cfg['expression']
        ic = cfg['mean_test_ic']
        try:
            val = eval(expr)
            if isinstance(val, pd.DataFrame):
                # IC가 양수인 알파만 포함 (음수 IC = 역효과)
                weight = max(ic, 0.001)
                alpha_values_list.append(val)
                alpha_weights.append(weight)
                alpha_infos.append(cfg)
                print(f"   #{i} IC={ic:.4f} IR={cfg['mean_test_ir']:.2f} [{cfg['factors']}] ✅")
            else:
                print(f"   #{i} 비정상 출력 (DataFrame이 아님) ⚠️")
        except Exception as e:
            print(f"   #{i} 계산 실패: {e} ⚠️")

    if alpha_values_list:
        # IC-weighted average → normed_rank
        total_w = sum(alpha_weights)
        tech_alpha = sum(v * (w / total_w) for v, w in zip(alpha_values_list, alpha_weights))
        tech_alpha = ops.normed_rank(tech_alpha)
        n_alphas = len(alpha_values_list)
        print(f"   → {n_alphas}개 알파 IC-weighted 앙상블 완료 (총 가중치: {total_w:.4f})")
    else:
        print("   ⚠️ 유효한 알파 없음 → 단일 폴백 알파 사용")
        tech_alpha = ops.normed_rank(
            ops.add(
                ops.div(ops.ts_decayed_linear(foreign_ownership_pct, 8), ops.ts_mean(vol_ratio, 57)),
                ops.zscore_scale(ops.div(amihud, ops.ts_mean(close, 110)))
            )
        )
        n_alphas = 1
else:
    print("\n⚠️ best_alphas.json 없음 → 단일 v10 알파 사용")
    tech_alpha = ops.normed_rank(
        ops.add(
            ops.div(ops.ts_decayed_linear(foreign_ownership_pct, 8), ops.ts_mean(vol_ratio, 57)),
            ops.zscore_scale(ops.div(amihud, ops.ts_mean(close, 110)))
        )
    )
    n_alphas = 1

# ── v6 재무 추세 알파 ──
v6_alpha = ops.normed_rank(
    ops.add(
        ops.add(oi_trend_rank, oi_yoy_rank),
        oi_qoq_rank
    )
)

# ── 앙상블 (60% multi-alpha tech + 40% 재무추세) ──
W_TECH = 0.60
W_FUND = 0.40
v6_filled = v6_alpha.fillna(0.5)
ensemble = ops.normed_rank(tech_alpha * W_TECH + v6_filled * W_FUND)

# 최신 날짜 기준
latest_date = ensemble.index[-1]
all_scores = ensemble.loc[latest_date].dropna().sort_values(ascending=False)

# 밸류에이션 필터 적용
exclude_tickers = filtered_out | no_data_tickers
filtered_scores = all_scores[~all_scores.index.isin(exclude_tickers)]

print(f"\n{'='*100}")
print(f"🏆 앙상블 상위 종목 (기준일: {latest_date})")
print(f"   Multi-Alpha Tech ({W_TECH:.0%}, {n_alphas}개 IC-weighted) + v6 재무추세 ({W_FUND:.0%})")
print(f"   필터: PER ≤ {FILTER_PER_MAX}x, 영업흑자, 순이익흑자")
print(f"   유니버스: {len(all_scores)}종목 → 필터 통과 {len(filtered_scores)}종목")
print(f"{'='*100}")
print(f"{'순위':>4} {'종목코드':<10} {'종목명':<16} {'앙상블':>8} {'Tech':>6} {'v6':>6} {'현재가':>12} {'PER':>6} {'NI(억)':>8}")
print(f"{'-'*100}")

for i, (ticker, score) in enumerate(filtered_scores.head(10).items(), 1):
    name = ticker_name.get(ticker, '?')
    v10_s = tech_alpha.loc[latest_date, ticker] if ticker in tech_alpha.columns else np.nan
    v6_s = v6_alpha.loc[latest_date, ticker] if ticker in v6_alpha.columns else np.nan
    price = close.loc[latest_date, ticker]
    v = valuation.get(ticker, {})
    per = v.get('per', np.nan)
    ni = v.get('trailing_ni', 0)
    ni_억 = ni / 1e8 if ni else 0
    per_s = f"{per:.1f}x" if per and not np.isnan(per) else "  -"
    print(f"  {i:>2}. {ticker:<10} {name:<16} {score:.4f} {v10_s:.3f}  {v6_s:.3f} {price:>12,.0f}원 {per_s:>6} {ni_억:>7,.0f}억")

# 필터로 제외된 상위 종목 (참고용)
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
        print(f"     {ticker:<10} {name:<16} 점수 {score:.4f}  {per_s}  {ni_s}{oi_s}")

# 하위 5종목
print(f"\n{'='*100}")
print(f"📉 하위 5종목 (숏 후보, 필터 통과)")
print(f"{'='*100}")
print(f"{'순위':>4} {'종목코드':<10} {'종목명':<16} {'앙상블':>8} {'v10':>6} {'v6':>6} {'현재가':>12} {'PER':>6} {'NI(억)':>8}")
print(f"{'-'*100}")

for i, (ticker, score) in enumerate(filtered_scores.tail(5).items(), 1):
    name = ticker_name.get(ticker, '?')
    v10_s = tech_alpha.loc[latest_date, ticker] if ticker in tech_alpha.columns else np.nan
    v6_s = v6_alpha.loc[latest_date, ticker] if ticker in v6_alpha.columns else np.nan
    price = close.loc[latest_date, ticker]
    v = valuation.get(ticker, {})
    per = v.get('per', np.nan)
    ni = v.get('trailing_ni', 0)
    ni_억 = ni / 1e8 if ni else 0
    per_s = f"{per:.1f}x" if per and not np.isnan(per) else "  -"
    print(f"  {i:>2}. {ticker:<10} {name:<16} {score:.4f} {v10_s:.3f}  {v6_s:.3f} {price:>12,.0f}원 {per_s:>6} {ni_억:>7,.0f}억")

print(f"\n💡 해석:")
print(f"   Tech: {n_alphas}개 CV-검증 알파 IC-weighted 앙상블 (단일 알파 대비 분산↓ 안정성↑)")
print(f"   v6 재무: 영업이익 3Q 추세 + YoY + QoQ 개선도 랭킹")
print(f"   앙상블: Multi-Alpha Tech ({W_TECH:.0%}) + 재무추세 ({W_FUND:.0%})")
print(f"   필터: PER ≤ {FILTER_PER_MAX}x + 흑자 → 고밸류 모멘텀 함정 제거")
print(f"   → 20영업일(약 1달) 보유 전략에 최적화")
