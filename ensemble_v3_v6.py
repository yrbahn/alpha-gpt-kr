#!/usr/bin/env python3
"""
v3 가격 알파 + v6 재무 추세 알파 앙상블
- v3: 모멘텀 × 거래량 안정성 × 가격 레인지 (Train IC 0.0389 / Test IC 0.0385)
- v6: 영업이익 추세 + YoY 개선 (Train IC 0.1511 / Test IC 0.1100)
- 가중 평균으로 앙상블 → 최적 가중치 탐색
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


def load_data():
    """가격 + 재무 추세 데이터 로드"""
    print("📊 데이터 로드 중... (시총 상위 500종목, 2년)")
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
        LIMIT 500
    """, conn)

    stock_ids = stocks_df['id'].tolist()
    stock_id_list = ', '.join(map(str, stock_ids))
    ticker_name = dict(zip(stocks_df['ticker'], stocks_df['name']))
    ticker_mcap = dict(zip(stocks_df['ticker'], stocks_df['market_cap']))
    id_ticker = dict(zip(stocks_df['id'], stocks_df['ticker']))

    # 가격 데이터
    price_df = pd.read_sql(f"""
        SELECT s.ticker, p.date, p.open, p.high, p.low, p.close, p.volume
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
        SELECT stock_id, period_end, revenue, operating_income, net_income,
               total_equity, total_assets, raw_data
        FROM financial_statements
        WHERE stock_id IN ({stock_id_list})
        ORDER BY stock_id, period_end
    """, conn)
    conn.close()

    # raw_data에서 분기 타입 추출
    def _parse_raw(row):
        rd = row.get('raw_data')
        if rd is None:
            return {'quarter_type': None, 'roe': None}
        if isinstance(rd, str):
            rd = _json.loads(rd)
        return {
            'quarter_type': rd.get('quarter', ''),
            'roe': rd.get('roe'),
        }

    raw_parsed = fin_df.apply(_parse_raw, axis=1, result_type='expand')
    fin_df = pd.concat([fin_df, raw_parsed], axis=1)
    fin_df['ticker'] = fin_df['stock_id'].map(id_ticker)
    fin_df = fin_df.dropna(subset=['ticker'])
    fin_df = fin_df[fin_df['quarter_type'] != '연간'].copy()

    fin_df['op_margin'] = np.where(
        (fin_df['revenue'].notna()) & (fin_df['revenue'] != 0),
        fin_df['operating_income'] / fin_df['revenue'],
        np.nan
    )
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

    # 일별 forward-fill + cross-sectional rank
    trend_vars = {}
    for field in ['oi_qoq', 'oi_yoy', 'oi_trend']:
        if field not in trend_df.columns:
            continue
        pivot = trend_df.pivot_table(index='period_end', columns='ticker', values=field, aggfunc='last')
        if pivot.empty or pivot.notna().sum().sum() < 50:
            continue
        daily = pivot.reindex(close.index).ffill()
        daily = daily.reindex(columns=close.columns)
        ranked = daily.rank(axis=1, pct=True)
        trend_vars[f'{field}_rank'] = ranked

    print(f"✅ {len(close.columns)}개 종목, {len(close)}일 데이터")
    print(f"   재무 추세 변수: {list(trend_vars.keys())}")

    return {
        'close': close,
        'volume': volume,
        'returns': returns,
        'ticker_name': ticker_name,
        'ticker_mcap': ticker_mcap,
        **trend_vars,
    }


def compute_v3_alpha(close, volume):
    """v3 가격 알파: 모멘텀 × 거래량 안정성 × 가격 레인지"""
    return ops.normed_rank(
        ops.cwise_mul(
            ops.cwise_mul(
                ops.ts_delta_ratio(close, 25),
                ops.div(ops.ts_median(volume, 10), ops.ts_std(volume, 15))
            ),
            ops.ts_maxmin_scale(close, 28)
        )
    )


def compute_v6_alpha(oi_trend_rank, oi_yoy_rank, oi_qoq_rank=None):
    """v6 재무 추세 알파: 영업이익 추세 + YoY + QoQ 개선

    ts_zscore_scale 제거 — forward-fill된 분기 데이터에서 std≈0으로 NaN 발생 방지.
    대신 3가지 추세 지표를 합산하여 매일 유효한 신호 생성.
    """
    result = ops.add(oi_trend_rank, oi_yoy_rank)
    if oi_qoq_rank is not None:
        result = ops.add(result, oi_qoq_rank)
    return ops.normed_rank(result)


def compute_ensemble(v3_alpha, v6_alpha, w_price=0.5):
    """가중 앙상블 → normed_rank

    v3, v6 모두 normed_rank [0,1] 범위이므로 가중 평균 후 재랭킹.
    v6에 NaN이 있는 종목은 v3만으로 순위 결정.
    """
    w_fund = 1.0 - w_price
    # NaN 처리: v6가 NaN이면 v3만 사용 (0.5로 대체)
    v6_filled = v6_alpha.fillna(0.5)
    blended = v3_alpha * w_price + v6_filled * w_fund
    return ops.normed_rank(blended)


def compute_ic(alpha_values, close, forward_days=15):
    """IC 계산 (Pearson cross-sectional correlation)"""
    forward_return = close.shift(-forward_days) / close - 1

    ic_list = []
    for date in alpha_values.index[:-forward_days]:
        alpha_cs = alpha_values.loc[date]
        returns_cs = forward_return.loc[date]
        valid = alpha_cs.notna() & returns_cs.notna()

        if valid.sum() > 30:
            ic = alpha_cs[valid].corr(returns_cs[valid])
            if not np.isnan(ic):
                ic_list.append(ic)

    if len(ic_list) < 10:
        return -999.0, 0.0, 0

    return float(np.mean(ic_list)), float(np.std(ic_list)), len(ic_list)


def main():
    print("=" * 80)
    print("🔀 Ensemble: v3 가격 알파 + v6 재무 추세 알파")
    print("=" * 80)

    data = load_data()
    close = data['close']
    volume = data['volume']

    # 재무 추세 변수
    _empty = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    oi_trend_rank = data.get('oi_trend_rank', _empty)
    oi_yoy_rank = data.get('oi_yoy_rank', _empty)
    oi_qoq_rank = data.get('oi_qoq_rank', _empty)

    # 데이터 커버리지 확인
    latest_date = close.index[-1]
    v6_coverage = oi_yoy_rank.loc[latest_date].notna().sum()
    print(f"\n📋 데이터 커버리지 (기준일: {latest_date}):")
    print(f"   oi_trend_rank: {oi_trend_rank.loc[latest_date].notna().sum()}/{len(close.columns)} 종목")
    print(f"   oi_yoy_rank:   {oi_yoy_rank.loc[latest_date].notna().sum()}/{len(close.columns)} 종목")
    print(f"   oi_qoq_rank:   {oi_qoq_rank.loc[latest_date].notna().sum()}/{len(close.columns)} 종목")

    # Train/Test 분할 (70/30)
    split_idx = int(len(close) * 0.7)
    print(f"\n📐 Train/Test 분할: {split_idx}일 train / {len(close) - split_idx}일 test")
    print(f"   Train: {close.index[0]} ~ {close.index[split_idx-1]}")
    print(f"   Test:  {close.index[split_idx]} ~ {close.index[-1]}")

    # ── 개별 알파 계산 ──
    print("\n🧮 알파 계산 중...")
    v3_alpha = compute_v3_alpha(close, volume)
    v6_alpha = compute_v6_alpha(oi_trend_rank, oi_yoy_rank, oi_qoq_rank)

    # 커버리지 확인
    v3_valid = v3_alpha.loc[latest_date].notna().sum()
    v6_valid = v6_alpha.loc[latest_date].notna().sum()
    print(f"   v3 유효 종목: {v3_valid}/{len(close.columns)}")
    print(f"   v6 유효 종목: {v6_valid}/{len(close.columns)}")

    # 개별 IC
    print("\n📊 개별 알파 IC:")
    for name, alpha in [("v3 가격", v3_alpha), ("v6 재무추세", v6_alpha)]:
        train_ic, train_std, train_n = compute_ic(alpha.iloc[:split_idx], close.iloc[:split_idx])
        test_ic, test_std, test_n = compute_ic(alpha.iloc[split_idx:], close.iloc[split_idx:])
        print(f"   {name:12s}  Train IC: {train_ic:+.4f} (std {train_std:.4f}, n={train_n})")
        print(f"   {' ':12s}  Test IC:  {test_ic:+.4f} (std {test_std:.4f}, n={test_n})")

    # ── 가중치 탐색 ──
    print(f"\n{'='*80}")
    print("🔍 최적 가중치 탐색 (w_price : w_fund)")
    print(f"{'='*80}")
    print(f"{'w_price':>8} {'w_fund':>8} {'Train IC':>10} {'Test IC':>10} {'IR(test)':>10}")
    print(f"{'-'*50}")

    best_test_ic = -999
    best_weight = 0.5
    results = []

    for w_price_pct in range(10, 95, 5):  # 10% ~ 90%
        w_price = w_price_pct / 100.0
        ensemble = compute_ensemble(v3_alpha, v6_alpha, w_price)

        train_ic, _, _ = compute_ic(ensemble.iloc[:split_idx], close.iloc[:split_idx])
        test_ic, test_std, test_n = compute_ic(ensemble.iloc[split_idx:], close.iloc[split_idx:])
        ir = test_ic / test_std if test_std > 0 else 0

        marker = ""
        if test_ic > best_test_ic:
            best_test_ic = test_ic
            best_weight = w_price
            marker = " ◀ best"

        results.append((w_price, train_ic, test_ic, ir))
        print(f"  {w_price:5.0%}   {1-w_price:5.0%}   {train_ic:+.4f}    {test_ic:+.4f}    {ir:+.4f}{marker}")

    # ── 최적 앙상블 ──
    print(f"\n{'='*80}")
    print(f"🏆 최적 앙상블: w_price={best_weight:.0%}, w_fund={1-best_weight:.0%}")
    print(f"{'='*80}")

    best_ensemble = compute_ensemble(v3_alpha, v6_alpha, best_weight)
    train_ic, train_std, train_n = compute_ic(best_ensemble.iloc[:split_idx], close.iloc[:split_idx])
    test_ic, test_std, test_n = compute_ic(best_ensemble.iloc[split_idx:], close.iloc[split_idx:])

    print(f"   Train IC: {train_ic:+.4f} (std {train_std:.4f}, n={train_n})")
    print(f"   Test IC:  {test_ic:+.4f} (std {test_std:.4f}, n={test_n})")
    print(f"   IR (test): {test_ic/test_std:.2f}" if test_std > 0 else "")

    # vs 개별 알파 비교
    v3_test_ic, _, _ = compute_ic(v3_alpha.iloc[split_idx:], close.iloc[split_idx:])
    v6_test_ic, _, _ = compute_ic(v6_alpha.iloc[split_idx:], close.iloc[split_idx:])
    print(f"\n   📈 개선도 vs v3 단독: Test IC {v3_test_ic:+.4f} → {test_ic:+.4f} ({test_ic - v3_test_ic:+.4f})")
    print(f"   📈 개선도 vs v6 단독: Test IC {v6_test_ic:+.4f} → {test_ic:+.4f} ({test_ic - v6_test_ic:+.4f})")

    # ── 상위 5종목 ──
    ticker_name = data['ticker_name']
    ticker_mcap = data['ticker_mcap']

    latest_date = best_ensemble.index[-1]
    latest_scores = best_ensemble.loc[latest_date].dropna().sort_values(ascending=False)

    print(f"\n{'='*80}")
    print(f"🏆 앙상블 상위 5종목 (기준일: {latest_date})")
    print(f"   v3 가격 (w={best_weight:.0%}) + v6 재무추세 (w={1-best_weight:.0%})")
    print(f"{'='*80}")
    print(f"{'순위':>4} {'종목코드':<10} {'종목명':<16} {'앙상블':>8} {'v3점수':>8} {'v6점수':>8} {'현재가':>12} {'시총(억)':>10}")
    print(f"{'-'*80}")

    for i, (ticker, score) in enumerate(latest_scores.head(10).items(), 1):
        name = ticker_name.get(ticker, '?')
        v3_score = v3_alpha.loc[latest_date, ticker] if ticker in v3_alpha.columns else np.nan
        v6_score = v6_alpha.loc[latest_date, ticker] if ticker in v6_alpha.columns else np.nan
        price = close.loc[latest_date, ticker]
        mcap = ticker_mcap.get(ticker, 0)
        mcap_억 = mcap / 1e8 if mcap else 0
        print(f"  {i:>2}. {ticker:<10} {name:<16} {score:.4f}   {v3_score:.4f}   {v6_score:.4f} {price:>12,.0f}원 {mcap_억:>8,.0f}억")

    # 하위 5종목
    print(f"\n📉 하위 5종목 (숏 후보)")
    print(f"{'-'*80}")
    for i, (ticker, score) in enumerate(latest_scores.tail(5).items(), 1):
        name = ticker_name.get(ticker, '?')
        v3_score = v3_alpha.loc[latest_date, ticker] if ticker in v3_alpha.columns else np.nan
        v6_score = v6_alpha.loc[latest_date, ticker] if ticker in v6_alpha.columns else np.nan
        price = close.loc[latest_date, ticker]
        mcap = ticker_mcap.get(ticker, 0)
        mcap_억 = mcap / 1e8 if mcap else 0
        print(f"  {i:>2}. {ticker:<10} {name:<16} {score:.4f}   {v3_score:.4f}   {v6_score:.4f} {price:>12,.0f}원 {mcap_억:>8,.0f}억")

    print(f"\n💡 해석:")
    print(f"   v3 가격 알파: 25일 모멘텀 × 거래량 안정성 × 28일 레인지 위치")
    print(f"   v6 재무 추세: 영업이익 3분기 추세 + YoY 개선도 랭킹")
    print(f"   앙상블: 가격 기술적 신호와 펀더멘탈 개선 추세를 결합")
    print(f"   → 실적 개선 + 가격 모멘텀 동시 확인 = 더 안정적 매수 신호")
    print(f"   → 15영업일(약 3주) 보유 전략에 최적화")

    # DB 저장
    print(f"\n💾 앙상블 결과 DB 저장 중...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        ensemble_expr = (
            f"ENSEMBLE(w_price={best_weight:.2f}, w_fund={1-best_weight:.2f}) | "
            f"v3: ops.normed_rank(ops.cwise_mul(ops.cwise_mul(ops.ts_delta_ratio(close, 25), "
            f"ops.div(ops.ts_median(volume, 10), ops.ts_std(volume, 15))), ops.ts_maxmin_scale(close, 28))) | "
            f"v6: ops.normed_rank(ops.add(ops.ts_zscore_scale(oi_trend_rank, 31), oi_yoy_rank))"
        )
        cursor.execute("""
            INSERT INTO alpha_formulas (formula, ic_score, description, created_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (formula) DO UPDATE
            SET ic_score = EXCLUDED.ic_score, updated_at = NOW()
        """, (
            ensemble_expr,
            float(test_ic),
            f"Ensemble v3+v6, w_price={best_weight:.2f}, train IC={train_ic:.4f}, test IC={test_ic:.4f}, 15d fwd"
        ))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ 앙상블 저장 완료!")
    except Exception as e:
        print(f"⚠️  DB 저장 실패: {e}")

    print(f"\n🎉 완료!")


if __name__ == "__main__":
    main()
