#!/usr/bin/env python3
"""
Alpha-GPT: 15-day Forward with GPT-4o (v2 — Improved Prompt)
개선된 QuantDeveloper 프롬프트 + ops.xxx() 문법 + 병렬 GP
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
import gc
from multiprocessing import Pool

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alpha_gpt_kr.mining.operators import AlphaOperators as ops
from alpha_gpt_kr.agents.quant_developer import QuantDeveloper

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', '192.168.0.248'),
        port=int(os.getenv('DB_PORT', 5432)),
        database=os.getenv('DB_NAME', 'marketsense'),
        user=os.getenv('DB_USER', 'yrbahn'),
        password=os.getenv('DB_PASSWORD', '1234')
    )

def load_market_data():
    """500개 종목 데이터 로드 (시가총액 상위)"""
    print("📊 데이터 로드 중... (시총 상위 500종목, 2년)")
    
    conn = get_db_connection()
    
    # 시가총액 상위 500개
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
            AND p.date >= CURRENT_DATE - INTERVAL '730 days'
            LIMIT 1
        )
        ORDER BY s.market_cap DESC
        LIMIT 500
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
        AND p.date >= CURRENT_DATE - INTERVAL '730 days'
        ORDER BY s.ticker, p.date
    """

    price_df = pd.read_sql(query_prices, conn)

    open_price = price_df.pivot(index='date', columns='ticker', values='open')
    high = price_df.pivot(index='date', columns='ticker', values='high')
    low = price_df.pivot(index='date', columns='ticker', values='low')
    close = price_df.pivot(index='date', columns='ticker', values='close')
    volume = price_df.pivot(index='date', columns='ticker', values='volume')
    returns = close.pct_change()

    # 파생 변수 (OHLC)
    vwap = (high + low + close) / 3
    high_low_range = (high - low) / close
    body = (close - open_price) / open_price
    upper_shadow = (high - close.clip(lower=open_price)) / close
    lower_shadow = (close.clip(upper=open_price) - low) / close

    # ── 재무 4분기 추세 변수 ──
    print("   재무 추세 데이터 로드 중...")
    import json as _json
    id_ticker = dict(zip(stocks_df['id'], stocks_df['ticker']))

    fin_df = pd.read_sql(f"""
        SELECT stock_id, period_end, revenue, operating_income, net_income,
               total_equity, total_assets, raw_data
        FROM financial_statements
        WHERE stock_id IN ({stock_id_list})
        ORDER BY stock_id, period_end
    """, conn)
    conn.close()

    # raw_data에서 ROE, 영업이익률 추출
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

    # 연간(12-31) 제외 → standalone quarterly만 사용
    fin_df = fin_df[fin_df['quarter_type'] != '연간'].copy()

    # 영업이익률 계산
    fin_df['op_margin'] = np.where(
        (fin_df['revenue'].notna()) & (fin_df['revenue'] != 0),
        fin_df['operating_income'] / fin_df['revenue'],
        np.nan
    )

    # 분기 인덱스 (정렬용)
    fin_df = fin_df.sort_values(['ticker', 'period_end'])

    # 각 종목별 QoQ/YoY 추세 계산
    trend_records = []
    for ticker, grp in fin_df.groupby('ticker'):
        grp = grp.sort_values('period_end').reset_index(drop=True)
        for i in range(len(grp)):
            row = grp.iloc[i]
            rec = {'ticker': ticker, 'period_end': row['period_end']}

            # QoQ (전분기 대비)
            if i >= 1:
                prev = grp.iloc[i - 1]
                if prev['operating_income'] and prev['operating_income'] != 0:
                    rec['oi_qoq'] = (row['operating_income'] - prev['operating_income']) / abs(prev['operating_income'])
                if prev['net_income'] and prev['net_income'] != 0:
                    rec['ni_qoq'] = (row['net_income'] - prev['net_income']) / abs(prev['net_income'])
                if row['op_margin'] is not None and prev['op_margin'] is not None:
                    rec['margin_qoq'] = row['op_margin'] - prev['op_margin']
                if row['roe'] is not None and prev['roe'] is not None:
                    rec['roe_qoq'] = row['roe'] - prev['roe']

            # YoY (4분기 전 = 같은 분기 전년) - Q1/Q2/Q3 순서로 3분기씩이므로 3칸 뒤
            if i >= 3:
                yoy_prev = grp.iloc[i - 3]
                # 같은 분기인지 확인 (3/31↔3/31, 6/30↔6/30, 9/30↔9/30)
                if row['period_end'].month == yoy_prev['period_end'].month:
                    if yoy_prev['operating_income'] and yoy_prev['operating_income'] != 0:
                        rec['oi_yoy'] = (row['operating_income'] - yoy_prev['operating_income']) / abs(yoy_prev['operating_income'])
                    if yoy_prev['net_income'] and yoy_prev['net_income'] != 0:
                        rec['ni_yoy'] = (row['net_income'] - yoy_prev['net_income']) / abs(yoy_prev['net_income'])
                    if row['op_margin'] is not None and yoy_prev['op_margin'] is not None:
                        rec['margin_yoy'] = row['op_margin'] - yoy_prev['op_margin']
                    if row['roe'] is not None and yoy_prev['roe'] is not None:
                        rec['roe_yoy'] = row['roe'] - yoy_prev['roe']

            # 3분기 추세 기울기 (최근 3분기 OI의 선형 추세)
            if i >= 2:
                oi_vals = [grp.iloc[j]['operating_income'] for j in range(i - 2, i + 1)
                           if grp.iloc[j]['operating_income'] is not None and not np.isnan(grp.iloc[j]['operating_income'])]
                if len(oi_vals) == 3:
                    # 간단한 선형 기울기: (last - first) / 2
                    rec['oi_trend'] = (oi_vals[2] - oi_vals[0]) / (abs(oi_vals[0]) + 1e-10)

            trend_records.append(rec)

    trend_df = pd.DataFrame(trend_records)

    # 일별 데이터로 변환: 분기별 → 일별 forward-fill, cross-sectional rank
    trend_vars = {}
    trend_fields = ['oi_qoq', 'ni_qoq', 'oi_yoy', 'ni_yoy', 'margin_yoy', 'roe_yoy', 'oi_trend']
    trend_field_names = []

    for field in trend_fields:
        if field not in trend_df.columns:
            continue
        pivot = trend_df.pivot_table(index='period_end', columns='ticker', values=field, aggfunc='last')
        if pivot.empty or pivot.notna().sum().sum() < 50:
            continue
        # 일별 reindex + forward-fill
        daily = pivot.reindex(close.index).ffill()
        daily = daily.reindex(columns=close.columns)
        # cross-sectional rank [0,1] (매일)
        ranked = daily.rank(axis=1, pct=True)
        var_name = f'{field}_rank'
        trend_vars[var_name] = ranked
        trend_field_names.append(var_name)

    print(f"✅ {len(close.columns)}개 종목, {len(close)}일 데이터")
    print(f"   가격 변수: close, open_price, high, low, volume, returns, vwap, high_low_range, body")
    if trend_field_names:
        print(f"   재무 추세 변수 ({len(trend_field_names)}개): {', '.join(trend_field_names)}")
    else:
        print(f"   ⚠️  재무 추세 변수 생성 실패")

    result = {
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
    }
    result.update(trend_vars)
    return result

def generate_seed_alphas_gpt4o(num_seeds=20):
    """GPT-4o + 개선된 QuantDeveloper 프롬프트로 시드 알파 생성"""
    print(f"\n🤖 GPT-4o로 초기 알파 {num_seeds}개 생성 중 (개선된 프롬프트)...")

    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # QuantDeveloper의 개선된 프롬프트 재사용
    system_prompt = QuantDeveloper.SYSTEM_PROMPT

    # 15일 보유에 특화된 user prompt
    prompt = f"""### Task
Generate {num_seeds} diverse, high-performance alpha expressions optimized for **15-day forward returns** in the Korean stock market.

### Trading Idea
15일 보유 전략에 최적화된 중기 알파 팩터. 단기 노이즈를 필터링하고,
15일 후 수익률과 높은 상관관계(IC)를 가지는 시그널을 찾아야 함.
모멘텀, 거래량, 변동성, 추세 강도를 조합하여 다양한 팩터를 생성.

### Available Data Fields
close, open_price, high, low, volume, returns, vwap, high_low_range, body, upper_shadow, lower_shadow,
oi_yoy_rank, ni_yoy_rank, oi_qoq_rank, oi_trend_rank, margin_yoy_rank, roe_yoy_rank

**Price variables** (daily time-series):
- `close`, `open_price`, `high`, `low`: OHLC prices
- `volume`: trading volume
- `returns`: daily close-to-close returns
- `vwap`: (high + low + close) / 3
- `high_low_range`, `body`, `upper_shadow`, `lower_shadow`: candle pattern ratios

**Fundamental TREND rank variables** (cross-sectional rank [0,1], higher = more improvement):
- `oi_yoy_rank`: 영업이익 YoY 변화율 순위 (전년 동분기 대비 개선도)
- `ni_yoy_rank`: 순이익 YoY 변화율 순위
- `oi_qoq_rank`: 영업이익 QoQ 변화율 순위 (전분기 대비 개선도)
- `oi_trend_rank`: 영업이익 3분기 추세 기울기 순위
- `margin_yoy_rank`: 영업이익률 YoY 변화 순위
- `roe_yoy_rank`: ROE YoY 변화 순위

**IMPORTANT for fundamental trend variables**:
- These capture IMPROVEMENT (not level) — "ROE가 개선중인 종목" not "ROE가 높은 종목"
- Already cross-sectionally ranked [0,1], do NOT apply normed_rank() again
- Use as weights with price signals: `ops.cwise_mul(ops.ts_delta_ratio(close, 15), oi_yoy_rank)` — momentum × earnings improvement
- DO NOT apply ts_delta/ts_delta_ratio/ts_corr on them (they change quarterly, will create artifacts)

### Requirements

**Diversity** — Each alpha MUST belong to a DIFFERENT category:
  1. `ma_golden_cross` — Moving average crossover: `ops.div(ops.ts_mean(close, 5), ops.ts_mean(close, 20))` × fundamental rank (PROVEN: IC 0.039)
  2. `ma_distance` — Price distance from long-term MA: `ops.div(close, ops.ts_mean(close, 120))` (PROVEN: IC 0.037)
  3. `ma_slope` — MA slope/trend strength: `ops.ts_delta_ratio(ops.ts_mean(close, 60), 10)` (PROVEN: IC 0.032)
  4. `ma_multi_volume` — Multiple MA crossover + volume confirmation (PROVEN: IC 0.031)
  5. `momentum_volume` — Momentum confirmed by volume surge
  6. `volatility_adjusted` — Signal adjusted/filtered by volatility (use high_low_range)
  7. `short_term_reversal` — Mean-reversion exploiting KRX reversal effect
  8. `multi_timeframe` — Combining short + medium + long timeframes (use 5/20/60/120 windows)
  9. `price_volume_diverge` — Price-volume divergence / smart money
  10. `trend_strength` — Trend strength via regression slope or IR
  11. `earnings_momentum` — Price momentum × earnings improvement (oi_yoy_rank, oi_qoq_rank)
  12. `price_position` — Price position relative to recent high/low
  13. `volume_anomaly` — Abnormal volume detection
  14. `earnings_reversal` — Oversold + earnings improving (reversal × oi_yoy_rank)
  15. `quality_momentum` — Momentum weighted by margin improvement (margin_yoy_rank)
  16. `roe_improvement` — Price × ROE improvement (roe_yoy_rank)
  17. `trend_confirmation` — Price trend + earnings trend aligned (oi_trend_rank)
  18. `candle_pattern` — Candle body/shadow patterns
  19. `ma_earnings_composite` — MA signal × multiple fundamental ranks (3+ factors)
  20. `composite_all` — Most complex: MA + price + volume + fundamental trend

**15-Day Holding Optimization**:
- **Moving averages are highly effective** — use `ops.ts_mean(close, N)` for MA(N), `ops.div(close, ops.ts_mean(close, N))` for distance from MA
- Use diverse lookback windows: 5, 10, 15, 20, 30, 60, 120 days
- Combine at least 2 timeframes per alpha
- Volume confirmation is critical for 15-day predictions
- **Multiplicative combination with fundamental ranks works better than additive**: use `ops.cwise_mul(price_signal, oi_trend_rank)`

**Quality Checklist** — Every alpha must satisfy ALL:
- [ ] Multi-factor: combines 2+ distinct signal types
- [ ] Market-neutral: wrapped with `ops.normed_rank()` or `ops.zscore_scale()`
- [ ] Multi-timeframe: uses 2+ lookback windows
- [ ] No look-ahead bias
- [ ] Complexity 2~4 nesting levels
- [ ] Safe division: use `ops.div()` instead of raw `/`

### Output Format
Return a JSON array:
```json
[
  {{
    "alpha_name": "Alpha_Name",
    "category": "category_name",
    "rationale": "Economic logic explanation",
    "expression": "ops.normed_rank(...)",
    "complexity": 4,
    "operators_used": ["op1", "op2"],
    "timeframes_used": [10, 20]
  }}
]
```

**CRITICAL**:
- You MUST return a JSON object with key "alphas" containing an array of {num_seeds} alpha objects.
- Format: {{"alphas": [{{...}}, {{...}}, ...]}}
- Each object MUST have "expression" field with valid ops.xxx() Python code.
- Generate ALL {num_seeds} alphas. Do NOT return just 1."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=16000,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content
    print(f"   GPT-4o 응답 길이: {len(content)}자")
    print(f"   응답 미리보기: {content[:200]}...")

    # JSON 파싱
    alphas = []
    try:
        data = json.loads(content)
        print(f"   파싱된 타입: {type(data).__name__}")

        # dict → 리스트 추출
        if isinstance(data, dict):
            print(f"   키 목록: {list(data.keys())}")

            # 1순위: dict 자체가 단일 알파인 경우 (expression 키 존재)
            if 'expression' in data or 'expr' in data:
                data = [data]
                print(f"   단일 알파 dict → 리스트로 변환")

            else:
                # 2순위: {"alphas": [{...}, ...]} 형태 — dict 리스트를 가진 키 찾기
                found_list = False
                for key in data:
                    if isinstance(data[key], list) and data[key] and isinstance(data[key][0], dict):
                        data = data[key]
                        print(f"   '{key}' 키에서 {len(data)}개 항목 추출")
                        found_list = True
                        break

                if not found_list:
                    # 3순위: 중첩 dict: {"alpha_1": {...}, "alpha_2": {...}}
                    items = []
                    for key, val in data.items():
                        if isinstance(val, dict) and ('expression' in val or 'expr' in val):
                            items.append(val)
                    if items:
                        data = items
                        print(f"   중첩 dict에서 {len(items)}개 항목 추출")
                    else:
                        print(f"   ⚠️  알 수 없는 dict 구조: {list(data.keys())[:5]}")
                        data = []

        for item in data:
            if isinstance(item, str):
                if 'ops.' in item:
                    alphas.append(item)
                continue
            if not isinstance(item, dict):
                continue
            expr = item.get('expression', item.get('expr', ''))
            if expr and 'ops.' in expr:
                alphas.append(expr)
            elif expr:
                print(f"   ⚠️  ops. 없는 표현식 스킵: {expr[:80]}")

    except (json.JSONDecodeError, Exception) as e:
        print(f"⚠️  JSON 파싱 실패: {e}")
        # 마크다운 코드블록 안의 JSON 추출 시도
        json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                for item in data:
                    if isinstance(item, dict):
                        expr = item.get('expression', item.get('expr', ''))
                        if expr and 'ops.' in expr:
                            alphas.append(expr)
                print(f"   마크다운 블록에서 {len(alphas)}개 복구")
            except Exception:
                pass

    # 폴백: 개선된 복합 팩터 + MA 기반 알파
    if len(alphas) < 10:
        print(f"⚠️  {len(alphas)}개만 파싱됨, 폴백 추가")
        fallback = [
            # ── MA 기반 알파 (검증 완료, Test IC 0.03~0.04) ──
            # 골든크로스(5/20) × 영업이익 추세 (Test IC 0.0391)
            "ops.normed_rank(ops.cwise_mul(ops.div(ops.ts_mean(close, 5), ops.ts_mean(close, 20)), oi_trend_rank))",
            # 120일 이격도 (Test IC 0.0369)
            "ops.normed_rank(ops.div(close, ops.ts_mean(close, 120)))",
            # 60일 MA 기울기 (Test IC 0.0323)
            "ops.normed_rank(ops.ts_delta_ratio(ops.ts_mean(close, 60), 10))",
            # 다중 MA 종합: 골든크로스 × 거래량 + MA기울기 (Test IC 0.0311)
            "ops.normed_rank(ops.add(ops.cwise_mul(ops.div(ops.ts_mean(close, 5), ops.ts_mean(close, 20)), ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20))), ops.ts_delta_ratio(ops.ts_mean(close, 20), 10)))",
            # 이격도 × 거래량 (Test IC 0.0288)
            "ops.normed_rank(ops.cwise_mul(ops.div(close, ops.ts_mean(close, 20)), ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20))))",
            # 골든크로스 × 거래량 (Test IC 0.0280)
            "ops.normed_rank(ops.cwise_mul(ops.div(ops.ts_mean(close, 5), ops.ts_mean(close, 20)), ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20))))",
            # MA이격 120d × oi_yoy (Test IC 0.0233)
            "ops.normed_rank(ops.cwise_mul(ops.div(close, ops.ts_mean(close, 120)), oi_yoy_rank))",
            # MA기울기 60d × oi_yoy (Test IC 0.0237)
            "ops.normed_rank(ops.cwise_mul(ops.ts_delta_ratio(ops.ts_mean(close, 60), 10), oi_yoy_rank))",
            # ── 기존 검증된 팩터 ──
            "ops.normed_rank(ops.cwise_mul(ops.cwise_mul(ops.ts_delta_ratio(close, 25), ops.div(ops.ts_median(volume, 10), ops.ts_std(volume, 15))), ops.ts_maxmin_scale(close, 28)))",
            "ops.normed_rank(ops.cwise_mul(ops.ts_delta_ratio(close, 15), ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20))))",
            "ops.normed_rank(ops.neg(ops.ts_corr(ops.ts_delta(close, 5), ops.ts_delta(volume, 5), 20)))",
            # ── 가격 × 재무 추세 복합 팩터 ──
            "ops.normed_rank(ops.cwise_mul(ops.ts_delta_ratio(close, 15), oi_yoy_rank))",
            "ops.normed_rank(ops.cwise_mul(ops.neg(ops.ts_zscore_scale(close, 15)), oi_qoq_rank))",
            "ops.normed_rank(ops.cwise_mul(ops.ts_maxmin_scale(close, 20), oi_trend_rank))",
            "ops.normed_rank(ops.cwise_mul(ops.ts_delta_ratio(close, 20), margin_yoy_rank))",
            "ops.normed_rank(ops.cwise_mul(ops.cwise_mul(ops.ts_delta_ratio(close, 15), oi_yoy_rank), ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20))))",
            "ops.normed_rank(ops.cwise_mul(ops.ts_linear_reg(close, 20), roe_yoy_rank))",
            "ops.normed_rank(ops.cwise_mul(ops.ts_delta_ratio(close, 20), ops.neg(ops.ts_mean(high_low_range, 15))))",
            "ops.normed_rank(ops.div(ops.neg(ops.ts_zscore_scale(close, 10)), ops.ts_std(returns, 20)))",
            "ops.normed_rank(ops.minus(ops.ts_ir(returns, 5), ops.ts_ir(returns, 20)))",
        ]
        alphas = alphas + [f for f in fallback if f not in alphas]

    print(f"✅ {len(alphas)}개 초기 알파 생성")
    for i, a in enumerate(alphas[:5], 1):
        print(f"   {i}. {a[:80]}...")

    return alphas[:num_seeds]

# 전역 데이터
_global_data = None

def set_global_data(data):
    global _global_data
    _global_data = data

def _compute_raw_ic(alpha_expr, data):
    """알파의 raw IC 계산 (train 또는 test 데이터)"""
    close = data['close']
    open_price = data['open_price']
    high = data['high']
    low = data['low']
    volume = data['volume']
    returns = data['returns']
    vwap = data['vwap']
    high_low_range = data['high_low_range']
    body = data['body']
    upper_shadow = data.get('upper_shadow', (high - close.clip(lower=open_price)) / close)
    lower_shadow = data.get('lower_shadow', (close.clip(upper=open_price) - low) / close)
    # 재무 추세 rank 변수 (0~1, cross-sectional rank, 높을수록 개선 중)
    _empty = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    oi_yoy_rank = data.get('oi_yoy_rank', _empty)
    ni_yoy_rank = data.get('ni_yoy_rank', _empty)
    oi_qoq_rank = data.get('oi_qoq_rank', _empty)
    ni_qoq_rank = data.get('ni_qoq_rank', _empty)
    oi_trend_rank = data.get('oi_trend_rank', _empty)
    margin_yoy_rank = data.get('margin_yoy_rank', _empty)
    roe_yoy_rank = data.get('roe_yoy_rank', _empty)

    forward_return_15d = close.shift(-15) / close - 1
    alpha_values = eval(alpha_expr)

    if not isinstance(alpha_values, pd.DataFrame):
        return -999.0

    ic_list = []
    for date in alpha_values.index[:-15]:
        alpha_cs = alpha_values.loc[date]
        returns_cs = forward_return_15d.loc[date]
        valid = alpha_cs.notna() & returns_cs.notna()

        if valid.sum() > 30:
            ic = alpha_cs[valid].corr(returns_cs[valid])
            if not np.isnan(ic):
                ic_list.append(ic)

    if len(ic_list) < 10:
        return -999.0

    return float(np.mean(ic_list))

def _multi_factor_bonus(alpha_expr):
    """다중 팩터 구조 보너스"""
    bonus = 0.0
    trend_vars = ['oi_yoy_rank', 'ni_yoy_rank', 'oi_qoq_rank', 'ni_qoq_rank', 'oi_trend_rank', 'margin_yoy_rank', 'roe_yoy_rank']

    # rank 변수에 ts_delta/ts_corr → 강한 페널티 (분기 경계 artifacts)
    for rv in trend_vars:
        if rv in alpha_expr:
            if re.search(rf'ts_(delta|delta_ratio|corr|cov)\([^)]*{rv}', alpha_expr):
                return -0.05

    # 거래량 사용 보너스
    if 'volume' in alpha_expr:
        bonus += 0.002
    # MA 구조 보너스 (검증된 패턴)
    has_ma = bool(re.search(r'ts_mean\([^)]*close[^)]*,\s*\d+\)', alpha_expr))
    if has_ma:
        bonus += 0.002
        # MA + 재무추세 곱셈 결합 보너스 (가장 강력한 패턴)
        has_trend = any(tv in alpha_expr for tv in trend_vars)
        if has_trend and 'cwise_mul' in alpha_expr:
            bonus += 0.003
    # 가격+추세 결합 보너스 (다중팩터 장려)
    has_price_ts = bool(re.search(r'ts_\w+\([^)]*(?:close|open_price|high|low|volume|returns)', alpha_expr))
    has_trend = any(tv in alpha_expr for tv in trend_vars)
    if has_price_ts and has_trend:
        bonus += 0.003
    # 다중 타임프레임 보너스 (윈도우 차이 ≥ 2배)
    windows = [int(w) for w in re.findall(r',\s*(\d+)\)', alpha_expr)]
    if len(windows) >= 2 and max(windows) >= min(windows) * 2:
        bonus += 0.002
    # 장기 윈도우 보너스 (60일+ 사용 시)
    if windows and max(windows) >= 60:
        bonus += 0.001
    # 복잡도 페널티
    depth = alpha_expr.count('(')
    if depth < 3:
        bonus -= 0.002
    if depth > 10:
        bonus -= 0.003
    return bonus

def evaluate_alpha_worker(alpha_expr):
    """병렬 처리용 알파 평가 — train IC + 다중팩터 보너스"""
    global _global_data
    data = _global_data

    try:
        raw_ic = _compute_raw_ic(alpha_expr, data)
        if raw_ic <= -999.0:
            return (alpha_expr, -999.0)
        bonus = _multi_factor_bonus(alpha_expr)
        return (alpha_expr, raw_ic + bonus)
    except Exception:
        return (alpha_expr, -999.0)

def evaluate_alpha_oos(alpha_expr, test_data):
    """Out-of-sample IC 계산 (보너스 없이 순수 IC)"""
    try:
        return _compute_raw_ic(alpha_expr, test_data)
    except Exception:
        return -999.0

# 연산자 교환 그룹 (같은 시그니처끼리만 교체)
OPERATOR_SWAP_GROUPS = [
    ['ts_mean', 'ts_std', 'ts_median', 'ts_ema', 'ts_linear_reg', 'ts_decayed_linear'],
    ['ts_zscore_scale', 'ts_maxmin_scale', 'ts_rank'],
    ['ts_delta', 'ts_delta_ratio'],
    ['ts_skew', 'ts_kurt', 'ts_ir'],
    ['ts_min', 'ts_max'],
    ['ts_argmin', 'ts_argmax'],
    ['ts_max_diff', 'ts_min_diff'],
    ['normed_rank', 'zscore_scale'],
    ['cwise_mul', 'add', 'minus'],
]

OPERAND_POOL = ['close', 'open_price', 'high', 'low', 'volume', 'returns', 'vwap', 'high_low_range', 'body',
                'oi_yoy_rank', 'ni_yoy_rank', 'oi_qoq_rank', 'oi_trend_rank', 'margin_yoy_rank', 'roe_yoy_rank']

def mutate_alpha(alpha_expr):
    """알파 변이 — 3가지 타입: 윈도우(50%), 연산자(30%), 피연산자(20%)"""
    try:
        mutation_type = random.choices(
            ['window', 'operator', 'operand'],
            weights=[0.5, 0.3, 0.2]
        )[0]

        if mutation_type == 'window':
            return _mutate_window(alpha_expr)
        elif mutation_type == 'operator':
            return _mutate_operator(alpha_expr)
        else:
            return _mutate_operand(alpha_expr)
    except Exception:
        return None

def _mutate_window(alpha_expr):
    """윈도우 파라미터 변경 (범위 5~120, MA 장기 시그널 지원)"""
    matches = list(re.finditer(r'(ts_\w+|shift)\([^,]+,\s*(\d+)\)', alpha_expr))
    if not matches:
        return None
    match = random.choice(matches)
    old_window = int(match.group(2))
    # 현재 윈도우 크기에 따라 변이 폭 조절 (비례적 변이)
    if old_window <= 20:
        deltas = [-5, -3, -2, 2, 3, 5, 7, 10, 15]
    elif old_window <= 60:
        deltas = [-15, -10, -7, -5, 5, 7, 10, 15, 20, 30]
    else:
        deltas = [-30, -20, -10, 10, 20, 30]
    new_window = max(5, min(120, old_window + random.choice(deltas)))
    if new_window == old_window:
        new_window = max(5, min(120, old_window + random.choice([-20, 20])))
    start, end = match.span(2)
    return alpha_expr[:start] + str(new_window) + alpha_expr[end:]

def _mutate_operator(alpha_expr):
    """연산자 교체 — 같은 시그니처 그룹 내에서만"""
    # 현재 표현식에서 연산자 추출
    op_matches = list(re.finditer(r'ops\.(\w+)\(', alpha_expr))
    if not op_matches:
        return None

    # 교환 가능한 연산자만 필터
    swappable = []
    for m in op_matches:
        op_name = m.group(1)
        for group in OPERATOR_SWAP_GROUPS:
            if op_name in group:
                swappable.append((m, op_name, group))
                break

    if not swappable:
        return _mutate_window(alpha_expr)  # 교환 불가면 윈도우 변이

    match, old_op, group = random.choice(swappable)
    candidates = [op for op in group if op != old_op]
    if not candidates:
        return _mutate_window(alpha_expr)

    new_op = random.choice(candidates)
    start, end = match.span(1)
    return alpha_expr[:start] + new_op + alpha_expr[end:]

def _mutate_operand(alpha_expr):
    """피연산자 교체 — close/volume/returns 간 교환"""
    present = [op for op in OPERAND_POOL if op in alpha_expr]
    if not present:
        return _mutate_window(alpha_expr)

    old_operand = random.choice(present)
    candidates = [op for op in OPERAND_POOL if op != old_operand]
    new_operand = random.choice(candidates)

    # 첫 번째 등장만 교체 (전체 교체 방지)
    return alpha_expr.replace(old_operand, new_operand, 1)


def _subtree_crossover(alpha1, alpha2):
    """서브트리 교차 — 한 알파의 서브트리를 다른 알파의 서브트리로 교체"""
    try:
        # ops.xxx(...) 패턴의 서브트리 추출
        def find_subtrees(expr):
            """괄호 매칭으로 ops.xxx(...) 서브트리 위치 찾기"""
            subtrees = []
            for m in re.finditer(r'ops\.\w+\(', expr):
                start = m.start()
                depth = 0
                for i in range(m.end() - 1, len(expr)):
                    if expr[i] == '(':
                        depth += 1
                    elif expr[i] == ')':
                        depth -= 1
                    if depth == 0:
                        subtrees.append((start, i + 1, expr[start:i+1]))
                        break
            return subtrees

        trees1 = find_subtrees(alpha1)
        trees2 = find_subtrees(alpha2)

        if len(trees1) < 2 or not trees2:
            return None

        # alpha1에서 교체할 서브트리 선택 (최상위 제외)
        replaceable = [t for t in trees1 if t[2] != alpha1]
        if not replaceable:
            return None

        target = random.choice(replaceable)
        donor = random.choice(trees2)

        result = alpha1[:target[0]] + donor[2] + alpha1[target[1]:]
        # 유효성 검사: ops.가 있고 괄호가 맞는지
        if result.count('(') != result.count(')') or 'ops.' not in result:
            return None
        return result
    except Exception:
        return None


def crossover_alphas(alpha1, alpha2):
    """알파 교차 — 윈도우 교환(60%) + 서브트리 교차(40%)"""
    try:
        # 40% 확률로 서브트리 교차 시도
        if random.random() < 0.4:
            result = _subtree_crossover(alpha1, alpha2)
            if result:
                return result

        matches1 = list(re.finditer(r'(ts_\w+|shift)\(([^,]+),\s*(\d+)\)', alpha1))
        matches2 = list(re.finditer(r'(ts_\w+|shift)\(([^,]+),\s*(\d+)\)', alpha2))

        if not matches1 or not matches2:
            return None

        # 같은 연산자가 있으면 우선 교차
        ops1 = {m.group(1): m for m in matches1}
        ops2 = {m.group(1): m for m in matches2}
        common_ops = set(ops1.keys()) & set(ops2.keys())

        if common_ops:
            op = random.choice(list(common_ops))
            m1, m2 = ops1[op], ops2[op]
        else:
            m1 = random.choice(matches1)
            m2 = random.choice(matches2)

        # alpha1의 윈도우를 alpha2의 값으로 교체
        win2 = m2.group(3)
        start, end = m1.span(3)
        return alpha1[:start] + win2 + alpha1[end:]
    except Exception:
        return None

def _get_alpha_structure(alpha_expr):
    """알파의 구조 시그니처 (윈도우 제거) — 다양성 비교용"""
    return re.sub(r',\s*\d+\)', ', N)', alpha_expr)

def _select_diverse_top_n(results, n=5):
    """IC 상위에서 구조가 다른 Top-N 선택"""
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    selected = []
    seen_structures = set()

    for alpha, ic in sorted_results:
        if ic <= -999.0:
            continue
        structure = _get_alpha_structure(alpha)
        if structure not in seen_structures:
            selected.append((alpha, ic))
            seen_structures.add(structure)
            if len(selected) >= n:
                break

    return selected

def _tournament_select(fitness_scores, tournament_size=5):
    """토너먼트 선택 — 다양성 유지하면서 우수 개체 선호"""
    candidates = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
    return max(candidates, key=lambda x: x[1])[0]


def _fitness_sharing(fitness_scores, sharing_radius=0.8):
    """적합도 공유 — 같은 구조의 알파끼리 fitness를 나눠 다양성 보존"""
    structures = {}
    for alpha, ic in fitness_scores:
        struct = _get_alpha_structure(alpha)
        if struct not in structures:
            structures[struct] = []
        structures[struct].append((alpha, ic))

    shared = []
    for struct, members in structures.items():
        niche_size = len(members)
        for alpha, ic in members:
            # 같은 구조가 많을수록 fitness 감소 (niche pressure)
            shared_ic = ic / (1.0 + sharing_radius * (niche_size - 1))
            shared.append((alpha, shared_ic))

    return sorted(shared, key=lambda x: x[1], reverse=True)


def genetic_programming(seed_alphas, data, generations=50, population_size=200):
    """최적화된 병렬 GP — 이민 + 토너먼트 선택 + 적합도 공유 + 적응적 변이"""

    print(f"\n🧬 병렬 GP 시작 (v2 최적화)")
    print(f"   Seed: {len(seed_alphas)}개, 세대: {generations}, 개체수: {population_size}, 워커: 4")

    population = seed_alphas[:population_size]
    while len(population) < population_size:
        parent = random.choice(seed_alphas)
        mutated = mutate_alpha(parent)
        if mutated:
            population.append(mutated)

    set_global_data(data)
    best_ever = (None, -999.0)
    stagnation_count = 0
    immigration_count = 0
    all_results_history = []

    elite_count = max(5, population_size // 14)  # 7% 엘리트 (수렴 지연)
    base_mutation_rate = 0.45  # 기본 변이율

    for gen in range(1, generations + 1):
        # 적응적 변이율: 정체 시 변이 비중 증가
        mutation_rate = min(0.7, base_mutation_rate + stagnation_count * 0.05)
        crossover_rate = 1.0 - mutation_rate

        print(f"\n  세대 {gen}/{generations} (변이율: {mutation_rate:.0%}, 정체: {stagnation_count})")

        with Pool(4, initializer=set_global_data, initargs=(data,)) as pool:
            results = pool.map(evaluate_alpha_worker, population)

        # 적합도 공유 적용 (같은 구조끼리 fitness 분산)
        raw_scores = sorted(results, key=lambda x: x[1], reverse=True)
        fitness_scores = _fitness_sharing(raw_scores)
        all_results_history.extend([(a, ic) for a, ic in raw_scores if ic > -999.0])

        best_ic = raw_scores[0][1]  # 공유 전 실제 IC
        median_ic = raw_scores[len(raw_scores)//2][1] if raw_scores else -999.0
        unique_structures = len(set(_get_alpha_structure(a) for a, _ in raw_scores if _ > -999.0))
        print(f"    최고 IC: {best_ic:.4f}  중앙값: {median_ic:.4f}  고유구조: {unique_structures}개")

        if best_ic > best_ever[1]:
            best_ever = raw_scores[0]
            stagnation_count = 0
            print(f"    🏆 신기록!")
        else:
            stagnation_count += 1

        # 이민(immigration): 정체 시 새로운 개체 주입 (조기종료 대신)
        if stagnation_count >= 5 and immigration_count < 3:
            immigration_count += 1
            stagnation_count = 0
            n_immigrants = population_size // 4  # 25% 교체
            print(f"    🌍 이민 #{immigration_count}: {n_immigrants}개 새 개체 주입")
            # 시드에서 새 변이 생성
            immigrants = []
            for _ in range(n_immigrants):
                parent = random.choice(seed_alphas)
                # 2-3회 연속 변이로 다양성 극대화
                for _ in range(random.randint(2, 3)):
                    m = mutate_alpha(parent)
                    if m:
                        parent = m
                immigrants.append(parent)
            # 하위 25% 교체
            population = [a for a, _ in fitness_scores[:population_size - n_immigrants]] + immigrants
            continue

        # 최종 종료: 이민 3회 후에도 5세대 무개선
        if stagnation_count >= 5:
            print(f"    ⏹️  이민 {immigration_count}회 후 5세대 무개선 → 종료")
            break

        # 다음 세대 구성
        next_population = []

        # 엘리트 보존 (7%)
        for alpha, _ in fitness_scores[:elite_count]:
            next_population.append(alpha)

        # 토너먼트 선택 + 교차/변이
        while len(next_population) < population_size:
            if random.random() < crossover_rate:
                # 토너먼트 선택으로 부모 2개 선택 → 교차
                parent1 = _tournament_select(fitness_scores, tournament_size=5)
                parent2 = _tournament_select(fitness_scores, tournament_size=5)
                child = crossover_alphas(parent1, parent2)
                if child:
                    next_population.append(child)
                else:
                    next_population.append(parent1)
            else:
                # 토너먼트 선택 → 변이
                parent = _tournament_select(fitness_scores, tournament_size=5)
                mutated = mutate_alpha(parent)
                if mutated:
                    next_population.append(mutated)
                else:
                    next_population.append(parent)

        population = next_population[:population_size]

        del results, raw_scores, fitness_scores, next_population
        gc.collect()

    # Top-5 다양한 알파 선택
    top_diverse = _select_diverse_top_n(all_results_history, n=5)

    return best_ever, top_diverse

def main():
    print("=" * 80)
    print("Alpha-GPT: 15-day Forward with GPT-4o (v6 — Price + Fundamental Trend)")
    print("=" * 80)
    print()

    # 1. 전체 데이터 로드
    full_data = load_market_data()

    # 2. Train/Test 분할 (70/30)
    close = full_data['close']
    split_idx = int(len(close) * 0.7)
    split_date = close.index[split_idx]
    print(f"\n📐 Train/Test 분할: {split_idx}일 train / {len(close) - split_idx}일 test")
    print(f"   Train: ~{close.index[0]} ~ {close.index[split_idx-1]}")
    print(f"   Test:  ~{split_date} ~ {close.index[-1]}")

    train_data = {k: v.iloc[:split_idx] for k, v in full_data.items()}
    test_data = {k: v.iloc[split_idx:] for k, v in full_data.items()}

    # 3. GPT-4o 시드 생성
    seed_alphas = generate_seed_alphas_gpt4o()

    # 4. GP 진화 (train 데이터로)
    (best_alpha, best_ic), top_diverse = genetic_programming(
        seed_alphas,
        train_data,
        generations=50,
        population_size=200
    )

    # 5. Top-5 OOS 검증
    print("\n" + "=" * 80)
    print("🏆 TOP 5 ALPHAS (Train IC + Test IC)")
    print("=" * 80)

    validated_alphas = []
    for i, (alpha, train_ic_with_bonus) in enumerate(top_diverse, 1):
        # 순수 train IC (보너스 제거)
        train_ic = _compute_raw_ic(alpha, train_data)
        # OOS test IC
        test_ic = evaluate_alpha_oos(alpha, test_data)
        # 팩터 분류
        factors = []
        if any(kw in alpha for kw in ['close', 'open_price', 'high', 'low', 'vwap']):
            factors.append('price')
        if 'volume' in alpha:
            factors.append('volume')
        if 'returns' in alpha:
            factors.append('returns')
        if any(kw in alpha for kw in ['high_low_range', 'body', 'upper_shadow', 'lower_shadow']):
            factors.append('candle')
        if any(kw in alpha for kw in ['oi_yoy_rank', 'ni_yoy_rank', 'oi_qoq_rank', 'ni_qoq_rank', 'oi_trend_rank', 'margin_yoy_rank', 'roe_yoy_rank']):
            factors.append('fund_trend')
        factor_str = '+'.join(factors) if factors else 'unknown'

        status = "✅" if test_ic > 0.015 else "⚠️"
        print(f"\n  #{i} {status}")
        print(f"     Train IC: {train_ic:.4f}  |  Test IC: {test_ic:.4f}  [{factor_str}]")
        print(f"     {alpha[:100]}{'...' if len(alpha) > 100 else ''}")

        validated_alphas.append({
            'expr': alpha,
            'train_ic': train_ic,
            'test_ic': test_ic,
            'factors': factor_str,
        })

    # 6. 최종 Best 선정 (test IC가 양수인 것 중 train IC 최고)
    valid_alphas = [a for a in validated_alphas if a['test_ic'] > 0]
    if valid_alphas:
        final_best = max(valid_alphas, key=lambda x: x['train_ic'])
    else:
        final_best = validated_alphas[0] if validated_alphas else {'expr': best_alpha, 'train_ic': best_ic, 'test_ic': -999, 'factors': '?'}

    print("\n" + "=" * 80)
    print("🥇 FINAL BEST (OOS-validated)")
    print("=" * 80)
    print(f"Train IC: {final_best['train_ic']:.4f}")
    print(f"Test IC:  {final_best['test_ic']:.4f}")
    print(f"Factors:  {final_best['factors']}")
    print(f"Expression: {final_best['expr']}")

    # 7. DB 저장 (Top-5 전부)
    print("\n💾 데이터베이스에 저장 중...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        for a in validated_alphas:
            cursor.execute("""
                INSERT INTO alpha_formulas (formula, ic_score, description, created_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (formula) DO UPDATE
                SET ic_score = EXCLUDED.ic_score, updated_at = NOW()
            """, (
                a['expr'],
                float(a['train_ic']),
                f"15d fwd, train IC={a['train_ic']:.4f}, test IC={a['test_ic']:.4f}, factors={a['factors']}, v3-enhanced"
            ))

        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ {len(validated_alphas)}개 알파 저장 완료!")
    except Exception as e:
        print(f"⚠️  DB 저장 실패: {e}")

    print("\n🎉 완료!")

if __name__ == "__main__":
    main()
