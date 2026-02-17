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
from scipy.stats import spearmanr

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
    """200개 종목 데이터 로드 (시가총액 상위)"""
    print("📊 데이터 로드 중... (KOSDAQ 시총 상위 200종목, 2년)")
    
    conn = get_db_connection()
    
    # 시가총액 상위 200개
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
        AND s.ticker >= '400000' ORDER BY s.market_cap DESC
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

    # ── 수급 (Investor Flow) 지표 ──
    try:
        flow_query = f"""
            SELECT s.ticker, sd.date,
                   sd.foreign_net_buy, sd.institution_net_buy,
                   sd.individual_net_buy, sd.foreign_ownership
            FROM supply_demand_data sd
            JOIN stocks s ON sd.stock_id = s.id
            WHERE sd.stock_id IN ({stock_id_list})
            AND sd.date >= CURRENT_DATE - INTERVAL '730 days'
            ORDER BY s.ticker, sd.date
        """
        flow_df = pd.read_sql(flow_query, conn)
        foreign_buy_raw = flow_df.pivot(index='date', columns='ticker', values='foreign_net_buy')
        inst_buy_raw = flow_df.pivot(index='date', columns='ticker', values='institution_net_buy')
        retail_buy_raw = flow_df.pivot(index='date', columns='ticker', values='individual_net_buy')
        foreign_own_raw = flow_df.pivot(index='date', columns='ticker', values='foreign_ownership')
        has_flow = True
    except Exception as e:
        print(f"   ⚠️ 수급 데이터 로드 실패: {e}")
        has_flow = False

    conn.close()

    # ── 파생 기술적 지표 ──
    vwap = (high + low + close) / 3
    high_low_range = (high - low) / close
    body = (close - open_price) / open_price
    upper_shadow = (high - close.clip(lower=open_price)) / close
    lower_shadow = (close.clip(upper=open_price) - low) / close

    # ATR (Average True Range) — 변동성 지표
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3]).groupby(level=0).max()
    true_range = true_range.reindex(close.index)  # 인덱스 정렬
    atr_ratio = true_range / close  # 가격 대비 ATR

    # 거래대금 (amount = close × volume) — 유동성 지표
    amount = close * volume

    # Amihud 비유동성 (|returns| / amount) — 유동성 역수
    amihud = returns.abs() / amount.replace(0, np.nan)
    amihud = amihud.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 갭 (gap = open / prev_close - 1) — 야간 정보
    gap = open_price / close.shift(1) - 1

    # 장중 수익률 (intraday = close / open - 1) — 장중 움직임
    intraday_ret = close / open_price - 1

    # 거래량 회전율 상대비 (volume / 20일 평균)
    vol_ratio = volume / volume.rolling(20, min_periods=5).mean()
    vol_ratio = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1)

    # ── 수급 비율 계산 ──
    if has_flow:
        # price_data와 인덱스/컬럼 정렬
        foreign_buy_raw = foreign_buy_raw.reindex(index=close.index, columns=close.columns)
        inst_buy_raw = inst_buy_raw.reindex(index=close.index, columns=close.columns)
        retail_buy_raw = retail_buy_raw.reindex(index=close.index, columns=close.columns)
        foreign_own_raw = foreign_own_raw.reindex(index=close.index, columns=close.columns)

        # 순매수 비율 = 순매수주수 / 거래량 (clip to [-1, 1])
        safe_volume = volume.replace(0, np.nan)
        foreign_net_ratio = (foreign_buy_raw / safe_volume).clip(-1, 1).fillna(0)
        inst_net_ratio = (inst_buy_raw / safe_volume).clip(-1, 1).fillna(0)
        retail_net_ratio = (retail_buy_raw / safe_volume).clip(-1, 1).fillna(0)
        foreign_ownership_pct = (foreign_own_raw / 100).clip(0, 1).fillna(0)
        print(f"   수급 지표 4개 로드 (foreign/inst/retail net ratio + ownership)")
    else:
        foreign_net_ratio = close * 0.0
        inst_net_ratio = close * 0.0
        retail_net_ratio = close * 0.0
        foreign_ownership_pct = close * 0.0

    tech_vars = [
        'close', 'open_price', 'high', 'low', 'volume', 'returns',
        'vwap', 'high_low_range', 'body', 'upper_shadow', 'lower_shadow',
        'atr_ratio', 'amount', 'amihud', 'gap', 'intraday_ret', 'vol_ratio',
        'foreign_net_ratio', 'inst_net_ratio', 'retail_net_ratio', 'foreign_ownership_pct',
    ]
    print(f"✅ {len(close.columns)}개 종목, {len(close)}일 데이터")
    print(f"   지표 ({len(tech_vars)}개): {', '.join(tech_vars)}")

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
        'gap': gap,
        'intraday_ret': intraday_ret,
        'vol_ratio': vol_ratio,
        'foreign_net_ratio': foreign_net_ratio,
        'inst_net_ratio': inst_net_ratio,
        'retail_net_ratio': retail_net_ratio,
        'foreign_ownership_pct': foreign_ownership_pct,
    }

# ── CogAlpha-inspired: LLM-guided Mutation + Adaptive Feedback ──

def _build_adaptive_feedback(raw_scores, prev_feedback=None):
    """매 세대 top-3 성공 + bottom-3 실패를 CoT 분석하여 누적 피드백 생성.

    CogAlpha 논문 확장: 더 많은 샘플로 풍부한 패턴 학습.
    """
    valid = [(a, ic) for a, ic in raw_scores if ic > -999.0]
    if len(valid) < 6:
        return prev_feedback or ""

    top2 = valid[:3]
    bottom2 = valid[-3:]

    # 팩터 분석 함수
    def _analyze_factors(expr):
        factors = []
        if any(v in expr for v in ['close', 'open_price', 'high', 'low', 'vwap']):
            factors.append('price')
        if any(v in expr for v in ['volume', 'amount', 'vol_ratio']):
            factors.append('volume')
        if 'returns' in expr:
            factors.append('returns')
        if any(v in expr for v in ['high_low_range', 'body', 'upper_shadow', 'lower_shadow', 'atr_ratio']):
            factors.append('volatility')
        if any(v in expr for v in ['amihud', 'gap', 'intraday_ret']):
            factors.append('micro')
        if any(v in expr for v in ['foreign_net_ratio', 'inst_net_ratio', 'retail_net_ratio', 'foreign_ownership_pct']):
            factors.append('flow')
        return '+'.join(factors) if factors else 'unknown'

    def _extract_windows(expr):
        return [int(w) for w in re.findall(r',\s*(\d+)\)', expr)]

    feedback = "### Generation Feedback (Top-3 vs Bottom-3 analysis)\n"
    feedback += "**Top performers this generation:**\n"
    for expr, ic in top2:
        factors = _analyze_factors(expr)
        windows = _extract_windows(expr)
        win_str = f"windows={windows}" if windows else ""
        feedback += f"  - IC={ic:.4f} [{factors}] {win_str}: `{expr[:100]}`\n"

    feedback += "**Bottom performers (patterns to avoid):**\n"
    for expr, ic in bottom2:
        factors = _analyze_factors(expr)
        feedback += f"  - IC={ic:.4f} [{factors}]: `{expr[:100]}`\n"

    # 성공 패턴 vs 실패 패턴 대비
    top_factors = set()
    for expr, _ in top2:
        for v in ['amihud', 'vol_ratio', 'foreign_ownership_pct', 'foreign_net_ratio',
                   'inst_net_ratio', 'vwap', 'lower_shadow', 'close', 'volume']:
            if v in expr:
                top_factors.add(v)

    bottom_factors = set()
    for expr, _ in bottom2:
        for v in ['amihud', 'vol_ratio', 'foreign_ownership_pct', 'foreign_net_ratio',
                   'inst_net_ratio', 'vwap', 'lower_shadow', 'close', 'volume']:
            if v in expr:
                bottom_factors.add(v)

    winning_vars = top_factors - bottom_factors
    if winning_vars:
        feedback += f"**Winning variables**: {', '.join(winning_vars)}\n"

    # 이전 피드백과 병합 (최근 3세대까지만 유지)
    if prev_feedback:
        prev_lines = prev_feedback.strip().split('\n')
        # 이전 피드백에서 가장 중요한 인사이트만 유지
        kept = [l for l in prev_lines if l.startswith('**') or l.startswith('  - IC=')]
        if len(kept) > 12:
            kept = kept[:12]  # 최근 것만 (더 긴 기억)
        feedback += "\n### Previous generation insights:\n" + '\n'.join(kept) + "\n"

    return feedback


def _llm_guided_mutation(top_alphas, adaptive_feedback, num_mutations=15):
    """CogAlpha-inspired LLM-guided mutation (v11 — 7 Diversification Modes).

    랜덤 변이 대신 GPT-4o가 금융 로직을 이해하면서 변이 수행.
    7가지 Diversification Guidance Mode + 모드별 차등 temperature 적용.
    """
    try:
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    except Exception:
        return []

    # Top 알파들을 분석용 텍스트로 변환 (더 넓은 부모 풀)
    parent_block = ""
    for i, (expr, ic) in enumerate(top_alphas[:8], 1):
        parent_block += f"  Parent #{i} (IC={ic:.4f}): `{expr}`\n"

    prompt = f"""### Task: Intelligent Alpha Mutation (7-Mode Diversity Expansion)
You are an expert quant researcher performing **guided mutation** on high-performing alpha expressions.
Unlike random mutation, you understand the financial logic behind each expression and make targeted improvements.
Your goal is to MAXIMIZE DIVERSITY — each mutation should explore a meaningfully different region of alpha space.

### Parent Alphas (these already work well — improve them):
{parent_block}
{adaptive_feedback}
### Mutation Guidance Modes (apply ALL 7 modes, {num_mutations} total mutations):

**Mode 1 - Light** (2 mutations): Fine-tune lookback windows only. If a parent uses ts_mean(x, 5), try 8 or 10.
  Consider: Monthly rebalancing (20d) favors medium-to-long windows (10-150d).

**Mode 2 - Moderate** (2 mutations): Replace operators with similar ones (e.g., ts_mean→ts_ema, ts_median→ts_decayed_linear).
  Keep the same financial logic but change computation method.

**Mode 3 - Creative** (2 mutations): Add a COMPLETELY NEW variable to an existing parent.
  Example: If parent uses amihud/close_MA, add inst_net_ratio or gap or atr_ratio.
  IMPORTANT: Use variables NOT already present in the parent.

**Mode 4 - Divergent** (2 mutations): Combine building blocks from 2+ DISTANT parents into a new alpha.
  Example: Take the numerator structure from Parent #1 and the normalization from Parent #6.
  Choose parents that are structurally MOST different from each other.

**Mode 5 - Concrete** (2 mutations): Create a precise refinement based on the feedback analysis.
  If feedback says "flow variables win", create a new flow-centric combination.

**Mode 6 - Orthogonal** (2 mutations): Use `ops.ts_regression_residual(y, x, window)` to create signals
  that are ORTHOGONAL to existing parents. Extract what existing alphas CANNOT explain.
  Example: `ops.normed_rank(ops.ts_regression_residual(returns, vol_ratio, 20))` = returns unexplained by volume
  Example: `ops.zscore_scale(ops.ts_regression_residual(close, foreign_net_ratio, 30))` = price moves unexplained by foreign flow

**Mode 7 - Conditional** (1 mutation): Use `ops.sign()`, `ops.greater()`, or `ops.relu()` to create
  regime-conditional alphas that behave differently in different market states.
  Example: `ops.normed_rank(ops.cwise_mul(ops.sign(ops.ts_delta(close, 20)), ops.ts_ir(returns, 10)))` = momentum direction × IR
  Example: `ops.normed_rank(ops.cwise_mul(ops.relu(ops.ts_delta_ratio(close, 15)), ops.div(amihud, ops.ts_mean(amihud, 60))))` = only positive momentum × illiquidity

**+ 2 Bonus mutations**: Your most creative ideas combining ANY of the above modes.

### Available Data (21 variables)
close, open_price, high, low, volume, returns, vwap, high_low_range, body, upper_shadow, lower_shadow,
atr_ratio, amount, amihud, gap, intraday_ret, vol_ratio,
foreign_net_ratio, inst_net_ratio, retail_net_ratio, foreign_ownership_pct

### Operator DSL
Time-series (1-var): ts_mean, ts_std, ts_median, ts_ema, ts_linear_reg, ts_delta, ts_delta_ratio,
  ts_zscore_scale, ts_maxmin_scale, ts_rank, ts_ir, ts_skew, ts_min, ts_max, ts_decayed_linear
Time-series (2-var): ts_corr(x, y, window), ts_regression_residual(y, x, window)
Cross-sectional: normed_rank, zscore_scale
Arithmetic: add, minus, cwise_mul, div, neg, abs, log
Conditional: sign(x), relu(x), greater(x, y)

### Rules
- Wrap every alpha with `ops.normed_rank()` or `ops.zscore_scale()`
- Use `ops.` prefix for ALL operators
- Use 2+ lookback windows (multi-timeframe)
- Complexity: 2~5 nesting levels
- Window range: 3~150 (explore extreme short/long windows)
- Try using ts_regression_residual for orthogonal signals
- MAXIMIZE DIVERSITY: each mutation should use DIFFERENT variable combinations

### Output Format
{{"mutations": [
  {{"mode": "Light|Moderate|Creative|Divergent|Concrete|Orthogonal|Conditional",
    "parent_id": 1,
    "reasoning": "Why this mutation improves the parent",
    "expression": "ops.normed_rank(...)"}}
]}}

**CRITICAL**: Return exactly {num_mutations} mutations with valid ops.xxx() expressions.
Modes 3,4,6,7 should be BOLD and explore novel territory — don't play it safe."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert quantitative alpha researcher performing intelligent guided mutation on financial alpha expressions. Prioritize DIVERSITY over incremental improvement. Return your response as a valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=8000,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        mutations = []
        items = []
        if isinstance(data, dict):
            for key in data:
                val = data[key]
                if isinstance(val, list):
                    items = val
                    break

        for item in items:
            expr = None
            if isinstance(item, dict):
                expr = item.get('expression') or item.get('expr')
            elif isinstance(item, str):
                expr = item

            if expr and 'ops.' in expr and expr.count('(') == expr.count(')'):
                mutations.append(expr)

        # 마크다운 폴백
        if not mutations:
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('ops.') and '(' in line:
                    expr = line.strip('",` ')
                    if expr.count('(') == expr.count(')'):
                        mutations.append(expr)

        return mutations[:num_mutations]

    except Exception as e:
        print(f"      ⚠️ LLM mutation 실패: {e}")
        return []


def _llm_guided_crossover(top_alphas, adaptive_feedback, num_children=8):
    """CogAlpha-inspired LLM-guided crossover (v11 — 더 넓은 부모 풀 + 원거리 교차).

    두 부모 알파의 금융 로직을 이해하고 의미있는 교차를 수행.
    더 먼 부모 간 교차로 다양성 극대화.
    """
    try:
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    except Exception:
        return []

    parent_block = ""
    for i, (expr, ic) in enumerate(top_alphas[:10], 1):
        parent_block += f"  Parent #{i} (IC={ic:.4f}): `{expr}`\n"

    prompt = f"""### Task: Intelligent Alpha Crossover
Combine building blocks from multiple parent alphas to create novel offspring.

### Parent Alphas:
{parent_block}
{adaptive_feedback}
### Crossover Strategy
For each offspring:
1. Select 2 parents — PREFER DISTANT parents (e.g., #1 × #8, not #1 × #2) for maximum novelty
2. Identify the "winning component" from each (e.g., the numerator logic, the normalization method, the variable selection)
3. Combine them into a new alpha that inherits strengths from both parents
4. Explain WHY this combination should work
5. At least 2 crossovers should use ts_regression_residual or conditional operators (sign/greater/relu)

### Rules
- Wrap with `ops.normed_rank()` or `ops.zscore_scale()`
- Use `ops.` prefix for ALL operators
- 2+ lookback windows, 2~5 nesting levels
- Window range: 3~150
- MAXIMIZE the number of UNIQUE variable combinations across offspring

### Output Format
{{"crossovers": [
  {{"parent1_id": 1, "parent2_id": 8,
    "reasoning": "Combines X's liquidity signal with Y's flow signal",
    "expression": "ops.normed_rank(...)"}}
]}}

Generate exactly {num_children} crossover offspring. Each MUST use a different variable combination."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert quantitative alpha researcher performing intelligent crossover on financial alpha expressions. Prioritize DISTANT parent combinations for maximum novelty. Return your response as a valid JSON object."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=8000,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        children = []
        items = []
        if isinstance(data, dict):
            for key in data:
                val = data[key]
                if isinstance(val, list):
                    items = val
                    break

        for item in items:
            expr = None
            if isinstance(item, dict):
                expr = item.get('expression') or item.get('expr')
            elif isinstance(item, str):
                expr = item

            if expr and 'ops.' in expr and expr.count('(') == expr.count(')'):
                children.append(expr)

        return children[:num_children]

    except Exception as e:
        print(f"      ⚠️ LLM crossover 실패: {e}")
        return []


def _load_previous_results():
    """DB에서 이전 GP 결과 (best/worst) 로드"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT formula, ic_score, description
            FROM alpha_formulas
            WHERE description LIKE '%15d fwd%'
            ORDER BY ic_score DESC
            LIMIT 20
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        if rows:
            best = [(r[0], r[1], r[2]) for r in rows[:5]]
            worst = [(r[0], r[1], r[2]) for r in rows[-3:] if r[1] is not None]
            return best, worst
    except Exception:
        pass
    return [], []


def generate_seed_alphas_gpt4o(num_seeds=30):
    """2단계 시드 생성: (1) 가설 10개 생성 → (2) 가설 기반 알파 생성 + 이전 결과 피드백"""
    print(f"\n🤖 GPT-4o 2단계 시드 생성 (가설→알파, {num_seeds}개)...")

    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    system_prompt = QuantDeveloper.SYSTEM_PROMPT

    # ── 이전 GP 결과 로드 ──
    prev_best, prev_worst = _load_previous_results()
    feedback_block = ""
    if prev_best:
        feedback_block += "\n### Previous GP Results (LEVERAGE these insights)\n"
        feedback_block += "**Top performers (replicate and extend these patterns):**\n"
        for expr, ic, desc in prev_best[:5]:
            test_ic_str = ""
            if desc:
                import re as _re
                m = _re.search(r'test IC=([0-9.-]+)', desc)
                if m:
                    test_ic_str = f", test IC={m.group(1)}"
            feedback_block += f"  - IC={ic:.4f}{test_ic_str}: `{expr[:120]}`\n"
        feedback_block += "\n**Patterns to AVOID (these failed OOS validation):**\n"
        for expr, ic, desc in prev_worst:
            feedback_block += f"  - IC={ic:.4f}: `{expr[:100]}`\n"
        feedback_block += "\n**Key lessons**: Focus on patterns similar to top performers. "
        feedback_block += "Explore NEW combinations of successful building blocks (MA slope, amihud, lower_shadow, vwap).\n"
        print(f"   이전 결과 피드백: best {len(prev_best)}개, worst {len(prev_worst)}개")

    # ── 1단계: 가설 생성 (temperature=0.5, 다양한 가설) ──
    print("   [1/2] 가설 생성 중...")
    hypothesis_prompt = f"""You are a quantitative finance researcher. Generate 10 structured trading hypotheses
for a **20-day (1-month) holding period** strategy in the Korean stock market (KRX).

Each hypothesis must follow this EXACT JSON format:
{{"hypotheses": [
  {{
    "hypothesis": "Complete hypothesis statement",
    "reason": "Why this captures alpha — economic/behavioral explanation",
    "observation": "Key market pattern or anomaly being exploited",
    "knowledge": "If [condition], then [expected outcome] over 20 trading days (~1 month)"
  }}
]}}

Generate EXACTLY 10 hypotheses, one per theme:
1. **Momentum + Volume confirmation**: Price trend confirmed by trading activity pattern
2. **Volatility regime + Mean-reversion**: Candle body/shadow patterns predicting reversals
3. **Liquidity premium**: Amihud illiquidity ratio combined with price structure
4. **Multi-timeframe divergence**: Short-term vs long-term momentum disagreement
5. **Cross-variable decorrelation**: Using ts_regression_residual to extract orthogonal signals
   (e.g., returns not explained by volume, price moves not explained by flow)
6. **Microstructure signals**: Gap, intraday returns, candle shape as information signals
7. **Institutional flow momentum**: 기관/외국인 순매수 추세가 가격에 선행하는 패턴
   (foreign_net_ratio, inst_net_ratio의 누적 흐름이 향후 수익률을 예측)
8. **Volatility compression breakout**: ATR 수축 후 확장 → 추세 시작 신호
   (atr_ratio가 낮아졌다가 높아지는 종목이 20일 후 수익률 높음)
9. **Turnover anomaly**: 거래대금(amount) 기반 유동성 프리미엄
   (거래대금 변화율과 가격 모멘텀의 비선형 관계)
10. **Regime-conditional alpha**: 변동성 레짐에 따라 다른 신호 적용
    (sign/greater 조건 연산자로 고변동성 vs 저변동성 구간 분기)
{feedback_block}
Available data: close, open_price, high, low, volume, returns, vwap, high_low_range, body,
upper_shadow, lower_shadow, atr_ratio, amount, amihud, gap, intraday_ret, vol_ratio,
foreign_net_ratio, inst_net_ratio, retail_net_ratio, foreign_ownership_pct"""

    try:
        hyp_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert quantitative researcher specializing in KRX alpha factor hypothesis generation."},
                {"role": "user", "content": hypothesis_prompt}
            ],
            temperature=0.5,
            max_tokens=6000,
            response_format={"type": "json_object"}
        )
        hyp_content = hyp_response.choices[0].message.content
        hyp_data = json.loads(hyp_content)
        hypotheses = hyp_data.get('hypotheses', [hyp_data]) if isinstance(hyp_data, dict) else [hyp_data]
        if not isinstance(hypotheses, list):
            hypotheses = [hypotheses]
        print(f"   ✅ {len(hypotheses)}개 가설 생성")
        for i, h in enumerate(hypotheses[:4], 1):
            hyp_text = h.get('hypothesis', h.get('reason', '?'))
            print(f"      {i}. {hyp_text[:80]}...")
    except Exception as e:
        print(f"   ⚠️  가설 생성 실패: {e}, 기본 가설 사용")
        hypotheses = [
            {"hypothesis": "Momentum confirmed by volume surge predicts 20-day continuation",
             "knowledge": "If price rises with above-average volume, then trend continues for 20 days"},
            {"hypothesis": "Low volatility stocks with buying pressure show mean-reversion alpha",
             "knowledge": "If ATR is low and lower_shadow is large, then price rebounds within 20 trading days"},
            {"hypothesis": "Illiquid stocks with trend signals offer liquidity premium",
             "knowledge": "If amihud is high and MA slope is positive, then excess returns over 20 trading days"},
            {"hypothesis": "Short-term vs long-term momentum divergence signals regime change",
             "knowledge": "If 5-day momentum diverges from 60-day trend, then reversal within 20 trading days"},
            {"hypothesis": "Returns orthogonal to volume activity predict future returns",
             "knowledge": "If stock returns are high but not explained by volume, then alpha persists for 20 days"},
            {"hypothesis": "Overnight gap + intraday reversal patterns predict next month",
             "knowledge": "If gap and intraday return diverge, then price corrects within 20 trading days"},
            {"hypothesis": "Institutional flow momentum leads price by 1-4 weeks",
             "knowledge": "If foreign_net_ratio or inst_net_ratio accumulates over 10-20 days, then price follows within 20 trading days"},
            {"hypothesis": "Volatility compression followed by expansion signals breakout",
             "knowledge": "If atr_ratio contracts then expands, then directional move occurs within 20 trading days"},
            {"hypothesis": "Trading amount anomalies predict liquidity-driven returns",
             "knowledge": "If amount surges relative to history while price is flat, then price catches up within 20 days"},
            {"hypothesis": "Regime-conditional signals: different alphas work in different volatility regimes",
             "knowledge": "If market volatility is low, momentum works; if high, mean-reversion works over 20 days"},
        ]

    # ── 2단계: 가설 기반 알파 생성 (temperature=0.7, 다양성 극대화) ──
    print("   [2/2] 가설 기반 알파 생성 중...")
    hypotheses_text = ""
    for i, h in enumerate(hypotheses[:10], 1):
        hyp = h.get('hypothesis', '') or h.get('reason', '')
        knowledge = h.get('knowledge', '') or h.get('concise_knowledge', '')
        hypotheses_text += f"\n**Hypothesis {i}**: {hyp}\n"
        if knowledge:
            hypotheses_text += f"  Knowledge: {knowledge}\n"

    alphas_per_hyp = num_seeds // 10
    remaining = num_seeds - alphas_per_hyp * 10

    prompt = f"""### Task
Generate {num_seeds} diverse alpha expressions for **20-day (1-month) forward returns** in KRX.
Each alpha MUST be grounded in one of the hypotheses below.
{hypotheses_text}
### Alpha Generation Rules
- Generate {alphas_per_hyp} alphas per hypothesis ({alphas_per_hyp}×10 = {alphas_per_hyp*10}), plus {remaining} bonus composite alphas.
- Each alpha MUST reference which hypothesis (1-6) it implements.
{feedback_block}

### ⚠️ BANNED PATTERNS (these are overfit — DO NOT generate anything similar)
- `ops.div(ops.ts_mean(foreign_ownership_pct, N), ops.ts_decayed_linear(vol_ratio, N))` — overfit on quarterly data
- Any alpha where `foreign_ownership_pct` is the PRIMARY driver (it's forward-filled quarterly, not daily)
- `amihud / ts_mean(close, N)` as the ONLY signal — too simplistic, already discovered
- Any simple combination of ONLY foreign_ownership_pct + vol_ratio + amihud + close

### ✅ DIVERSITY REQUIREMENTS (MUST follow)
- At least 3 alphas MUST use `ts_corr(x, y, window)` for cross-variable correlation
- At least 2 alphas MUST use `ts_regression_residual(y, x, window)` for orthogonal signals
- At least 2 alphas MUST use conditional operators: `sign()`, `relu()`, or `greater()`
- At least 4 alphas MUST use NONE of the flow variables (pure technical: close/volume/vwap/body/shadow/gap/amihud etc.)
- Each alpha MUST use a DIFFERENT main variable combination from the others

### Available Data (21 variables: 17 technical + 4 supply/demand)
**Price**: close, open_price, high, low, vwap
**Volume**: volume, amount, vol_ratio
**Returns**: returns
**Volatility/Shape**: high_low_range, body, upper_shadow, lower_shadow, atr_ratio
**Microstructure**: amihud (illiquidity), gap (overnight), intraday_ret (intraday)
**Investor Flow**: foreign_net_ratio, inst_net_ratio, retail_net_ratio, foreign_ownership_pct

### Operator DSL
Time-series (1-var): ts_mean, ts_std, ts_median, ts_ema, ts_linear_reg, ts_delta, ts_delta_ratio,
  ts_zscore_scale, ts_maxmin_scale, ts_rank, ts_ir, ts_skew, ts_min, ts_max, ts_decayed_linear
Time-series (2-var):
  ts_corr(x, y, window) — rolling correlation between two variables
  ts_regression_residual(y, x, window) — rolling OLS residual (y unexplained by x)
Cross-sectional: normed_rank, zscore_scale
Arithmetic: add, minus, cwise_mul, div, neg, abs, log
Conditional: sign(x) — returns -1/0/+1, relu(x) — max(0, x), greater(x, y) — 1 if x>y else 0

### HIGH-VALUE STRUCTURAL TEMPLATES (use these as building blocks)
1. **Cross-correlation**: `ops.ts_corr(returns, vol_ratio, 20)` — price-volume divergence
2. **Orthogonal signal**: `ops.ts_regression_residual(returns, vol_ratio, 30)` — returns NOT explained by volume
3. **Directional filter**: `ops.cwise_mul(ops.sign(ops.ts_delta(close, 20)), other_signal)` — only long in uptrend
4. **Asymmetric capture**: `ops.relu(ops.ts_delta_ratio(vwap, 15))` — upside momentum only
5. **Conditional regime**: `ops.cwise_mul(ops.greater(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20)), signal)` — volume breakout filter
6. **Divergence signal**: `ops.minus(ops.ts_rank(close, 20), ops.ts_rank(volume, 20))` — price-volume rank divergence
7. **Residual momentum**: `ops.ts_linear_reg(ops.ts_regression_residual(returns, vol_ratio, 30), 10)` — trend in residual returns

### Quality Rules
- Wrap every alpha with `ops.normed_rank()` or `ops.zscore_scale()`
- Use 2+ lookback windows per alpha (multi-timeframe)
- Use `ops.div()` for division (safe)
- Prefer `ops.cwise_mul()` for multiplicative signals
- Complexity: 2~4 nesting levels
- MAXIMIZE diversity: each alpha should explore a DIFFERENT combination of variables and operators

### Output Format
{{"alphas": [
  {{"alpha_name": "...", "hypothesis_id": 1, "category": "...",
    "rationale": "...", "expression": "ops.normed_rank(...)"}}
]}}

**CRITICAL**: Return a valid JSON object with {num_seeds} alphas total. Each MUST have "expression" with valid ops.xxx() code."""

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

    # ── 검증된 OOS-validated 시드 (핵심만, 다양성 우선) ──
    proven_seeds = [
        # [price+micro] vwap MA slope + amihud (IC 0.060)
        "ops.normed_rank(ops.add(ops.normed_rank(ops.ts_delta_ratio(ops.ts_mean(vwap, 120), 10)), ops.normed_rank(ops.div(amihud, ops.ts_median(close, 20)))))",
        # [price] lower_shadow + close MA (IC 0.127)
        "ops.normed_rank(ops.add(ops.ts_mean(lower_shadow, 15), ops.div(close, ops.ts_mean(close, 120))))",
        # [price] 120일 이격도
        "ops.normed_rank(ops.div(close, ops.ts_mean(close, 120)))",
        # [price+volume] 모멘텀 × 거래량 안정성
        "ops.normed_rank(ops.cwise_mul(ops.cwise_mul(ops.ts_delta_ratio(close, 25), ops.div(ops.ts_median(volume, 10), ops.ts_std(volume, 15))), ops.ts_maxmin_scale(close, 28)))",
        # [micro] amihud / close MA
        "ops.normed_rank(ops.div(amihud, ops.ts_median(close, 20)))",
        # [price+volume] 다중 MA × 거래량 + MA기울기
        "ops.normed_rank(ops.add(ops.cwise_mul(ops.div(ops.ts_mean(close, 5), ops.ts_mean(close, 20)), ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20))), ops.ts_delta_ratio(ops.ts_mean(close, 20), 10)))",
        # ── 새 패턴: ts_regression_residual ──
        # [orthogonal] returns에서 vol_ratio 영향 제거 → 순수 가격 시그널
        "ops.normed_rank(ops.ts_regression_residual(returns, vol_ratio, 30))",
        # [orthogonal] close 움직임에서 amihud 영향 제거 → 유동성 프리미엄 제외 모멘텀
        "ops.normed_rank(ops.ts_linear_reg(ops.ts_regression_residual(close, amihud, 20), 10))",
        # [orthogonal + micro] 수익률 잔차 × gap 반전
        "ops.normed_rank(ops.cwise_mul(ops.ts_regression_residual(returns, vol_ratio, 20), ops.neg(ops.ts_mean(gap, 10))))",
        # ── 새 패턴: rank divergence ──
        # [divergence] close rank vs volume rank 괴리 (가격 과열/과냉 탐지)
        "ops.normed_rank(ops.minus(ops.ts_rank(close, 20), ops.ts_rank(volume, 20)))",
        # [divergence] vwap rank vs amihud rank 괴리
        "ops.normed_rank(ops.minus(ops.ts_rank(vwap, 30), ops.ts_rank(amihud, 30)))",
        # ── 기존 검증 패턴: ts_corr ──
        # [corr] 가격-거래량 상관 역전
        "ops.normed_rank(ops.neg(ops.ts_corr(returns, vol_ratio, 20)))",
        # [conditional] 상승 추세에서만 비유동성 프리미엄
        "ops.normed_rank(ops.cwise_mul(ops.sign(ops.ts_delta(close, 20)), ops.div(amihud, ops.ts_mean(close, 60))))",
        # [conditional+volume] 거래량 급증 구간에서 body 시그널
        "ops.normed_rank(ops.cwise_mul(ops.greater(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20)), ops.ts_mean(body, 10)))",
    ]

    # 항상 검증된 시드를 먼저 포함 + GPT-4o 시드 추가
    proven_not_in = [s for s in proven_seeds if s not in alphas]
    gpt_alphas = [a for a in alphas if a not in proven_seeds]
    alphas = proven_not_in + gpt_alphas  # 검증된 시드 우선
    print(f"   검증된 시드: {len(proven_not_in)}개 + GPT-4o 시드: {len(gpt_alphas)}개")

    # 부족하면 추가 폴백
    if len(alphas) < 10:
        print(f"⚠️  {len(alphas)}개만 파싱됨, 추가 폴백")
        extra_fallback = [
            "ops.normed_rank(ops.cwise_mul(ops.div(close, ops.ts_mean(close, 20)), ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20))))",
            "ops.normed_rank(ops.cwise_mul(ops.ts_delta_ratio(close, 15), ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20))))",
            "ops.normed_rank(ops.neg(ops.ts_corr(ops.ts_delta(close, 5), ops.ts_delta(volume, 5), 20)))",
            "ops.normed_rank(ops.cwise_mul(ops.ts_delta_ratio(close, 20), ops.neg(ops.ts_mean(high_low_range, 15))))",
            "ops.normed_rank(ops.ts_linear_reg(close, 20))",
            "ops.normed_rank(ops.ts_maxmin_scale(close, 60))",
            "ops.normed_rank(ops.div(close, ops.ts_mean(vwap, 20)))",
            "ops.normed_rank(ops.ts_mean(body, 10))",
            "ops.normed_rank(ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 60)))",
            "ops.normed_rank(ops.ts_zscore_scale(close, 20))",
            "ops.normed_rank(ops.ts_delta_ratio(ops.ts_mean(close, 120), 20))",
        ]
        alphas = alphas + [f for f in extra_fallback if f not in alphas]

    print(f"✅ {len(alphas)}개 초기 알파 생성 (검증된 시드 포함)")
    for i, a in enumerate(alphas[:5], 1):
        print(f"   {i}. {a[:80]}...")

    return alphas[:num_seeds]

# 전역 데이터
_global_data = None
_global_train_start_date = None
_global_train_end_date = None

def set_global_data(data, train_start_date=None, train_end_date=None):
    global _global_data, _global_train_start_date, _global_train_end_date
    _global_data = data
    _global_train_start_date = train_start_date
    _global_train_end_date = train_end_date

def _compute_ic_series(alpha_expr, data, date_start=None, date_end=None):
    """알파의 일별 IC 리스트 + 턴오버 계산.

    Returns:
        (ic_list, turnover) — turnover = 1 - rank_autocorrelation (낮을수록 안정)
    """
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
    atr_ratio = data.get('atr_ratio', high_low_range)
    amount = data.get('amount', close * volume)
    amihud = data.get('amihud', returns.abs() / amount.replace(0, np.nan))
    gap = data.get('gap', open_price / close.shift(1) - 1)
    intraday_ret = data.get('intraday_ret', close / open_price - 1)
    vol_ratio = data.get('vol_ratio', volume / volume.rolling(20, min_periods=5).mean())
    foreign_net_ratio = data.get('foreign_net_ratio', close * 0.0)
    inst_net_ratio = data.get('inst_net_ratio', close * 0.0)
    retail_net_ratio = data.get('retail_net_ratio', close * 0.0)
    foreign_ownership_pct = data.get('foreign_ownership_pct', close * 0.0)

    forward_return = close.shift(-20) / close - 1  # 20영업일 (~1달) 선행수익률
    alpha_values = eval(alpha_expr)

    if not isinstance(alpha_values, pd.DataFrame):
        return [], 1.0

    alpha_values = alpha_values.replace([np.inf, -np.inf], np.nan)

    n_stocks = len(close.columns)
    coverage_threshold = n_stocks * 0.5
    ic_list = []
    low_coverage_days = 0
    total_days = 0

    # 턴오버: 20일 간격 rank autocorrelation
    rank_autocorrs = []
    prev_ranks = None
    day_counter = 0

    for date in alpha_values.index[:-20]:
        if date_start is not None and date < date_start:
            continue
        if date_end is not None and date > date_end:
            continue

        alpha_cs = alpha_values.loc[date]
        returns_cs = forward_return.loc[date]
        valid = alpha_cs.notna() & returns_cs.notna()
        total_days += 1

        if valid.sum() < coverage_threshold:
            low_coverage_days += 1

        if valid.sum() > 30:
            # 극단 수익률 필터 (3-sigma, 기업 이벤트/착오 제거)
            ret_mean = returns_cs[valid].mean()
            ret_std = returns_cs[valid].std()
            if ret_std > 0:
                extreme = (returns_cs - ret_mean).abs() > 3 * ret_std
                valid = valid & ~extreme
            if valid.sum() > 30:
                rho, _ = spearmanr(alpha_cs[valid].values, returns_cs[valid].values)
                if not np.isnan(rho):
                    ic_list.append(rho)

        # 20일 간격 rank autocorrelation (턴오버 proxy)
        current_ranks = alpha_cs.rank()
        day_counter += 1
        if day_counter % 20 == 0 and prev_ranks is not None:
            joint_valid = current_ranks.notna() & prev_ranks.notna()
            if joint_valid.sum() > 30:
                rank_corr, _ = spearmanr(
                    current_ranks[joint_valid].values,
                    prev_ranks[joint_valid].values
                )
                if not np.isnan(rank_corr):
                    rank_autocorrs.append(rank_corr)
            prev_ranks = current_ranks
        elif day_counter % 20 == 0:
            prev_ranks = current_ranks

    # 커버리지 페널티
    if total_days > 0:
        valid_day_ratio = 1.0 - (low_coverage_days / total_days)
        if valid_day_ratio < 0.8:
            ic_list = [ic * valid_day_ratio for ic in ic_list]

    # 턴오버 = 1 - rank_autocorrelation (0=안정, 1=완전 교체)
    avg_rank_autocorr = float(np.mean(rank_autocorrs)) if rank_autocorrs else 0.5
    turnover = 1.0 - avg_rank_autocorr

    return ic_list, turnover


def _compute_raw_ic(alpha_expr, data, date_start=None, date_end=None):
    """하위 호환: mean IC만 반환."""
    try:
        ic_list, _ = _compute_ic_series(alpha_expr, data, date_start, date_end)
        if len(ic_list) < 10:
            return -999.0
        return float(np.mean(ic_list))
    except Exception:
        return -999.0

def _multi_factor_bonus(alpha_expr):
    """다중 팩터 구조 보너스 — 10x 강화, 단일 카테고리 페널티"""
    bonus = 0.0

    # 카테고리 분류 — 보너스 1/3 축소 (순수 IC 최적화 집중)
    categories_used = 0
    if any(v in alpha_expr for v in ['close', 'open_price', 'high', 'low', 'vwap']):
        categories_used += 1
    if any(v in alpha_expr for v in ['volume', 'amount', 'vol_ratio']):
        categories_used += 1
        bonus += 0.003
    if 'returns' in alpha_expr:
        categories_used += 1
    if any(v in alpha_expr for v in ['high_low_range', 'atr_ratio']):
        categories_used += 1
    if any(v in alpha_expr for v in ['body', 'upper_shadow', 'lower_shadow']):
        categories_used += 1
    if any(v in alpha_expr for v in ['amihud', 'gap', 'intraday_ret']):
        categories_used += 1
        bonus += 0.003
    if any(v in alpha_expr for v in ['foreign_net_ratio', 'inst_net_ratio', 'retail_net_ratio', 'foreign_ownership_pct']):
        categories_used += 1
        bonus += 0.003

    # 단일 카테고리 페널티
    if categories_used <= 1:
        bonus -= 0.010
    elif categories_used >= 3:
        bonus += 0.005
    elif categories_used >= 2:
        bonus += 0.002

    # MA 구조 보너스
    has_ma = bool(re.search(r'ts_mean\([^)]*close[^)]*,\s*\d+\)', alpha_expr))
    if has_ma:
        bonus += 0.002

    # 새 연산자 사용 보너스 (탐색 장려)
    if 'ts_regression_residual' in alpha_expr:
        bonus += 0.003
    if 'ts_corr' in alpha_expr:
        bonus += 0.002

    # 다중 타임프레임 보너스
    windows = [int(w) for w in re.findall(r',\s*(\d+)\)', alpha_expr)]
    if len(windows) >= 2 and max(windows) >= min(windows) * 2:
        bonus += 0.002
    if windows and max(windows) >= 60:
        bonus += 0.001

    # 복잡도 페널티
    depth = alpha_expr.count('(')
    if depth < 3:
        bonus -= 0.003
    if depth > 8:
        bonus -= 0.003 * (depth - 8)

    return bonus

def evaluate_alpha_worker(alpha_expr):
    """Fitness = 0.85 × mean_IC + 0.15 × IC_IR × 0.05 - turnover_penalty + bonus.

    순수 IC 최적화 집중 (보너스 축소, IC 가중 강화).
    """
    global _global_data, _global_train_start_date, _global_train_end_date
    data = _global_data

    try:
        ic_list, turnover = _compute_ic_series(
            alpha_expr, data,
            date_start=_global_train_start_date,
            date_end=_global_train_end_date
        )
        if len(ic_list) < 10:
            return (alpha_expr, -999.0)

        mean_ic = float(np.mean(ic_list))
        std_ic = float(np.std(ic_list))
        ic_ir = mean_ic / std_ic if std_ic > 0.001 else 0.0

        # Fitness: 85% mean IC + 15% IC_IR (IC 최적화 집중)
        fitness = 0.85 * mean_ic + 0.15 * ic_ir * 0.05

        # 턴오버 페널티: 30% 초과 시 비례 감점
        turnover_penalty = max(0, turnover - 0.3) * 0.02

        bonus = _multi_factor_bonus(alpha_expr)
        return (alpha_expr, fitness - turnover_penalty + bonus)
    except Exception:
        return (alpha_expr, -999.0)

def evaluate_alpha_oos(alpha_expr, data, date_start=None):
    """Out-of-sample IC 계산 (보너스 없이 순수 IC, 날짜 범위 지원)"""
    try:
        return _compute_raw_ic(alpha_expr, data, date_start=date_start)
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
    ['ts_corr', 'ts_regression_residual'],  # 2-var 연산자 교환
]

OPERAND_POOL = ['close', 'open_price', 'high', 'low', 'volume', 'returns', 'vwap', 'high_low_range', 'body',
                'upper_shadow', 'lower_shadow', 'atr_ratio', 'amount', 'amihud', 'gap', 'intraday_ret', 'vol_ratio',
                'foreign_net_ratio', 'inst_net_ratio', 'retail_net_ratio', 'foreign_ownership_pct']

def mutate_alpha(alpha_expr):
    """알파 변이 — 4가지 타입: 윈도우(20%), 연산자(20%), 피연산자(30%), 구조(30%)
    피연산자와 구조 변이 비중을 높여 더 다양한 변수/구조 조합 탐색."""
    try:
        mutation_type = random.choices(
            ['window', 'operator', 'operand', 'structural'],
            weights=[0.20, 0.20, 0.30, 0.30]
        )[0]

        if mutation_type == 'window':
            return _mutate_window(alpha_expr)
        elif mutation_type == 'operator':
            return _mutate_operator(alpha_expr)
        elif mutation_type == 'operand':
            return _mutate_operand(alpha_expr)
        else:
            return _mutate_structural(alpha_expr)
    except Exception:
        return None

def _mutate_window(alpha_expr):
    """윈도우 파라미터 변경 (범위 3~150, 극단적 단기/장기 탐색 포함)"""
    matches = list(re.finditer(r'(ts_\w+|shift)\([^,]+,\s*(\d+)\)', alpha_expr))
    if not matches:
        return None
    match = random.choice(matches)
    old_window = int(match.group(2))
    # 현재 윈도우 크기에 따라 변이 폭 조절 (비례적 변이)
    if old_window <= 20:
        deltas = [-7, -5, -3, -2, 2, 3, 5, 7, 10, 15, 20]
    elif old_window <= 60:
        deltas = [-20, -15, -10, -7, -5, 5, 7, 10, 15, 20, 30, 40]
    else:
        deltas = [-40, -30, -20, -10, 10, 20, 30, 40]
    new_window = max(3, min(150, old_window + random.choice(deltas)))
    if new_window == old_window:
        new_window = max(3, min(150, old_window + random.choice([-25, 25])))
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


def _mutate_structural(alpha_expr):
    """구조적 변이 — 6가지 타입으로 탐색 공간 대폭 확장.

    ts_corr, ts_regression_residual, sign, relu, greater, rank_divergence 등
    새로운 연산자를 주입하여 완전히 다른 알파 구조 생성.
    """
    structural_type = random.choices(
        ['ts_corr_new', 'ts_corr_add', 'sign_filter', 'relu_clip',
         'regression_residual', 'rank_divergence'],
        weights=[0.15, 0.15, 0.15, 0.15, 0.25, 0.15]
    )[0]

    def _unwrap(expr):
        """normed_rank/zscore_scale 래퍼 제거"""
        for wrapper in ['ops.normed_rank(', 'ops.zscore_scale(']:
            if expr.startswith(wrapper) and expr.endswith(')'):
                return expr[len(wrapper):-1]
        return expr

    try:
        if structural_type == 'ts_corr_new':
            var1 = random.choice(OPERAND_POOL)
            var2 = random.choice([v for v in OPERAND_POOL if v != var1])
            window = random.choice([10, 15, 20, 30, 60])
            return f"ops.normed_rank(ops.ts_corr({var1}, {var2}, {window}))"

        elif structural_type == 'ts_corr_add':
            inner = _unwrap(alpha_expr)
            var1 = random.choice(OPERAND_POOL)
            var2 = random.choice([v for v in OPERAND_POOL if v != var1])
            window = random.choice([10, 15, 20, 30, 60])
            return f"ops.normed_rank(ops.add({inner}, ops.ts_corr({var1}, {var2}, {window})))"

        elif structural_type == 'sign_filter':
            inner = _unwrap(alpha_expr)
            var = random.choice(OPERAND_POOL)
            window = random.choice([5, 10, 20, 40])
            return f"ops.normed_rank(ops.cwise_mul(ops.sign(ops.ts_delta({var}, {window})), {inner}))"

        elif structural_type == 'relu_clip':
            inner = _unwrap(alpha_expr)
            return f"ops.normed_rank(ops.relu({inner}))"

        elif structural_type == 'regression_residual':
            # ts_regression_residual: 한 변수에서 다른 변수의 영향 제거
            y_var = random.choice(OPERAND_POOL)
            x_var = random.choice([v for v in OPERAND_POOL if v != y_var])
            window = random.choice([15, 20, 30, 60])
            # 50% 확률로 잔차를 그대로 사용 vs 잔차의 추세 사용
            if random.random() < 0.5:
                return f"ops.normed_rank(ops.ts_regression_residual({y_var}, {x_var}, {window}))"
            else:
                trend_window = random.choice([5, 10, 15, 20])
                return f"ops.normed_rank(ops.ts_linear_reg(ops.ts_regression_residual({y_var}, {x_var}, {window}), {trend_window}))"

        elif structural_type == 'rank_divergence':
            # 두 변수의 시계열 순위 차이 → 괴리 포착
            var1 = random.choice(OPERAND_POOL)
            var2 = random.choice([v for v in OPERAND_POOL if v != var1])
            window = random.choice([10, 20, 30, 60])
            return f"ops.normed_rank(ops.minus(ops.ts_rank({var1}, {window}), ops.ts_rank({var2}, {window})))"

    except Exception:
        pass
    return None


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

def _get_variable_signature(alpha_expr):
    """알파에 사용된 변수 조합 (순서 무관) — 변수 수준 다양성 비교용.

    같은 변수 조합 {foreign_ownership_pct, vol_ratio, amihud, close}의
    window/operator 변형은 동일 시그니처로 취급.
    """
    vars_used = set()
    for var in OPERAND_POOL:
        if var in alpha_expr:
            vars_used.add(var)
    return frozenset(vars_used)

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


def _fitness_sharing(fitness_scores, sharing_radius=1.0):
    """적합도 공유 — 구조 + 변수 조합 2단계 니쉬 페널티로 다양성 보존.

    Level 1: 같은 구조(operator+variable, window 제외) → 강한 페널티
    Level 2: 같은 변수 조합(operator 무관) → 추가 페널티
    → foreign_ownership/vol_ratio/amihud/close 조합이 독점하는 것을 방지
    """
    # Level 1: 구조 니쉬 (기존)
    structures = {}
    for alpha, ic in fitness_scores:
        struct = _get_alpha_structure(alpha)
        if struct not in structures:
            structures[struct] = []
        structures[struct].append((alpha, ic))

    # Level 2: 변수 조합 니쉬
    var_sigs = {}
    for alpha, ic in fitness_scores:
        sig = _get_variable_signature(alpha)
        if sig not in var_sigs:
            var_sigs[sig] = 0
        var_sigs[sig] += 1

    shared = []
    for struct, members in structures.items():
        struct_niche = len(members)
        for alpha, ic in members:
            # 구조 니쉬 페널티
            penalty = 1.0 + sharing_radius * (struct_niche - 1)
            # 변수 조합 니쉬 페널티 (같은 변수 조합이 8개 이상이면 추가 감점)
            var_sig = _get_variable_signature(alpha)
            var_count = var_sigs.get(var_sig, 1)
            if var_count > 8:
                penalty += 0.3 * (var_count - 8) / 8  # 점진적 추가 페널티 (조기 발동)
            shared_ic = ic / penalty
            shared.append((alpha, shared_ic))

    return sorted(shared, key=lambda x: x[1], reverse=True)


def genetic_programming(seed_alphas, data, train_start_date=None, train_end_date=None,
                        generations=70, population_size=300):
    """병렬 GP v11 — CogAlpha-inspired: LLM-guided mutation + adaptive feedback + 다양성 극대화"""

    close_idx = data['close'].index
    print(f"\n🧬 병렬 GP 시작 (v11 — CogAlpha: LLM-guided + diversity expansion)")
    if train_end_date is not None:
        print(f"   Train IC range: {train_start_date or close_idx[0]} ~ {train_end_date}")
    else:
        print(f"   Train IC range: full data")
    print(f"   Seed: {len(seed_alphas)}개, 세대: {generations}, 개체수: {population_size}, 워커: 8")
    print(f"   LLM mutation: 매 3세대, LLM crossover: 매 5세대")

    population = seed_alphas[:population_size]
    while len(population) < population_size:
        parent = random.choice(seed_alphas)
        mutated = mutate_alpha(parent)
        if mutated:
            population.append(mutated)

    set_global_data(data, train_start_date=train_start_date, train_end_date=train_end_date)
    best_ever = (None, -999.0)
    stagnation_count = 0
    immigration_count = 0
    all_results_history = []

    # CogAlpha: 누적 adaptive feedback
    adaptive_feedback = ""
    llm_injection_count = 0

    elite_count = max(5, population_size // 20)  # 5% 엘리트 (다양성 우선)
    base_mutation_rate = 0.50  # 기본 변이율 (탐색 비중 강화)

    # LLM mutation/crossover 주기 (더 빈번한 LLM 개입으로 다양성 극대화)
    LLM_MUTATION_INTERVAL = 3
    LLM_CROSSOVER_INTERVAL = 5

    for gen in range(1, generations + 1):
        # 적응적 변이율: 정체 시 변이 비중 증가
        mutation_rate = min(0.80, base_mutation_rate + stagnation_count * 0.05)
        crossover_rate = 1.0 - mutation_rate

        print(f"\n  세대 {gen}/{generations} (변이율: {mutation_rate:.0%}, 정체: {stagnation_count})")

        with Pool(8, initializer=set_global_data, initargs=(data, train_start_date, train_end_date)) as pool:
            results = pool.map(evaluate_alpha_worker, population)

        # 적합도 공유 적용 (같은 구조끼리 fitness 분산)
        raw_scores = sorted(results, key=lambda x: x[1], reverse=True)
        all_results_history.extend([(a, ic) for a, ic in raw_scores if ic > -999.0])

        # 순수 train fitness + fitness sharing (validation 없음)
        fitness_scores = _fitness_sharing(raw_scores)

        best_ic = raw_scores[0][1]  # 공유 전 실제 IC
        median_ic = raw_scores[len(raw_scores)//2][1] if raw_scores else -999.0
        unique_structures = len(set(_get_alpha_structure(a) for a, _ in raw_scores if _ > -999.0))
        print(f"    최고 IC: {best_ic:.4f}  중앙값: {median_ic:.4f}  고유구조: {unique_structures}개")

        # ── CogAlpha: Per-Generation Adaptive Feedback ──
        adaptive_feedback = _build_adaptive_feedback(raw_scores, prev_feedback=adaptive_feedback)

        if best_ic > best_ever[1]:
            best_ever = raw_scores[0]
            stagnation_count = 0
            print(f"    🏆 신기록!")
        else:
            stagnation_count += 1

        # 이민(immigration): 정체 시 LLM-guided 개체 주입 (CogAlpha 스타일)
        if stagnation_count >= 4 and immigration_count < 5:
            immigration_count += 1
            stagnation_count = 0
            n_immigrants = int(population_size * 0.30)  # 30% 교체
            print(f"    🌍 이민 #{immigration_count}: {n_immigrants}개 새 개체 주입 (LLM-guided)")

            # CogAlpha 개선: 이민 시 LLM mutation + 랜덤 mutation 혼합
            top_for_llm = [(a, ic) for a, ic in raw_scores[:10] if ic > -999.0]
            llm_immigrants = _llm_guided_mutation(top_for_llm, adaptive_feedback, num_mutations=min(15, n_immigrants // 3))
            if llm_immigrants:
                print(f"      🤖 LLM mutation: {len(llm_immigrants)}개 생성")
                llm_injection_count += len(llm_immigrants)

            # 나머지는 랜덤 변이로 채움
            random_immigrants = []
            needed = n_immigrants - len(llm_immigrants)
            for _ in range(needed):
                parent = random.choice(seed_alphas)
                for _ in range(random.randint(2, 3)):
                    m = mutate_alpha(parent)
                    if m:
                        parent = m
                random_immigrants.append(parent)

            immigrants = llm_immigrants + random_immigrants
            # 하위 25% 교체
            population = [a for a, _ in fitness_scores[:population_size - n_immigrants]] + immigrants[:n_immigrants]
            continue

        # 최종 종료: 이민 3회 후에도 5세대 무개선
        if stagnation_count >= 4:
            print(f"    ⏹️  이민 {immigration_count}회 후 4세대 무개선 → 종료")
            break

        # ── CogAlpha: 주기적 LLM-guided Mutation 주입 ──
        llm_offspring = []
        if gen % LLM_MUTATION_INTERVAL == 0:
            top_for_llm = [(a, ic) for a, ic in raw_scores[:10] if ic > -999.0]
            print(f"    🤖 LLM-guided mutation (세대 {gen})...")
            llm_mutated = _llm_guided_mutation(top_for_llm, adaptive_feedback, num_mutations=15)
            if llm_mutated:
                llm_offspring.extend(llm_mutated)
                llm_injection_count += len(llm_mutated)
                print(f"      ✅ LLM mutation: {len(llm_mutated)}개 생성")

        # ── CogAlpha: 주기적 LLM-guided Crossover 주입 ──
        if gen % LLM_CROSSOVER_INTERVAL == 0:
            top_for_llm = [(a, ic) for a, ic in raw_scores[:10] if ic > -999.0]
            print(f"    🤖 LLM-guided crossover (세대 {gen})...")
            llm_crossed = _llm_guided_crossover(top_for_llm, adaptive_feedback, num_children=8)
            if llm_crossed:
                llm_offspring.extend(llm_crossed)
                llm_injection_count += len(llm_crossed)
                print(f"      ✅ LLM crossover: {len(llm_crossed)}개 생성")

        # 다음 세대 구성
        next_population = []

        # 엘리트 보존 (7%)
        for alpha, _ in fitness_scores[:elite_count]:
            next_population.append(alpha)

        # LLM offspring 주입 (하위 개체 교체)
        if llm_offspring:
            next_population.extend(llm_offspring)

        # 토너먼트 선택 + 교차/변이 (나머지)
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

    print(f"\n    📊 LLM 주입 총계: {llm_injection_count}개 (mutation + crossover)")

    # Top-20 다양한 알파 선택 (main에서 val/test IC로 최종 5개 선택)
    top_diverse = _select_diverse_top_n(all_results_history, n=30)

    # Proven seeds 항상 최종 후보에 포함 (GP에서 탈락해도 main()에서 재평가)
    existing_exprs = {a for a, _ in top_diverse}
    for seed in seed_alphas[:14]:  # 첫 14개 = proven seeds
        if seed not in existing_exprs:
            top_diverse.append((seed, 0.0))

    return best_ever, top_diverse


def _make_cv_folds(close_index, n_folds=4, test_days=60, purge_days=20):
    """Purged Walk-Forward CV 폴드 생성.

    Expanding window train + 고정 test + purge gap (forward return 누출 방지).
    뒤에서부터 역순으로 fold를 배치하여 최신 데이터를 항상 테스트.
    """
    n_total = len(close_index)
    min_train_days = 120  # 최소 train 기간

    folds = []
    current_end = n_total - 1

    for _ in range(n_folds):
        test_end_idx = current_end
        test_start_idx = max(0, current_end - test_days + 1)
        train_end_idx = test_start_idx - purge_days - 1
        train_start_idx = 0  # expanding window

        if train_end_idx - train_start_idx + 1 < min_train_days:
            break  # train 데이터 부족

        folds.append((
            close_index[train_start_idx],
            close_index[train_end_idx],
            close_index[test_start_idx],
            close_index[test_end_idx],
        ))
        current_end = test_start_idx - 1  # 다음 fold는 이전 기간

    folds.reverse()  # 시간순으로 정렬
    return folds


def main():
    print("=" * 80)
    print("Alpha-GPT: 20-day (1-month) Forward with GPT-4o (v10 — CogAlpha: LLM-guided Evolution)")
    print("=" * 80)
    print()

    # 1. 전체 데이터 로드
    full_data = load_market_data()

    # 2. Purged Walk-Forward CV 폴드 생성
    close = full_data['close']
    cv_folds = _make_cv_folds(close.index, n_folds=4, test_days=60, purge_days=20)

    print(f"\n📐 Purged Walk-Forward CV ({len(cv_folds)} folds, purge=20d):")
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(cv_folds, 1):
        tr_len = len(close.loc[tr_s:tr_e])
        te_len = len(close.loc[te_s:te_e])
        print(f"   Fold {i}: Train [{tr_s}~{tr_e}] ({tr_len}d) | Test [{te_s}~{te_e}] ({te_len}d)")

    # 3. GPT-4o 시드 생성
    seed_alphas = generate_seed_alphas_gpt4o()

    # 4. GP 진화 — 가장 큰 fold의 train 기간으로 1회 실행
    largest_fold = cv_folds[-1]
    gp_train_end = largest_fold[1]

    (best_alpha, best_ic), top_diverse = genetic_programming(
        seed_alphas,
        full_data,
        train_start_date=None,
        train_end_date=gp_train_end,
        generations=70,
        population_size=300
    )

    # 5. Top 후보를 모든 CV fold에서 평가
    print("\n" + "=" * 80)
    print(f"📊 Cross-Validation: Top-{len(top_diverse)} candidates x {len(cv_folds)} folds")
    print("=" * 80)

    all_candidates = []
    for i, (alpha, gp_fitness) in enumerate(top_diverse, 1):
        fold_train_ics = []
        fold_test_ics = []
        fold_test_irs = []

        for fi, (tr_s, tr_e, te_s, te_e) in enumerate(cv_folds):
            train_ic = _compute_raw_ic(alpha, full_data, date_start=tr_s, date_end=tr_e)
            if train_ic <= -999.0:
                train_ic = 0.0
            fold_train_ics.append(train_ic)

            test_ic_list, _ = _compute_ic_series(alpha, full_data, date_start=te_s, date_end=te_e)
            if len(test_ic_list) >= 5:
                test_ic = float(np.mean(test_ic_list))
                test_std = float(np.std(test_ic_list))
                test_ir = test_ic / max(test_std, 0.001)
            else:
                test_ic = -0.05
                test_ir = -1.0
            fold_test_ics.append(test_ic)
            fold_test_irs.append(test_ir)

        # CV 일관성 지표
        n_positive_folds = sum(1 for ic in fold_test_ics if ic > 0)
        mean_test_ic = float(np.mean(fold_test_ics))
        mean_test_ir = float(np.mean(fold_test_irs))
        mean_train_ic = float(np.mean(fold_train_ics))

        # 팩터 분류
        factors = []
        if any(kw in alpha for kw in ['close', 'open_price', 'high', 'low', 'vwap']):
            factors.append('price')
        if any(kw in alpha for kw in ['volume', 'amount', 'vol_ratio']):
            factors.append('volume')
        if 'returns' in alpha:
            factors.append('returns')
        if any(kw in alpha for kw in ['high_low_range', 'body', 'upper_shadow', 'lower_shadow', 'atr_ratio']):
            factors.append('volatility')
        if any(kw in alpha for kw in ['amihud', 'gap', 'intraday_ret']):
            factors.append('micro')
        if any(kw in alpha for kw in ['foreign_net_ratio', 'inst_net_ratio', 'retail_net_ratio', 'foreign_ownership_pct']):
            factors.append('flow')
        factor_str = '+'.join(factors) if factors else 'unknown'

        all_candidates.append({
            'expr': alpha,
            'mean_train_ic': mean_train_ic,
            'mean_test_ic': mean_test_ic,
            'mean_test_ir': mean_test_ir,
            'n_positive_folds': n_positive_folds,
            'fold_test_ics': fold_test_ics,
            'factors': factor_str,
        })

        if i <= 10:
            fold_str = ' '.join([f"F{fi+1}:{ic:+.3f}" for fi, ic in enumerate(fold_test_ics)])
            print(f"  #{i:2d} [{factor_str:20s}] Train={mean_train_ic:.4f} | Test={mean_test_ic:.4f} | "
                  f"Pos={n_positive_folds}/{len(cv_folds)} | {fold_str}")

    # 6. CV 일관성 기반 최종 5개 선별 — 변수 조합 다양성 강제
    #    같은 변수 조합(예: foreign_ownership+vol_ratio+amihud+close)은 최대 2개
    #    → 나머지 3개는 반드시 다른 변수 조합이어야 함
    MAX_SAME_VARS = 2

    tier1 = [a for a in all_candidates if a['n_positive_folds'] >= 3]
    tier2 = [a for a in all_candidates if a['n_positive_folds'] >= 2 and a not in tier1]
    tier3 = [a for a in all_candidates if a not in tier1 and a not in tier2]

    tier1.sort(key=lambda x: (x['n_positive_folds'], x['mean_test_ic']), reverse=True)
    tier2.sort(key=lambda x: x['mean_test_ic'], reverse=True)
    tier3.sort(key=lambda x: x['mean_train_ic'], reverse=True)

    ranked_candidates = tier1 + tier2 + tier3

    validated_alphas = []
    var_sig_counts = {}
    for a in ranked_candidates:
        sig = _get_variable_signature(a['expr'])
        count = var_sig_counts.get(sig, 0)
        if count < MAX_SAME_VARS:
            validated_alphas.append(a)
            var_sig_counts[sig] = count + 1
            if len(validated_alphas) >= 5:
                break

    # 다양성 통계 출력
    unique_var_sigs = len(set(_get_variable_signature(a['expr']) for a in validated_alphas))
    print(f"\n   🧬 변수 다양성: {unique_var_sigs}개 고유 변수 조합 / {len(validated_alphas)}개 알파 "
          f"(같은 조합 최대 {MAX_SAME_VARS}개)")
    for sig, cnt in sorted(var_sig_counts.items(), key=lambda x: -x[1]):
        if cnt > 0:
            var_list = sorted(sig)
            print(f"      {cnt}개: {{{', '.join(var_list)}}}")

    print("\n" + "=" * 80)
    print("🏆 TOP 5 ALPHAS (CV-validated)")
    print("=" * 80)

    for i, a in enumerate(validated_alphas, 1):
        n_pos = a['n_positive_folds']
        status = "✅" if n_pos >= 3 else ("🔶" if n_pos >= 2 else "⚠️")
        fold_detail = ' '.join([f"{ic:+.3f}" for ic in a['fold_test_ics']])
        print(f"\n  #{i} {status} (positive folds: {n_pos}/{len(cv_folds)})")
        print(f"     Train IC: {a['mean_train_ic']:.4f} | Test IC: {a['mean_test_ic']:.4f} | "
              f"Test IR: {a['mean_test_ir']:.2f} [{a['factors']}]")
        print(f"     Fold ICs: [{fold_detail}]")
        print(f"     {a['expr'][:100]}{'...' if len(a['expr']) > 100 else ''}")

    # 7. 최종 Best 선정
    if tier1:
        final_best = tier1[0]
    elif tier2:
        final_best = tier2[0]
    else:
        final_best = validated_alphas[0] if validated_alphas else {
            'expr': best_alpha, 'mean_train_ic': best_ic, 'mean_test_ic': -999,
            'mean_test_ir': -999, 'n_positive_folds': 0, 'fold_test_ics': [], 'factors': '?'
        }

    print("\n" + "=" * 80)
    print("🥇 FINAL BEST (CV-validated)")
    print("=" * 80)
    print(f"Mean Train IC:  {final_best['mean_train_ic']:.4f}")
    print(f"Mean Test IC:   {final_best['mean_test_ic']:.4f}")
    print(f"Mean Test IR:   {final_best['mean_test_ir']:.2f}")
    print(f"Positive Folds: {final_best['n_positive_folds']}/{len(cv_folds)}")
    print(f"Factors:        {final_best['factors']}")
    print(f"Expression:     {final_best['expr']}")

    # 8. DB 저장 (Top-5 전부)
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
                float(a['mean_test_ic']),
                f"20d fwd, train={a['mean_train_ic']:.4f}, test={a['mean_test_ic']:.4f}, "
                f"IR={a['mean_test_ir']:.2f}, pos_folds={a['n_positive_folds']}/{len(cv_folds)}, "
                f"{a['factors']}, v10-cogalpha-cv"
            ))

        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ {len(validated_alphas)}개 알파 저장 완료!")
    except Exception as e:
        print(f"⚠️  DB 저장 실패: {e}")

    # 9. Multi-Alpha Ensemble용 JSON 내보내기
    alpha_export = []
    for a in validated_alphas:
        alpha_export.append({
            'expression': a['expr'],
            'mean_test_ic': float(a['mean_test_ic']),
            'mean_test_ir': float(a['mean_test_ir']),
            'n_positive_folds': a['n_positive_folds'],
            'factors': a['factors'],
        })

    export_path = project_root / 'best_alphas.json'
    with open(export_path, 'w') as f:
        json.dump(alpha_export, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Multi-Alpha Ensemble 내보내기: {export_path}")
    for i, ae in enumerate(alpha_export, 1):
        print(f"   #{i} IC={ae['mean_test_ic']:.4f} IR={ae['mean_test_ir']:.2f} "
              f"[{ae['factors']}] {ae['expression'][:80]}...")

    print("\n🎉 완료!")

if __name__ == "__main__":
    main()
