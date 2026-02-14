"""
Quant Developer Agent
알파 표현식을 생성하는 에이전트
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import json
import re
from loguru import logger


@dataclass
class AlphaExpression:
    """알파 표현식"""
    expr: str  # 실행 가능한 Python 표현식
    description: str  # 설명 (alpha_name에서 파생)
    category: str  # 카테고리 (momentum_volume, volatility_adjusted 등)
    complexity: int  # 복잡도 (1-10)
    operators_used: List[str]  # 사용된 연산자들
    rationale: str = ""  # 경제적 논리 / 알파 설계 근거
    alpha_name: str = ""  # 고유 식별자 (예: Volume_Confirmed_Momentum_20d)
    timeframes_used: List[int] = None  # 사용된 시간대 (예: [5, 20, 60])


class QuantDeveloper:
    """
    Quant Developer Agent
    
    역할:
    1. 정제된 아이디어를 알파 표현식으로 변환
    2. 유효한 Python/pandas 코드 생성
    3. 다양한 변형 생성
    """
    
    SYSTEM_PROMPT = """### Role & Objective
You are an expert Quantitative Researcher (Quant) specializing in developing formulaic alphas for the Korean stock market (KRX).
Your goal is to translate natural language trading ideas into valid, high-performance alpha expressions.
Target performance: IC (Information Coefficient) > 0.03, Sharpe Ratio > 1.5.

### Dataset (Input Variables)
Daily stock data (OHLCV) as DataFrames — each column is a stock, each row is a date:
- `close`: Closing price
- `open_price`: Opening price
- `high`: Highest price of the day
- `low`: Lowest price of the day
- `volume`: Trading volume
- `vwap`: Volume-weighted average price
- `returns`: Daily returns (close-to-close)

### Operator List (Domain Specific Language)
You MUST use `ops.{operator_name}()` syntax. All operators return pandas DataFrames.

**1. Time-Series Operators** (operate on a single stock over time):
| Operator | Signature | Description | Use Case |
|---|---|---|---|
| ts_delta | (x, d) | x(t) - x(t-d) | Price/volume change |
| ts_delta_ratio | (x, d) | (x(t) - x(t-d)) / x(t-d) | Return calculation |
| ts_mean | (x, d) | d-day moving average | Smoothing, trend |
| ts_std | (x, d) | d-day moving std | Volatility measure |
| ts_sum | (x, d) | d-day rolling sum | Cumulative volume |
| ts_min / ts_max | (x, d) | d-day min/max | Support/resistance levels |
| ts_argmin / ts_argmax | (x, d) | Position of min/max in window | When was the recent high/low |
| ts_argmaxmin_diff | (x, d) | argmax - argmin | Trend direction detection |
| ts_max_diff | (x, d) | x(t) - d-day max | Drawdown from high |
| ts_min_diff | (x, d) | x(t) - d-day min | Recovery from low |
| ts_rank | (x, d) | Rank within d-day window [0,1] | Current price position |
| ts_corr | (x, y, d) | d-day rolling correlation | Price-volume relationship |
| ts_cov | (x, y, d) | d-day rolling covariance | Co-movement |
| ts_ema | (x, d, alpha) | Exponential moving average | Smooth trend |
| ts_zscore_scale | (x, d) | Z-score within d-day window | Mean reversion signal |
| ts_maxmin_scale | (x, d) | Min-Max normalize [0,1] | Price position in range |
| ts_skew | (x, d) | Rolling skewness | Tail risk / asymmetry |
| ts_kurt | (x, d) | Rolling kurtosis | Fat tail risk |
| ts_ir | (x, d) | mean / std | Time-series information ratio |
| ts_decayed_linear | (x, d) | Linear decay weighted avg | Recent-data emphasis |
| ts_linear_reg | (x, d) | Linear regression slope | Trend strength |
| ts_median | (x, d) | Rolling median | Robust average |
| ts_percentile | (x, d, q) | Rolling percentile | Quantile position |
| shift | (x, d) | Lag by d days | Delayed reference |

**2. Cross-Sectional Operators** (operate across all stocks at a point in time):
| Operator | Signature | Description | Use Case |
|---|---|---|---|
| normed_rank | (x) | Rank across stocks [0,1] | Market-neutral normalization |
| zscore_scale | (x) | Cross-sectional Z-score | Standardization |
| normed_rank_diff | (x, y) | rank(x) - rank(y) | Relative comparison |
| winsorize_scale | (x, lo, hi) | Clip outliers [default 5%,95%] | Robust scaling |
| cwise_max / cwise_min | (x, y) | Element-wise max/min | Conditional bounds |

**3. Element-wise Operators**:
| Operator | Signature | Description | Use Case |
|---|---|---|---|
| abs | (x) | Absolute value | Magnitude |
| log | (x) | Natural logarithm | Compress scale |
| sign | (x) | Sign: -1, 0, +1 | Direction only |
| neg | (x) | Negate: -x | Reverse signal |
| pow | (x, n) | x^n | Non-linear amplification |
| pow_sign | (x, n) | sign(x)*abs(x)^n | Non-linear, sign-preserving |
| relu | (x) | max(0, x) | Positive signal filter |
| add / minus | (x, y) | Addition / subtraction | Combine signals |
| cwise_mul | (x, y) | Element-wise multiply | Interaction / weighting |
| div | (x, y) | Safe division (0→NaN) | Ratio / normalization |
| greater / less | (x, y) | Comparison → 0/1 | Conditional filter |

### KRX Market Anomalies (Proven Factors)
Leverage these known patterns in the Korean market:
1. **Short-term Reversal (5~20d)**: Strong mean-reversion effect. Oversold stocks tend to bounce back.
2. **Volume Anomaly**: Abnormal volume surge → subsequent price reversal. Volume leads price.
3. **Low-Volatility Premium**: Low-vol stocks outperform high-vol stocks over time.
4. **Filtered Momentum**: Pure momentum is noisy; combining with volume/volatility filters improves IC.
5. **Price Position**: Distance from 52-week high predicts future returns.
6. **Price-Volume Divergence**: Price drops + volume dries up = selling exhaustion → reversal.
7. **Tail Risk**: Return skewness/kurtosis predict future cross-sectional returns.

### Constraints
1. **Neutrality**: All final expressions MUST be wrapped with `ops.normed_rank()` or `ops.zscore_scale()` for market neutrality.
2. **No Look-Ahead Bias**: Never use `shift(-n)` (negative shift). Only past data allowed.
3. **Robustness**: Use `ops.div()` (safe division) instead of raw `/` when denominator can be zero.
4. **Complexity**: 2~4 operator nesting levels. Single-operator alphas are too weak; 5+ levels risk overfitting.
5. **Multi-Factor**: Combine at least 2 different signal types (e.g., price + volume, momentum + volatility).
6. **Multi-Timeframe**: Use at least 2 different lookback windows (e.g., 5d + 20d, 10d + 60d).

### Alpha Design Patterns (Proven Combinations)

```python
# Pattern 1: Volume-Confirmed Momentum
# Momentum signal weighted by volume surge → higher conviction
ops.normed_rank(
    ops.cwise_mul(
        ops.ts_delta_ratio(close, 20),                          # 20d momentum
        ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20))  # volume surge ratio
    )
)

# Pattern 2: Volatility-Adjusted Reversal
# Short-term oversold + low volatility → reversal with less risk
ops.normed_rank(
    ops.div(
        ops.neg(ops.ts_zscore_scale(close, 10)),   # mean-reversion signal
        ops.ts_std(returns, 20)                     # penalize high volatility
    )
)

# Pattern 3: Smart Money (Price-Volume Divergence)
# Price falling + volume declining = selling exhaustion → expect bounce
ops.normed_rank(
    ops.neg(ops.ts_corr(
        ops.ts_delta(close, 5),
        ops.ts_delta(volume, 5),
        20
    ))
)

# Pattern 4: Multi-Timeframe Momentum Acceleration
# Short-term IR exceeding long-term IR = accelerating momentum
ops.normed_rank(
    ops.minus(
        ops.ts_ir(returns, 5),
        ops.ts_ir(returns, 20)
    )
)

# Pattern 5: Price Position + Volume Confirmation
# Near recent highs with strong volume = breakout potential
ops.normed_rank(
    ops.cwise_mul(
        ops.ts_maxmin_scale(close, 20),
        ops.normed_rank(ops.ts_mean(volume, 5))
    )
)

# Pattern 6: Trend + Tail Risk
# Strong uptrend with positive skewness = favorable risk profile
ops.normed_rank(
    ops.cwise_mul(
        ops.relu(ops.ts_linear_reg(close, 20)),
        ops.relu(ops.ts_skew(returns, 20))
    )
)

# Pattern 7: Composite (Volume Anomaly + Reversal + Volatility Filter)
# Volume spike + price drop + low volatility = high-conviction reversal
ops.normed_rank(
    ops.cwise_mul(
        ops.cwise_mul(
            ops.greater(ops.ts_delta_ratio(volume, 5), 0.5),
            ops.less(ops.ts_delta_ratio(close, 5), 0)
        ),
        ops.neg(ops.normed_rank(ops.ts_std(returns, 20)))
    )
)
```
"""
    
    OPERATOR_CATEGORIES = {
        "time_series": ["ts_delta", "ts_mean", "ts_std", "ts_corr", "ts_ema", "ts_rank"],
        "cross_sectional": ["zscore_scale", "normed_rank", "winsorize_scale"],
        "group_wise": ["grouped_demean", "grouped_zscore_scale"],
        "element_wise": ["abs", "log", "sign", "add", "minus", "div", "cwise_mul"]
    }
    
    def __init__(self, llm_client, alpha_library: Optional[Dict] = None):
        """
        Args:
            llm_client: LLM 클라이언트
            alpha_library: 기존 알파 라이브러리 (예시용)
        """
        self.llm = llm_client
        self.alpha_library = alpha_library or {}
        logger.info("QuantDeveloper initialized")
    
    def generate_alphas(self,
                       refined_idea: str,
                       relevant_fields: List[str],
                       num_variations: int = 10) -> List[AlphaExpression]:
        """
        알파 표현식 생성
        
        Args:
            refined_idea: 정제된 트레이딩 아이디어
            relevant_fields: 관련 데이터 필드
            num_variations: 생성할 변형 개수
            
        Returns:
            AlphaExpression 리스트
        """
        logger.info(f"Generating {num_variations} alpha variations")
        
        # 유사 알파 검색
        similar_alphas = self._find_similar_alphas(refined_idea)
        
        # LLM 프롬프트 구성
        prompt = self._build_generation_prompt(
            refined_idea, 
            relevant_fields, 
            similar_alphas,
            num_variations
        )
        
        # LLM 호출
        response = self._call_llm(prompt)
        
        # 응답 파싱
        alphas = self._parse_alphas(response)
        
        # 검증 및 필터링
        valid_alphas = [a for a in alphas if self._validate_expression(a)]
        
        logger.info(f"Generated {len(valid_alphas)} valid alphas")
        return valid_alphas[:num_variations]
    
    ALPHA_CATEGORIES = [
        ("momentum_volume", "Momentum confirmed by volume surge"),
        ("volatility_adjusted", "Signal adjusted/filtered by volatility"),
        ("short_term_reversal", "Mean-reversion exploiting KRX reversal effect"),
        ("multi_timeframe", "Combining short + medium + long timeframes"),
        ("price_volume_diverge", "Price-volume divergence / smart money"),
        ("trend_strength", "Trend strength via regression slope or IR"),
        ("tail_risk", "Skewness/kurtosis-based risk signal"),
        ("price_position", "Price position relative to recent high/low"),
        ("volume_anomaly", "Abnormal volume detection"),
        ("composite", "3+ factor composite signal"),
    ]

    def _build_generation_prompt(self,
                                 idea: str,
                                 fields: List[str],
                                 examples: List[Dict],
                                 num_variations: int) -> str:
        """알파 생성 프롬프트 (Thought Decompiler 방식)"""

        examples_str = ""
        if examples:
            examples_str = "\n### Existing Alpha Library (generate alphas with LOW correlation to these)\n"
            for ex in examples[:5]:
                examples_str += f"- `{ex['expr']}` — {ex['desc']}\n"

        # 요청할 카테고리 배분
        categories_to_use = list(self.ALPHA_CATEGORIES[:num_variations])
        if num_variations > len(self.ALPHA_CATEGORIES):
            categories_to_use = list(self.ALPHA_CATEGORIES) * (num_variations // len(self.ALPHA_CATEGORIES) + 1)
            categories_to_use = categories_to_use[:num_variations]

        category_guide = "\n".join(
            f"  {i+1}. `{cat}` — {desc}" for i, (cat, desc) in enumerate(categories_to_use)
        )

        prompt = f"""### Task
Generate {num_variations} diverse, high-performance alpha expressions for the following trading idea.

### Trading Idea
{idea}

### Available Data Fields
{', '.join(fields)}
{examples_str}

### Requirements

**Diversity** — Each alpha MUST belong to a DIFFERENT category:
{category_guide}

**Quality Checklist** — Every alpha must satisfy ALL of these:
- [ ] Multi-factor: combines 2+ distinct signal types (NOT single-operator)
- [ ] Market-neutral: wrapped with `ops.normed_rank()` or `ops.zscore_scale()`
- [ ] Multi-timeframe: uses 2+ different lookback windows (e.g., 5d & 20d)
- [ ] No look-ahead bias: no `shift(-n)` or future data
- [ ] Complexity 2~4 nesting levels (avoid overfitting)
- [ ] Safe division: use `ops.div()` instead of raw `/`
- [ ] Volume confirmation: price signals should incorporate volume where possible

**Think step-by-step** for each alpha:
1. What anomaly or pattern does this alpha exploit?
2. What is the economic intuition?
3. How do the operators combine to capture this signal?
4. Why would this work specifically in the Korean market?

### Output Format (Thought Decompiler)
Return a JSON array. Each element must follow this structure:

```json
[
  {{
    "alpha_name": "Volume_Confirmed_Momentum_20d",
    "category": "momentum_volume",
    "rationale": "Momentum alone is noisy. By weighting with volume surge ratio (5d/20d), we filter for momentum backed by increasing participation, which is more likely to persist.",
    "expression": "ops.normed_rank(ops.cwise_mul(ops.ts_delta_ratio(close, 20), ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20))))",
    "complexity": 4,
    "operators_used": ["normed_rank", "cwise_mul", "ts_delta_ratio", "div", "ts_mean"],
    "timeframes_used": [5, 20]
  }}
]
```

**CRITICAL**:
- Return ONLY the JSON array. No markdown, no explanation outside JSON.
- `expression` must be valid Python code using `ops.xxx()` syntax.
- `rationale` must explain the economic logic, not just describe the formula.
- `alpha_name` should be a unique, descriptive identifier (PascalCase with underscores)."""

        return prompt
    
    def _find_similar_alphas(self, idea: str) -> List[Dict]:
        """유사 알파 검색 (키워드 기반)"""
        if not self.alpha_library:
            return []
        
        keywords = self._extract_keywords(idea)
        similar = []
        
        for alpha_id, alpha_data in self.alpha_library.items():
            score = sum(1 for kw in keywords if kw in alpha_data.get('desc', '').lower())
            if score > 0:
                similar.append({
                    'expr': alpha_data.get('expr', ''),
                    'desc': alpha_data.get('desc', ''),
                    'score': score
                })
        
        # 점수순 정렬
        similar.sort(key=lambda x: x['score'], reverse=True)
        return similar[:5]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """키워드 추출"""
        keywords = [
            "모멘텀", "momentum", "추세", "trend",
            "리버설", "reversal", "반전",
            "밸류", "value", "가치",
            "거래량", "volume",
            "변동성", "volatility",
            "상관", "correlation"
        ]
        
        return [kw for kw in keywords if kw in text.lower()]
    
    def _call_llm(self, prompt: str) -> str:
        """LLM 호출 — system prompt와 user prompt를 분리하여 전송"""
        try:
            if hasattr(self.llm, 'chat'):
                # OpenAI
                response = self.llm.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=8000,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content

            elif hasattr(self.llm, 'messages'):
                # Anthropic
                response = self.llm.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=8000,
                    temperature=0.7,
                    system=self.SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._generate_fallback_alphas()
    
    def _generate_fallback_alphas(self) -> str:
        """폴백 알파 생성 (LLM 실패 시 사용)"""
        fallback = [
            {
                "alpha_name": "Volume_Confirmed_Momentum_20d",
                "expression": "ops.normed_rank(ops.cwise_mul(ops.ts_delta_ratio(close, 20), ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20))))",
                "category": "momentum_volume",
                "complexity": 4,
                "operators_used": ["normed_rank", "cwise_mul", "ts_delta_ratio", "div", "ts_mean"],
                "rationale": "Momentum alone is noisy. Volume surge ratio (5d/20d) filters for momentum backed by increasing participation.",
                "timeframes_used": [5, 20]
            },
            {
                "alpha_name": "Smart_Money_PV_Divergence",
                "expression": "ops.normed_rank(ops.neg(ops.ts_corr(ops.ts_delta(close, 5), ops.ts_delta(volume, 5), 20)))",
                "category": "price_volume_diverge",
                "complexity": 4,
                "operators_used": ["normed_rank", "neg", "ts_corr", "ts_delta"],
                "rationale": "Negative price-volume correlation indicates selling exhaustion — price drops on declining volume signal a likely reversal.",
                "timeframes_used": [5, 20]
            },
            {
                "alpha_name": "VolAdj_ShortTerm_Reversal",
                "expression": "ops.normed_rank(ops.div(ops.neg(ops.ts_zscore_scale(close, 10)), ops.ts_std(returns, 20)))",
                "category": "short_term_reversal",
                "complexity": 4,
                "operators_used": ["normed_rank", "div", "neg", "ts_zscore_scale", "ts_std"],
                "rationale": "KRX shows strong 5-20d reversal. Dividing by 20d volatility enhances the signal by favoring low-vol stocks where reversal is more reliable.",
                "timeframes_used": [10, 20]
            }
        ]
        return json.dumps(fallback)
    
    def _parse_alphas(self, response: str) -> List[AlphaExpression]:
        """LLM 응답 파싱"""
        alphas = []
        
        try:
            # JSON 추출 (마크다운 코드블록 제거)
            json_str = response
            if "```json" in response:
                json_str = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
                if json_str:
                    json_str = json_str.group(1)
            elif "```" in response:
                json_str = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
                if json_str:
                    json_str = json_str.group(1)
            
            data = json.loads(json_str)
            
            # 리스트가 아니면 리스트로 감싸기
            if isinstance(data, dict):
                data = [data]
            
            for item in data:
                # expression 또는 expr 필드 지원
                expr = item.get('expression', item.get('expr', ''))
                alpha_name = item.get('alpha_name', '')
                # description은 alpha_name에서 파생하거나 직접 제공
                description = item.get('description', alpha_name.replace('_', ' '))
                alphas.append(AlphaExpression(
                    expr=expr,
                    description=description,
                    category=item.get('category', 'unknown'),
                    complexity=item.get('complexity', 5),
                    operators_used=item.get('operators_used', []),
                    rationale=item.get('rationale', ''),
                    alpha_name=alpha_name,
                    timeframes_used=item.get('timeframes_used', [])
                ))
                
        except Exception as e:
            logger.error(f"Failed to parse alphas: {e}")
            logger.debug(f"Response: {response[:500]}")
        
        return alphas
    
    def _validate_expression(self, alpha: AlphaExpression) -> bool:
        """표현식 검증"""
        if not alpha.expr or len(alpha.expr) < 5:
            return False
        
        # ops. 접두사 확인
        if "ops." not in alpha.expr:
            logger.warning(f"Missing 'ops.' prefix: {alpha.expr}")
            return False
        
        # 기본 구문 검사
        if alpha.expr.count('(') != alpha.expr.count(')'):
            logger.warning(f"Unbalanced parentheses: {alpha.expr}")
            return False
        
        # 금지된 키워드 체크
        dangerous_keywords = ['exec', 'eval', 'import', '__']
        if any(kw in alpha.expr.lower() for kw in dangerous_keywords):
            logger.warning(f"Dangerous keyword found: {alpha.expr}")
            return False
        
        return True
    
    def refine_alpha(self, 
                    alpha: AlphaExpression, 
                    feedback: str) -> AlphaExpression:
        """
        피드백을 반영하여 알파 개선
        
        Args:
            alpha: 기존 알파
            feedback: 사용자 피드백
            
        Returns:
            개선된 AlphaExpression
        """
        logger.info(f"Refining alpha with feedback: {feedback[:100]}")
        
        prompt = f"""### Task: Refine Alpha
Improve the following alpha expression based on the user's feedback.

### Current Alpha
- **Name**: {alpha.alpha_name or alpha.description}
- **Expression**: `{alpha.expr}`
- **Category**: {alpha.category}
- **Rationale**: {alpha.rationale}

### User Feedback
{feedback}

### Refinement Guidelines
1. Address the feedback's weaknesses while preserving existing strengths.
2. MUST wrap output with `ops.normed_rank()` or `ops.zscore_scale()`.
3. Adding volume confirmation to price signals improves IC.
4. Keep complexity at 2~4 nesting levels (avoid overfitting).
5. Use `ops.div()` for safe division.

### Output Format (JSON)
Return a single JSON object:
{{
  "alpha_name": "Refined_Alpha_Name",
  "expression": "ops.normed_rank(...)",
  "category": "{alpha.category}",
  "complexity": 4,
  "operators_used": [...],
  "rationale": "Explain what was improved and why the new version is better.",
  "timeframes_used": [...]
}}
"""
        
        response = self._call_llm(prompt)
        refined = self._parse_alphas(response)
        
        return refined[0] if refined else alpha
    
    def add_to_library(self, alpha: AlphaExpression, alpha_id: str):
        """알파를 라이브러리에 추가"""
        self.alpha_library[alpha_id] = {
            'expr': alpha.expr,
            'desc': alpha.description,
            'category': alpha.category
        }
        logger.info(f"Added alpha to library: {alpha_id}")
