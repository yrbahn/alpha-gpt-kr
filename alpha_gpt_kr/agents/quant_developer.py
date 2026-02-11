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
    description: str  # 설명
    category: str  # 카테고리 (momentum, value, reversal 등)
    complexity: int  # 복잡도 (1-10)
    operators_used: List[str]  # 사용된 연산자들


class QuantDeveloper:
    """
    Quant Developer Agent
    
    역할:
    1. 정제된 아이디어를 알파 표현식으로 변환
    2. 유효한 Python/pandas 코드 생성
    3. 다양한 변형 생성
    """
    
    SYSTEM_PROMPT = """당신은 전문 퀀트 개발자입니다.
트레이딩 아이디어를 실행 가능한 알파 표현식(alpha expression)으로 변환합니다.

## 사용 가능한 연산자 (AlphaOperators 클래스 메서드)

### Time-series Operators
- ts_delta(x, periods): 차분
- ts_mean(x, window): 이동 평균
- ts_std(x, window): 이동 표준편차
- ts_corr(x, y, window): 이동 상관계수
- ts_rank(x, window): 윈도우 내 순위
- ts_ema(x, window, alpha): 지수 이동 평균
- ts_zscore_scale(x, window): Z-score 정규화
- ts_min(x, window), ts_max(x, window): 최소/최대값
- ts_ir(x, window): Information Ratio (mean/std)

### Cross-sectional Operators
- zscore_scale(x): 횡단면 Z-score
- normed_rank(x): 정규화 순위 (0~1)
- winsorize_scale(x, lower, upper): 이상치 제거

### Group-wise Operators (산업 중립화 등)
- grouped_demean(x, groups): 그룹 평균 제거
- grouped_zscore_scale(x, groups): 그룹별 Z-score

### Element-wise Operators
- abs(x), log(x), sign(x), pow(x, n)
- add(x, y), minus(x, y), cwise_mul(x, y), div(x, y)
- greater(x, y), less(x, y)
- relu(x): max(0, x)

## 데이터 필드
- close, open, high, low: 가격 데이터
- volume, amount: 거래량, 거래대금
- vwap: Volume-weighted average price
- returns: 수익률

## 작성 규칙
1. ops.{연산자명} 형식으로 호출
2. 모든 표현식은 유효한 Python 코드여야 함
3. 결과는 pandas DataFrame이어야 함
4. 복잡도는 적절히 유지 (과도한 중첩 지양)
5. 한국 증시 특성 고려 (산업 중립화 등)

## 예시
```python
# 모멘텀: 20일 수익률
ops.ts_delta(close, 20) / close.shift(20)

# 리버설: 단기 과매도 후 반등
ops.cwise_mul(
    ops.greater(ops.ts_delta(close, 1), 0),  # 당일 상승
    ops.less(ops.ts_rank(close, 5), 0.2)     # 5일 중 하위 20%
)

# 거래량-가격 괴리: 산업 중립화
ops.grouped_demean(
    ops.ts_corr(volume, close, 20),
    sector_groups
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
    
    def _build_generation_prompt(self,
                                 idea: str,
                                 fields: List[str],
                                 examples: List[Dict],
                                 num_variations: int) -> str:
        """알파 생성 프롬프트"""
        
        examples_str = ""
        if examples:
            examples_str = "## 유사한 알파 예시\n"
            for ex in examples[:3]:
                examples_str += f"- {ex['expr']}\n  설명: {ex['desc']}\n\n"
        
        prompt = f"""{self.SYSTEM_PROMPT}

## 트레이딩 아이디어
{idea}

## 사용 가능한 데이터 필드
{', '.join(fields)}

{examples_str}

## 요구사항
{num_variations}개의 서로 다른 알파 표현식을 생성하세요.
각 표현식은 위 아이디어를 다른 방식으로 구현해야 합니다.

## 출력 형식 (JSON array)
[
  {{
    "expr": "ops.ts_delta(close, 20) / close.shift(20)",
    "description": "20일 모멘텀",
    "category": "momentum",
    "complexity": 3,
    "operators_used": ["ts_delta", "shift", "div"]
  }},
  ...
]

JSON 배열 형식으로만 응답하세요."""
        
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
        """LLM 호출"""
        try:
            if hasattr(self.llm, 'chat'):
                # OpenAI
                response = self.llm.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            
            elif hasattr(self.llm, 'messages'):
                # Anthropic
                response = self.llm.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4000,
                    temperature=0.7,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._generate_fallback_alphas()
    
    def _generate_fallback_alphas(self) -> str:
        """폴백 알파 생성"""
        fallback = [
            {
                "expr": "ops.ts_delta(close, 20) / close.shift(20)",
                "description": "20일 모멘텀",
                "category": "momentum",
                "complexity": 3,
                "operators_used": ["ts_delta", "shift", "div"]
            },
            {
                "expr": "ops.ts_corr(volume, close, 10)",
                "description": "거래량-가격 상관관계",
                "category": "volume",
                "complexity": 2,
                "operators_used": ["ts_corr"]
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
                alphas.append(AlphaExpression(
                    expr=item.get('expr', ''),
                    description=item.get('description', ''),
                    category=item.get('category', 'unknown'),
                    complexity=item.get('complexity', 5),
                    operators_used=item.get('operators_used', [])
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
        
        prompt = f"""다음 알파 표현식을 사용자 피드백에 따라 개선하세요.

## 현재 알파
표현식: {alpha.expr}
설명: {alpha.description}

## 피드백
{feedback}

## 개선된 알파 (JSON)
{{
  "expr": "...",
  "description": "...",
  "category": "...",
  "complexity": 5,
  "operators_used": [...]
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
