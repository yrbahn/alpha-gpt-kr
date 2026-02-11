"""
Analyst Agent
백테스트 결과를 분석하고 자연어로 해석하는 에이전트
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import json
from loguru import logger


@dataclass
class AnalysisReport:
    """분석 리포트"""
    summary: str
    key_findings: List[str]
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    risk_assessment: str


class Analyst:
    """
    Analyst Agent
    
    역할:
    1. 백테스트 결과 해석
    2. 알파의 강점과 약점 분석
    3. 개선 방향 제안
    4. 리스크 평가
    """
    
    SYSTEM_PROMPT = """당신은 전문 퀀트 애널리스트입니다.
백테스트 결과를 분석하고 투자자가 이해하기 쉽게 해석합니다.

분석 시 다음을 고려하세요:
1. IC (Information Coefficient): 예측력의 척도
   - IC > 0.03: 우수
   - 0.01 < IC < 0.03: 양호
   - IC < 0.01: 미흡

2. Sharpe Ratio: 위험 대비 수익
   - Sharpe > 2.0: 우수
   - 1.0 < Sharpe < 2.0: 양호
   - Sharpe < 1.0: 미흡

3. Turnover: 거래 비용
   - Turnover < 0.3: 저회전
   - 0.3 < Turnover < 1.0: 중회전
   - Turnover > 1.0: 고회전 (비용 주의)

4. Maximum Drawdown: 리스크
   - MDD > -10%: 낮은 리스크
   - -20% < MDD < -10%: 중간 리스크
   - MDD < -20%: 높은 리스크

한국 증시 특성을 고려하여 실용적인 조언을 제공하세요."""
    
    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLM 클라이언트
        """
        self.llm = llm_client
        logger.info("Analyst initialized")
    
    def analyze_backtest(self, 
                        backtest_result,
                        alpha_expr: str) -> AnalysisReport:
        """
        백테스트 결과 분석
        
        Args:
            backtest_result: BacktestResult 객체
            alpha_expr: 알파 표현식
            
        Returns:
            AnalysisReport
        """
        logger.info(f"Analyzing backtest result for: {alpha_expr[:100]}")
        
        # 메트릭 추출
        metrics = self._extract_metrics(backtest_result)
        
        # LLM 프롬프트 구성
        prompt = self._build_analysis_prompt(alpha_expr, metrics)
        
        # LLM 호출
        response = self._call_llm(prompt)
        
        # 응답 파싱
        report = self._parse_report(response)
        
        logger.info("Analysis complete")
        return report
    
    def compare_alphas(self,
                      results: List[tuple],  # [(alpha_expr, backtest_result), ...]
                      top_n: int = 5) -> str:
        """
        여러 알파 비교 분석
        
        Args:
            results: (알파 표현식, 백테스트 결과) 튜플 리스트
            top_n: 상위 몇 개 선택
            
        Returns:
            비교 분석 텍스트
        """
        logger.info(f"Comparing {len(results)} alphas")
        
        # IC 기준 정렬
        sorted_results = sorted(results, key=lambda x: x[1].ic, reverse=True)
        
        comparison = "## 알파 비교 분석\n\n"
        
        comparison += "### Top {} Alphas (IC 기준)\n\n".format(min(top_n, len(sorted_results)))
        
        for i, (expr, result) in enumerate(sorted_results[:top_n], 1):
            comparison += f"""
**{i}. {expr[:80]}{'...' if len(expr) > 80 else ''}**
- IC: {result.ic:.4f} (±{result.ic_std:.4f})
- Sharpe: {result.sharpe_ratio:.2f}
- Annual Return: {result.annual_return:.2%}
- MDD: {result.max_drawdown:.2%}
- Turnover: {result.turnover:.2%}

"""
        
        # 종합 분석
        prompt = f"""다음 상위 알파들의 특징을 종합 분석하세요:

{comparison}

공통점, 차이점, 그리고 최적의 조합 방법을 제안하세요."""
        
        summary = self._call_llm(prompt)
        comparison += "\n### 종합 분석\n" + summary
        
        return comparison
    
    def suggest_improvements(self,
                           alpha_expr: str,
                           backtest_result) -> List[str]:
        """
        개선 방향 제안
        
        Args:
            alpha_expr: 알파 표현식
            backtest_result: 백테스트 결과
            
        Returns:
            개선 제안 리스트
        """
        metrics = self._extract_metrics(backtest_result)
        
        prompt = f"""다음 알파의 개선 방향을 구체적으로 제안하세요:

알파: {alpha_expr}

성능:
- IC: {metrics['ic']:.4f}
- Sharpe: {metrics['sharpe']:.2f}
- Turnover: {metrics['turnover']:.2%}
- MDD: {metrics['mdd']:.2%}

약점을 보완하고 강점을 살릴 수 있는 3-5가지 구체적인 개선안을 제시하세요.
JSON 형식으로 응답:
{{
  "suggestions": [
    "제안 1",
    "제안 2",
    ...
  ]
}}
"""
        
        response = self._call_llm(prompt)
        
        try:
            data = json.loads(response)
            return data.get('suggestions', [])
        except:
            return ["개선 제안을 생성하지 못했습니다."]
    
    def _extract_metrics(self, result) -> Dict:
        """백테스트 결과에서 메트릭 추출"""
        return {
            'ic': result.ic,
            'ic_std': result.ic_std,
            'ir': result.ir,
            'sharpe': result.sharpe_ratio,
            'annual_return': result.annual_return,
            'total_return': result.total_return,
            'mdd': result.max_drawdown,
            'turnover': result.turnover,
            'win_rate': result.win_rate,
            'num_trades': result.num_trades
        }
    
    def _build_analysis_prompt(self, alpha_expr: str, metrics: Dict) -> str:
        """분석 프롬프트 생성"""
        prompt = f"""{self.SYSTEM_PROMPT}

## 알파 표현식
{alpha_expr}

## 백테스트 결과
- **IC (Information Coefficient)**: {metrics['ic']:.4f} (±{metrics['ic_std']:.4f})
- **IR (Information Ratio)**: {metrics['ir']:.2f}
- **Sharpe Ratio**: {metrics['sharpe']:.2f}
- **Annual Return**: {metrics['annual_return']:.2%}
- **Total Return**: {metrics['total_return']:.2%}
- **Maximum Drawdown**: {metrics['mdd']:.2%}
- **Turnover**: {metrics['turnover']:.2%}
- **Win Rate**: {metrics['win_rate']:.2%}
- **Number of Trades**: {metrics['num_trades']:,}

## 분석 요청
위 결과를 종합적으로 분석하고 JSON 형식으로 응답하세요:

{{
  "summary": "전체 요약 (2-3문장)",
  "key_findings": ["주요 발견 1", "주요 발견 2", ...],
  "strengths": ["강점 1", "강점 2", ...],
  "weaknesses": ["약점 1", "약점 2", ...],
  "suggestions": ["개선 제안 1", "개선 제안 2", ...],
  "risk_assessment": "리스크 평가"
}}
"""
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """LLM 호출"""
        try:
            if hasattr(self.llm, 'chat'):
                response = self.llm.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                return response.choices[0].message.content
            
            elif hasattr(self.llm, 'messages'):
                response = self.llm.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2000,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
                
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._generate_fallback_analysis()
    
    def _generate_fallback_analysis(self) -> str:
        """폴백 분석"""
        return json.dumps({
            "summary": "분석을 생성하지 못했습니다.",
            "key_findings": [],
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "risk_assessment": "알 수 없음"
        })
    
    def _parse_report(self, response: str) -> AnalysisReport:
        """LLM 응답 파싱"""
        try:
            data = json.loads(response)
            
            return AnalysisReport(
                summary=data.get('summary', ''),
                key_findings=data.get('key_findings', []),
                strengths=data.get('strengths', []),
                weaknesses=data.get('weaknesses', []),
                suggestions=data.get('suggestions', []),
                risk_assessment=data.get('risk_assessment', '')
            )
        except Exception as e:
            logger.error(f"Failed to parse report: {e}")
            return AnalysisReport(
                summary="분석 파싱 실패",
                key_findings=[],
                strengths=[],
                weaknesses=[],
                suggestions=[],
                risk_assessment="알 수 없음"
            )
    
    def format_report(self, report: AnalysisReport) -> str:
        """리포트를 보기 좋게 포맷팅"""
        formatted = f"""
# 알파 분석 리포트

## 📊 요약
{report.summary}

## 🔍 주요 발견
"""
        for finding in report.key_findings:
            formatted += f"- {finding}\n"
        
        formatted += "\n## ✅ 강점\n"
        for strength in report.strengths:
            formatted += f"- {strength}\n"
        
        formatted += "\n## ⚠️ 약점\n"
        for weakness in report.weaknesses:
            formatted += f"- {weakness}\n"
        
        formatted += "\n## 💡 개선 제안\n"
        for suggestion in report.suggestions:
            formatted += f"- {suggestion}\n"
        
        formatted += f"\n## 🛡️ 리스크 평가\n{report.risk_assessment}\n"
        
        return formatted
