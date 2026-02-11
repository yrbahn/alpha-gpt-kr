"""
Trading Idea Polisher Agent
사용자의 트레이딩 아이디어를 정제하고 구조화하는 에이전트
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import json
from loguru import logger


@dataclass
class PolishedIdea:
    """정제된 트레이딩 아이디어"""
    original_idea: str
    refined_idea: str
    market_context: str
    relevant_fields: List[str]
    patterns: List[str]
    constraints: List[str]


class TradingIdeaPolisher:
    """
    Trading Idea Polisher Agent
    
    역할:
    1. 사용자의 자연어 아이디어를 구조화
    2. 관련 데이터 필드 식별
    3. 시장 컨텍스트 추가
    4. 명확한 프롬프트 생성
    """
    
    SYSTEM_PROMPT = """당신은 전문 퀀트 리서처입니다. 
사용자의 트레이딩 아이디어를 분석하고 구조화된 형태로 정제하는 것이 임무입니다.

다음을 수행하세요:
1. 아이디어의 핵심 개념 추출
2. 필요한 데이터 필드 식별 (가격, 거래량, 재무제표 등)
3. 시장 패턴이나 이론적 배경 연결
4. 구현 가능한 형태로 재구조화
5. 제약사항이나 주의사항 명시

한국 증시 특성을 고려하여 분석하세요."""
    
    FIELD_LIBRARY = {
        "가격": ["open", "high", "low", "close", "vwap"],
        "거래량": ["volume", "amount", "trade_count"],
        "수익률": ["returns", "log_returns"],
        "모멘텀": ["roc", "momentum", "rsi"],
        "변동성": ["volatility", "atr", "std"],
        "시가총액": ["market_cap"],
        "재무": ["per", "pbr", "roe", "debt_ratio"],
        "산업": ["sector", "industry"],
    }
    
    def __init__(self, llm_client, knowledge_base: Optional[Dict] = None):
        """
        Args:
            llm_client: LLM 클라이언트 (OpenAI, Anthropic 등)
            knowledge_base: 지식 베이스 (논문, 알파 라이브러리 등)
        """
        self.llm = llm_client
        self.knowledge_base = knowledge_base or {}
        logger.info("TradingIdeaPolisher initialized")
    
    def polish(self, trading_idea: str) -> PolishedIdea:
        """
        트레이딩 아이디어 정제
        
        Args:
            trading_idea: 사용자의 원본 아이디어
            
        Returns:
            PolishedIdea
        """
        logger.info(f"Polishing idea: {trading_idea[:100]}...")
        
        # LLM 프롬프트 구성
        prompt = self._build_prompt(trading_idea)
        
        # LLM 호출
        response = self._call_llm(prompt)
        
        # 응답 파싱
        polished = self._parse_response(trading_idea, response)
        
        logger.info(f"Polished idea: {polished.refined_idea[:100]}...")
        return polished
    
    def _build_prompt(self, idea: str) -> str:
        """프롬프트 생성"""
        # 관련 지식 검색
        relevant_knowledge = self._retrieve_knowledge(idea)
        
        prompt = f"""{self.SYSTEM_PROMPT}

## 사용자 아이디어
{idea}

## 사용 가능한 데이터 필드
{json.dumps(self.FIELD_LIBRARY, ensure_ascii=False, indent=2)}

## 관련 지식 (참고용)
{relevant_knowledge}

## 출력 형식 (JSON)
{{
    "refined_idea": "명확하고 구체적으로 정제된 아이디어",
    "market_context": "관련 시장 이론이나 패턴",
    "relevant_fields": ["필요한 데이터 필드 리스트"],
    "patterns": ["탐지하려는 패턴들"],
    "constraints": ["제약사항이나 주의사항"]
}}

위 형식으로 응답해주세요."""
        
        return prompt
    
    def _retrieve_knowledge(self, query: str) -> str:
        """지식 베이스에서 관련 정보 검색"""
        if not self.knowledge_base:
            return "관련 지식 없음"
        
        # 간단한 키워드 매칭 (실제로는 임베딩 기반 검색)
        keywords = ["모멘텀", "밸류", "리버설", "거래량", "추세", "변동성"]
        
        relevant = []
        for keyword in keywords:
            if keyword in query and keyword in self.knowledge_base:
                relevant.append(f"- {keyword}: {self.knowledge_base[keyword]}")
        
        return "\n".join(relevant) if relevant else "관련 지식 없음"
    
    def _call_llm(self, prompt: str) -> str:
        """LLM 호출"""
        try:
            # OpenAI API 예시
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
            
            # Anthropic API 예시
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
            
            else:
                raise ValueError("Unsupported LLM client")
                
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # 폴백: 기본 정제
            return self._fallback_polish(prompt)
    
    def _fallback_polish(self, prompt: str) -> str:
        """LLM 실패 시 기본 정제"""
        return json.dumps({
            "refined_idea": "아이디어를 구조화하지 못했습니다.",
            "market_context": "알 수 없음",
            "relevant_fields": ["close", "volume"],
            "patterns": [],
            "constraints": ["LLM 호출 실패"]
        })
    
    def _parse_response(self, original: str, response: str) -> PolishedIdea:
        """LLM 응답 파싱"""
        try:
            data = json.loads(response)
            
            return PolishedIdea(
                original_idea=original,
                refined_idea=data.get("refined_idea", original),
                market_context=data.get("market_context", ""),
                relevant_fields=data.get("relevant_fields", []),
                patterns=data.get("patterns", []),
                constraints=data.get("constraints", [])
            )
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            return PolishedIdea(
                original_idea=original,
                refined_idea=original,
                market_context="",
                relevant_fields=["close", "volume"],
                patterns=[],
                constraints=["파싱 실패"]
            )
    
    def add_knowledge(self, key: str, value: str):
        """지식 베이스에 항목 추가"""
        self.knowledge_base[key] = value
        logger.debug(f"Added knowledge: {key}")
    
    def validate_idea(self, idea: PolishedIdea) -> bool:
        """정제된 아이디어 검증"""
        # 필수 필드 체크
        if not idea.refined_idea or len(idea.refined_idea) < 10:
            logger.warning("Refined idea too short")
            return False
        
        if not idea.relevant_fields:
            logger.warning("No relevant fields identified")
            return False
        
        return True
