"""
Alpha-GPT Core
메인 시스템 클래스
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from loguru import logger
from dotenv import load_dotenv

from .agents.trading_idea_polisher import TradingIdeaPolisher
from .agents.quant_developer import QuantDeveloper, AlphaExpression
from .agents.analyst import Analyst
from .mining.genetic_programming import AlphaGeneticProgramming
from .data.postgres_loader import PostgresDataLoader
from .backtest.engine import BacktestEngine, BacktestResult


@dataclass
class AlphaMiningResult:
    """알파 마이닝 결과"""
    top_alphas: List[tuple]  # [(expression, backtest_result), ...]
    best_ic: float
    best_sharpe: float
    analysis_report: str
    iteration_history: List[Dict]


class AlphaGPT:
    """
    Alpha-GPT Main System
    
    논문의 전체 워크플로우 구현:
    1. Ideation: 아이디어 정제
    2. Implementation: 알파 생성 + GP 진화
    3. Review: 백테스트 + 분석
    """
    
    def __init__(self,
                 market: str = "KRX",
                 llm_provider: str = "openai",
                 model: str = "gpt-4-turbo-preview",
                 data_dir: str = "./data",
                 cache_dir: str = "./data/cache"):
        """
        Args:
            market: 시장 (KRX, KOSPI, KOSDAQ)
            llm_provider: LLM 제공자 (openai, anthropic)
            model: 모델명
            data_dir: 데이터 디렉토리
            cache_dir: 캐시 디렉토리
        """
        load_dotenv()
        
        self.market = market
        self.llm_provider = llm_provider
        self.model = model
        
        # LLM 클라이언트 초기화
        self.llm_client = self._initialize_llm(llm_provider, model)
        
        # 에이전트 초기화
        self.idea_polisher = TradingIdeaPolisher(self.llm_client)
        self.quant_developer = QuantDeveloper(self.llm_client)
        self.analyst = Analyst(self.llm_client)
        
        # 데이터 로더 (PostgreSQL)
        self.data_loader = PostgresDataLoader()
        
        # 상태
        self.current_data = None
        self.current_universe = None
        
        logger.info(f"AlphaGPT initialized: market={market}, llm={llm_provider}/{model}")
    
    def _initialize_llm(self, provider: str, model: str):
        """LLM 클라이언트 초기화"""
        if provider == "openai":
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            return openai.OpenAI(api_key=api_key)
        
        elif provider == "anthropic":
            import anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            return anthropic.Anthropic(api_key=api_key)
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def load_data(self,
                  universe: Optional[List[str]] = None,
                  start_date: str = "2020-01-01",
                  end_date: str = "2024-12-31",
                  include_technical: bool = False):
        """
        데이터 로드 (PostgreSQL)
        
        Args:
            universe: 종목 리스트 또는 인덱스명 (None이면 KOSPI200)
            start_date: 시작일
            end_date: 종료일
            include_technical: 기술적 지표 포함 여부
        """
        logger.info(f"Loading data: {start_date} ~ {end_date}")
        
        # universe가 None이거나 문자열(인덱스명)인 경우
        if universe is None:
            universe = "KOSPI200"
            logger.info(f"Using KOSPI200 universe")
        
        # 인덱스명인 경우 종목 리스트로 변환
        if isinstance(universe, str):
            universe_list = self.data_loader.get_universe_by_index(universe)
            self.current_universe = universe_list
        else:
            self.current_universe = universe
            universe_list = universe
        
        # 패널 데이터 로드
        self.current_data = self.data_loader.load_data(
            start_date=start_date,
            end_date=end_date,
            universe=universe_list,
            include_technical=include_technical
        )
        
        # 수익률 계산
        if 'close' in self.current_data:
            self.current_data['returns'] = self.current_data['close'].pct_change()
        
        logger.info(f"Data loaded: {len(self.current_data)} fields, "
                   f"{len(self.current_data['close'])} days, "
                   f"{len(self.current_data['close'].columns)} stocks")
        
        return self.current_data
    
    def mine_alpha(self,
                  idea: str,
                  num_seeds: int = 10,
                  enhancement_rounds: int = 20,
                  mode: str = "interactive",
                  top_n: int = 5) -> AlphaMiningResult:
        """
        알파 마이닝 (메인 워크플로우)
        
        Args:
            idea: 트레이딩 아이디어
            num_seeds: 초기 알파 개수
            enhancement_rounds: GP 진화 세대 수
            mode: interactive 또는 autonomous
            top_n: 상위 몇 개 반환
            
        Returns:
            AlphaMiningResult
        """
        logger.info(f"Starting alpha mining: mode={mode}")
        
        if self.current_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # ===== Stage 1: Ideation =====
        logger.info("Stage 1: Ideation")
        polished_idea = self.idea_polisher.polish(idea)
        
        logger.info(f"Polished idea: {polished_idea.refined_idea}")
        
        # ===== Stage 2: Implementation =====
        logger.info("Stage 2: Implementation")
        
        # 알파 생성
        alpha_expressions = self.quant_developer.generate_alphas(
            polished_idea.refined_idea,
            polished_idea.relevant_fields,
            num_variations=num_seeds
        )
        
        logger.info(f"Generated {len(alpha_expressions)} seed alphas")
        
        # GP 진화
        if enhancement_rounds > 0:
            logger.info(f"Running GP enhancement: {enhancement_rounds} rounds")
            
            # 적합도 함수 정의
            def fitness_func(expr: str) -> float:
                return self._evaluate_alpha_ic(expr)
            
            gp = AlphaGeneticProgramming(
                fitness_func=fitness_func,
                population_size=50,
                generations=enhancement_rounds,
                crossover_prob=0.6,
                mutation_prob=0.3
            )
            
            seed_exprs = [alpha.expr for alpha in alpha_expressions]
            evolved_population = gp.evolve(seed_exprs)
            
            # 상위 알파 선택
            top_individuals = evolved_population[:top_n * 2]
        else:
            # GP 없이 초기 알파만 사용
            top_individuals = [{'expression': alpha.expr} for alpha in alpha_expressions[:top_n * 2]]
        
        # ===== Stage 3: Review =====
        logger.info("Stage 3: Review")
        
        # 백테스트
        backtest_results = []
        for ind in top_individuals:
            expr = ind.expression if hasattr(ind, 'expression') else ind['expression']
            
            try:
                result = self._backtest_alpha(expr)
                backtest_results.append((expr, result))
            except Exception as e:
                logger.warning(f"Backtest failed for {expr[:50]}: {e}")
        
        # IC 기준 정렬
        backtest_results.sort(key=lambda x: x[1].ic, reverse=True)
        top_results = backtest_results[:top_n]
        
        # 분석
        analysis = self.analyst.compare_alphas(backtest_results, top_n=top_n)
        
        # 결과 정리
        best_result = top_results[0][1]
        
        result = AlphaMiningResult(
            top_alphas=top_results,
            best_ic=best_result.ic,
            best_sharpe=best_result.sharpe_ratio,
            analysis_report=analysis,
            iteration_history=[]
        )
        
        logger.info(f"Alpha mining complete. Best IC: {result.best_ic:.4f}")
        
        return result
    
    def chat(self, message: str) -> str:
        """
        Interactive mode: 대화형 인터페이스
        
        Args:
            message: 사용자 메시지
            
        Returns:
            응답
        """
        # 간단한 대화형 인터페이스
        if "알파" in message or "팩터" in message or "전략" in message:
            result = self.mine_alpha(message, num_seeds=5, enhancement_rounds=10, top_n=3)
            
            response = f"""
## 생성된 알파

**Top 3 Alphas:**

"""
            for i, (expr, bt_result) in enumerate(result.top_alphas, 1):
                response += f"""
{i}. **{expr[:80]}{'...' if len(expr) > 80 else ''}**
   - IC: {bt_result.ic:.4f}
   - Sharpe: {bt_result.sharpe_ratio:.2f}
   - Annual Return: {bt_result.annual_return:.2%}

"""
            
            response += f"\n{result.analysis_report}"
            
            return response
        else:
            return "트레이딩 아이디어를 말씀해주세요. 예: '모멘텀과 거래량을 결합한 전략을 만들어줘'"
    
    def _evaluate_alpha_ic(self, expression: str) -> float:
        """알파 표현식의 IC 계산"""
        try:
            # 실행 환경 준비
            from .mining.operators import AlphaOperators as ops
            
            close = self.current_data.get('close')
            open_price = self.current_data.get('open')
            high = self.current_data.get('high')
            low = self.current_data.get('low')
            volume = self.current_data.get('volume')
            vwap = self.current_data.get('vwap', close)
            returns = self.current_data.get('returns')
            
            # 표현식 평가
            alpha = eval(expression)
            
            # IC 계산
            forward_returns = returns.shift(-1)
            
            ic_list = []
            for date in alpha.index[:-1]:
                alpha_vals = alpha.loc[date]
                return_vals = forward_returns.loc[date]
                
                valid = (~alpha_vals.isna()) & (~return_vals.isna())
                
                if valid.sum() >= 5:
                    ic = alpha_vals[valid].corr(return_vals[valid], method='spearman')
                    if not np.isnan(ic):
                        ic_list.append(ic)
            
            return np.mean(ic_list) if ic_list else -1.0
            
        except Exception as e:
            logger.debug(f"IC evaluation error: {e}")
            return -1.0
    
    def _backtest_alpha(self, expression: str) -> BacktestResult:
        """알파 백테스트"""
        from .mining.operators import AlphaOperators as ops
        
        close = self.current_data.get('close')
        open_price = self.current_data.get('open')
        high = self.current_data.get('high')
        low = self.current_data.get('low')
        volume = self.current_data.get('volume')
        vwap = self.current_data.get('vwap', close)
        returns = self.current_data.get('returns')
        
        # 알파 계산
        alpha = eval(expression)
        
        # 백테스트 엔진
        engine = BacktestEngine(
            universe=self.current_universe,
            price_data=close,
            return_data=returns
        )
        
        # 실행
        result = engine.backtest(alpha, alpha_expr=expression)
        
        return result
    
    def refine(self, alpha_expr: str, feedback: str):
        """
        알파 개선 (interactive mode)
        
        Args:
            alpha_expr: 기존 알파 표현식
            feedback: 사용자 피드백
            
        Returns:
            개선된 AlphaExpression
        """
        current_alpha = AlphaExpression(
            expr=alpha_expr,
            description="",
            category="",
            complexity=0,
            operators_used=[]
        )
        
        refined = self.quant_developer.refine_alpha(current_alpha, feedback)
        
        logger.info(f"Refined alpha: {refined.expr[:100]}")
        
        return refined
    
    def backtest(self,
                alpha_expr: str,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None,
                universe: Optional[List[str]] = None) -> BacktestResult:
        """
        알파 백테스트 (독립 실행)
        
        Args:
            alpha_expr: 알파 표현식
            start_date: 시작일
            end_date: 종료일
            universe: 종목 유니버스
            
        Returns:
            BacktestResult
        """
        if self.current_data is None or universe is not None:
            self.load_data(universe, start_date or "2020-01-01", end_date or "2024-12-31")
        
        return self._backtest_alpha(alpha_expr)
