"""
Genetic Programming for Alpha Evolution
논문의 Search Enhancement 구현
"""

import numpy as np
import pandas as pd
from typing import List, Callable, Tuple, Optional
from dataclasses import dataclass
import random
from copy import deepcopy
from loguru import logger


@dataclass
class Individual:
    """GP 개체 (알파)"""
    expression: str  # 알파 표현식
    fitness: float = -np.inf  # 적합도 (IC)
    complexity: int = 0  # 복잡도
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.4f}, expr={self.expression[:50]}...)"


class AlphaGeneticProgramming:
    """
    알파 Genetic Programming
    
    논문의 algorithmic search enhancement 구현:
    - Crossover: 두 알파의 부분 표현식 교환
    - Mutation: 연산자나 파라미터 변경
    - Selection: IC 기반 선택
    """
    
    def __init__(self,
                 fitness_func: Callable,
                 population_size: int = 100,
                 generations: int = 20,
                 crossover_prob: float = 0.6,
                 mutation_prob: float = 0.3,
                 tournament_size: int = 3,
                 elitism: int = 5):
        """
        Args:
            fitness_func: 적합도 함수 (expression -> IC)
            population_size: 개체군 크기
            generations: 세대 수
            crossover_prob: 교배 확률
            mutation_prob: 변이 확률
            tournament_size: 토너먼트 선택 크기
            elitism: 엘리트 보존 개수
        """
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.elitism = elitism
        
        logger.info(f"GP initialized: pop={population_size}, gen={generations}")
    
    def evolve(self, 
              seed_expressions: List[str],
              verbose: bool = True) -> List[Individual]:
        """
        진화 알고리즘 실행
        
        Args:
            seed_expressions: 초기 알파 표현식들
            verbose: 진행 상황 출력
            
        Returns:
            최종 개체군 (적합도순 정렬)
        """
        logger.info(f"Starting evolution with {len(seed_expressions)} seeds")
        
        # 초기 개체군 생성
        population = self._initialize_population(seed_expressions)
        
        # 세대 진화
        best_fitness_history = []
        
        for generation in range(self.generations):
            # 적합도 평가
            population = self._evaluate_population(population)
            
            # 통계
            best = max(population, key=lambda x: x.fitness)
            avg_fitness = np.mean([ind.fitness for ind in population if ind.fitness > -np.inf])
            best_fitness_history.append(best.fitness)
            
            if verbose:
                logger.info(f"Gen {generation+1}/{self.generations}: "
                          f"Best IC={best.fitness:.4f}, Avg IC={avg_fitness:.4f}")
            
            # 조기 종료 체크 (수렴)
            if len(best_fitness_history) >= 5:
                recent_improvement = best_fitness_history[-1] - best_fitness_history[-5]
                if recent_improvement < 0.0001:
                    logger.info("Converged. Stopping early.")
                    break
            
            # 다음 세대 생성
            population = self._create_next_generation(population)
        
        # 최종 평가 및 정렬
        population = self._evaluate_population(population)
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        logger.info(f"Evolution complete. Best IC: {population[0].fitness:.4f}")
        return population
    
    def _initialize_population(self, seeds: List[str]) -> List[Individual]:
        """초기 개체군 생성"""
        population = []
        
        # Seed 개체 추가
        for expr in seeds:
            population.append(Individual(expression=expr))
        
        # 랜덤 변이로 다양성 증대
        while len(population) < self.population_size:
            base = random.choice(seeds)
            mutated = self._mutate(Individual(expression=base))
            population.append(mutated)
        
        return population[:self.population_size]
    
    def _evaluate_population(self, population: List[Individual]) -> List[Individual]:
        """개체군 적합도 평가"""
        for ind in population:
            if ind.fitness == -np.inf:  # 아직 평가 안 됨
                try:
                    ind.fitness = self.fitness_func(ind.expression)
                    ind.complexity = self._calculate_complexity(ind.expression)
                except Exception as e:
                    logger.debug(f"Fitness evaluation failed: {e}")
                    ind.fitness = -1.0  # 실패한 개체는 낮은 점수
        
        return population
    
    def _create_next_generation(self, population: List[Individual]) -> List[Individual]:
        """다음 세대 생성"""
        next_gen = []
        
        # 엘리트 보존
        population.sort(key=lambda x: x.fitness, reverse=True)
        next_gen.extend(deepcopy(population[:self.elitism]))
        
        # 나머지는 교배/변이로 생성
        while len(next_gen) < self.population_size:
            if random.random() < self.crossover_prob:
                # 교배
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                child = self._crossover(parent1, parent2)
            else:
                # 복제
                parent = self._tournament_selection(population)
                child = deepcopy(parent)
            
            # 변이
            if random.random() < self.mutation_prob:
                child = self._mutate(child)
            
            next_gen.append(child)
        
        return next_gen[:self.population_size]
    
    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """토너먼트 선택"""
        tournament = random.sample(population, self.tournament_size)
        winner = max(tournament, key=lambda x: x.fitness)
        return winner
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """
        교배: 두 알파의 부분 표현식 교환
        
        예: parent1 = "ops.ts_mean(close, 10)"
           parent2 = "ops.ts_std(volume, 20)"
           child = "ops.ts_mean(volume, 20)"  # 연산자와 피연산자 조합
        """
        expr1 = parent1.expression
        expr2 = parent2.expression
        
        # 간단한 교배: 랜덤 서브트리 교환
        # 실제로는 AST 파싱이 필요하지만, 여기서는 단순화
        
        # 연산자 추출
        ops1 = self._extract_operators(expr1)
        ops2 = self._extract_operators(expr2)
        
        if ops1 and ops2:
            # 일부 연산자를 교환
            child_expr = expr1
            for op in random.sample(ops2, min(2, len(ops2))):
                if random.random() < 0.5:
                    child_expr = child_expr.replace(ops1[0], op, 1)
        else:
            child_expr = random.choice([expr1, expr2])
        
        return Individual(expression=child_expr)
    
    def _mutate(self, individual: Individual) -> Individual:
        """
        변이: 연산자나 파라미터 변경
        
        가능한 변이:
        - 윈도우 크기 변경 (10 -> 20)
        - 연산자 교체 (ts_mean -> ts_ema)
        - 피연산자 교체 (close -> volume)
        """
        expr = individual.expression
        
        mutation_type = random.choice(['window', 'operator', 'operand'])
        
        if mutation_type == 'window':
            # 윈도우 크기 변경
            expr = self._mutate_window(expr)
        elif mutation_type == 'operator':
            # 연산자 교체
            expr = self._mutate_operator(expr)
        else:
            # 피연산자 교체
            expr = self._mutate_operand(expr)
        
        return Individual(expression=expr)
    
    def _mutate_window(self, expr: str) -> str:
        """윈도우 파라미터 변경"""
        import re
        
        # 숫자 파라미터 찾기
        numbers = re.findall(r'\b(\d+)\b', expr)
        
        if numbers:
            old_num = random.choice(numbers)
            # ±50% 범위로 변경
            new_num = max(1, int(int(old_num) * random.uniform(0.5, 1.5)))
            expr = expr.replace(f', {old_num})', f', {new_num})', 1)
        
        return expr
    
    def _mutate_operator(self, expr: str) -> str:
        """연산자 교체"""
        ts_operators = [
            'ts_mean', 'ts_std', 'ts_ema', 'ts_rank',
            'ts_delta', 'ts_corr', 'ts_min', 'ts_max'
        ]
        
        current_ops = self._extract_operators(expr)
        
        if current_ops:
            old_op = random.choice(current_ops)
            new_op = random.choice(ts_operators)
            expr = expr.replace(old_op, new_op, 1)
        
        return expr
    
    def _mutate_operand(self, expr: str) -> str:
        """피연산자 교체"""
        operands = ['close', 'open', 'high', 'low', 'volume', 'vwap']
        
        for old_operand in operands:
            if old_operand in expr:
                new_operand = random.choice(operands)
                expr = expr.replace(old_operand, new_operand, 1)
                break
        
        return expr
    
    def _extract_operators(self, expr: str) -> List[str]:
        """표현식에서 연산자 추출"""
        import re
        
        pattern = r'ops\.(\w+)\('
        matches = re.findall(pattern, expr)
        return matches
    
    def _calculate_complexity(self, expr: str) -> int:
        """표현식 복잡도 계산"""
        # 연산자 개수 + 괄호 깊이
        operators = self._extract_operators(expr)
        depth = max(expr[:i].count('(') - expr[:i].count(')') 
                   for i in range(len(expr)))
        
        return len(operators) + depth


def simple_fitness_function(expression: str, 
                            data: pd.DataFrame,
                            returns: pd.DataFrame) -> float:
    """
    간단한 적합도 함수 예시
    
    Args:
        expression: 알파 표현식
        data: 가격 데이터
        returns: 수익률 데이터
        
    Returns:
        IC (Information Coefficient)
    """
    try:
        # 표현식 실행 (안전한 환경에서)
        from ..mining.operators import AlphaOperators as ops
        
        # 필요한 변수들
        close = data['close']
        open_price = data['open']
        high = data['high']
        low = data['low']
        volume = data['volume']
        vwap = data.get('vwap', close)
        
        # 표현식 평가
        alpha = eval(expression)
        
        # IC 계산
        forward_returns = returns.shift(-1)
        
        ic_series = []
        for date in alpha.index[:-1]:
            alpha_vals = alpha.loc[date]
            return_vals = forward_returns.loc[date]
            
            valid = (~alpha_vals.isna()) & (~return_vals.isna())
            
            if valid.sum() >= 5:
                ic = alpha_vals[valid].corr(return_vals[valid], method='spearman')
                ic_series.append(ic)
        
        if ic_series:
            return np.mean(ic_series)
        else:
            return -1.0
            
    except Exception as e:
        logger.debug(f"Fitness evaluation error: {e}")
        return -1.0
