#!/usr/bin/env python3
"""
Genetic Programming으로 알파 진화
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from alpha_gpt_kr.data.postgres_loader import PostgresDataLoader
from alpha_gpt_kr.backtest.engine import BacktestEngine
from alpha_gpt_kr.mining.genetic_programming import AlphaGeneticProgramming
from alpha_gpt_kr.mining.operators import AlphaOperators as ops
import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("Genetic Programming 알파 진화 (2023-2025)")
    print("=" * 60)
    
    try:
        # 1. 데이터 준비
        print("\n1. 데이터 준비...")
        loader = PostgresDataLoader()
        
        conn = loader._get_connection()
        stocks_df = pd.read_sql("""
            SELECT ticker FROM stocks 
            WHERE is_active = true AND market_cap IS NOT NULL
            ORDER BY market_cap DESC LIMIT 500;
        """, conn)
        conn.close()
        
        tickers = stocks_df['ticker'].tolist()
        print(f"✅ {len(tickers)}개 종목")
        
        # 2년 데이터 로드
        print("   데이터 로딩... (1-2분)")
        data = loader.load_data(
            universe=tickers,
            start_date="2023-01-01",
            end_date="2025-02-11"
        )
        
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()
        
        print(f"✅ {len(close)}일, {len(close.columns)}개 종목")
        
        # 2. 초기 시드 알파 생성
        print("\n2. 초기 시드 알파 생성...")
        
        seed_expressions = [
            # 모멘텀 계열
            "ops.ts_delta(close, 5)",
            "ops.ts_delta(close, 10)",
            "ops.ts_delta(close, 20)",
            
            # 이동평균 계열
            "ops.ts_mean(close, 5)",
            "ops.ts_mean(close, 10)",
            "ops.ts_mean(close, 20)",
            
            # 변동성 계열
            "ops.ts_std(close, 10)",
            "ops.ts_std(close, 20)",
            
            # 거래량 계열
            "ops.ts_mean(volume, 5)",
            "ops.ts_mean(volume, 10)",
            "ops.ts_delta(volume, 5)",
            
            # 상관관계
            "ops.ts_corr(close, volume, 10)",
            "ops.ts_corr(close, volume, 20)",
            
            # 순위
            "ops.ts_rank(close, 5)",
            "ops.ts_rank(close, 10)",
        ]
        
        print(f"✅ {len(seed_expressions)}개 시드 알파")
        
        # 3. 적합도 함수 정의
        print("\n3. 적합도 함수 설정...")
        
        def fitness_function(expression: str) -> float:
            """알파의 IC를 반환"""
            try:
                # 표현식 실행
                alpha_values = eval(expression)
                
                # 백테스트
                engine = BacktestEngine(
                    universe=tickers,
                    price_data=close,
                    return_data=returns
                )
                
                result = engine.backtest(
                    alpha=alpha_values,
                    alpha_expr=expression[:50],
                    quantiles=(0.2, 0.8)
                )
                
                # IC 반환
                return result.ic
                
            except Exception as e:
                # 실패한 알파는 낮은 점수
                return -1.0
        
        print("✅ 적합도 함수: IC (Information Coefficient)")
        
        # 4. Genetic Programming 진화
        print("\n4. Genetic Programming 진화 시작...")
        print("   설정:")
        print("   - 개체군 크기: 50")
        print("   - 세대 수: 30")
        print("   - 교배 확률: 60%")
        print("   - 변이 확률: 30%")
        print("   - 엘리트 보존: 5개")
        print()
        
        gp = AlphaGeneticProgramming(
            fitness_func=fitness_function,
            population_size=50,
            generations=30,
            crossover_prob=0.6,
            mutation_prob=0.3,
            tournament_size=3,
            elitism=5
        )
        
        # 진화 실행
        evolved_population = gp.evolve(
            seed_expressions=seed_expressions,
            verbose=True
        )
        
        # 5. 최고 알파 분석
        print("\n" + "=" * 60)
        print("🏆 진화 완료 - 최고 알파들")
        print("=" * 60)
        
        # 상위 5개
        for i, individual in enumerate(evolved_population[:5], 1):
            print(f"\n{i}위. IC: {individual.fitness:.4f}")
            print(f"     표현식: {individual.expression[:100]}")
            if len(individual.expression) > 100:
                print(f"              {individual.expression[100:200]}")
            print(f"     복잡도: {individual.complexity}")
        
        # 최고 알파 상세 백테스트
        best = evolved_population[0]
        
        print("\n" + "=" * 60)
        print("🥇 최고 알파 상세 분석")
        print("=" * 60)
        
        print(f"\n표현식:")
        print(f"  {best.expression}")
        
        print(f"\n진화 결과:")
        print(f"  적합도 (IC): {best.fitness:.4f}")
        print(f"  복잡도: {best.complexity}")
        
        # 상세 백테스트
        print("\n상세 백테스트 실행 중...")
        alpha_values = eval(best.expression)
        
        engine = BacktestEngine(
            universe=tickers,
            price_data=close,
            return_data=returns
        )
        
        result = engine.backtest(
            alpha=alpha_values,
            alpha_expr="Best Evolved Alpha",
            quantiles=(0.2, 0.8)
        )
        
        print("\n📈 2년 성과 (2023-2025):")
        print(f"  IC:                {result.ic:>8.4f}")
        print(f"  IC 표준편차:       {result.ic_std:>8.4f}")
        print(f"  IR:                {result.ir:>8.2f}")
        print(f"  Sharpe Ratio:      {result.sharpe_ratio:>8.2f}")
        print(f"  연평균 수익률:     {result.annual_return:>8.2%}")
        print(f"  2년 누적 수익률:   {result.total_return:>8.2%}")
        print(f"  최대 낙폭 (MDD):   {result.max_drawdown:>8.2%}")
        print(f"  평균 회전율:       {result.turnover:>8.2%}")
        print(f"  승률:              {result.win_rate:>8.2%}")
        
        print(f"\n💰 1억원 투자 시뮬레이션:")
        final_capital = 100_000_000 * (1 + result.total_return)
        profit = final_capital - 100_000_000
        print(f"  초기 자본: 100,000,000원")
        print(f"  최종 자본: {final_capital:>13,.0f}원")
        print(f"  순이익:    {profit:>13,.0f}원")
        
        print(f"\n📊 평가:")
        
        # IC 평가
        if result.ic > 0.05:
            print(f"  🎉 IC {result.ic:.4f}: 우수! (> 0.05)")
        elif result.ic > 0.03:
            print(f"  ✅ IC {result.ic:.4f}: 양호 (> 0.03)")
        elif result.ic > 0.01:
            print(f"  ⚠️  IC {result.ic:.4f}: 보통 (> 0.01)")
        elif result.ic > 0:
            print(f"  ⚠️  IC {result.ic:.4f}: 약함 (> 0)")
        else:
            print(f"  ❌ IC {result.ic:.4f}: 음수")
        
        # Sharpe 평가
        if result.sharpe_ratio > 2.0:
            print(f"  🎉 Sharpe {result.sharpe_ratio:.2f}: 탁월 (> 2.0)")
        elif result.sharpe_ratio > 1.5:
            print(f"  🎉 Sharpe {result.sharpe_ratio:.2f}: 우수 (> 1.5)")
        elif result.sharpe_ratio > 1.0:
            print(f"  ✅ Sharpe {result.sharpe_ratio:.2f}: 양호 (> 1.0)")
        elif result.sharpe_ratio > 0:
            print(f"  ⚠️  Sharpe {result.sharpe_ratio:.2f}: 보통")
        else:
            print(f"  ❌ Sharpe {result.sharpe_ratio:.2f}: 음수")
        
        # 비교
        print("\n📊 개선도:")
        initial_best_ic = max([gp.fitness_func(expr) for expr in seed_expressions[:3]])
        improvement = ((result.ic - initial_best_ic) / abs(initial_best_ic) * 100)
        print(f"  초기 최고 IC: {initial_best_ic:.4f}")
        print(f"  진화 후 IC:   {result.ic:.4f}")
        print(f"  개선율:       {improvement:+.1f}%")
        
        print("\n" + "=" * 60)
        print("✅ Genetic Programming 진화 완료!")
        print("=" * 60)
        
        print("\n💡 활용 방법:")
        print("  1. 최고 알파를 실전 모의투자에 적용")
        print("  2. 상위 5개 알파를 앙상블로 결합")
        print("  3. 다른 시장 구간에서 재검증")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
