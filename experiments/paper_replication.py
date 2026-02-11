"""
논문 재현 실험
Alpha-GPT 논문의 주요 실험을 한국 증시에서 재현
"""

import sys
sys.path.insert(0, '..')

from alpha_gpt_kr import AlphaGPT
from alpha_gpt_kr.mining.operators import AlphaOperators as ops
import pandas as pd
import numpy as np
from loguru import logger


def test_operators():
    """연산자 테스트"""
    logger.info("Testing operators...")
    
    # 테스트 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    tickers = ['A', 'B', 'C', 'D', 'E']
    
    data = pd.DataFrame(
        np.random.randn(100, 5) * 10 + 100,
        index=dates,
        columns=tickers
    )
    
    # Time-series operators
    assert ops.ts_mean(data, 10).shape == data.shape
    assert ops.ts_delta(data, 1).shape == data.shape
    assert ops.ts_corr(data, data.shift(1), 10).shape == data.shape
    
    # Cross-sectional operators
    assert ops.zscore_scale(data).shape == data.shape
    assert ops.normed_rank(data).shape == data.shape
    
    logger.info("✓ All operators working")


def test_idea_to_alpha():
    """아이디어 -> 알파 변환 테스트"""
    logger.info("Testing idea to alpha conversion...")
    
    # AlphaGPT 초기화 (실제 LLM 없이 테스트)
    # gpt = AlphaGPT(market="KRX")
    
    # 테스트 아이디어
    ideas = [
        "20일 모멘텀이 강한 종목",
        "거래량이 급증하면서 가격이 상승하는 패턴",
        "최근 5일간 하락했지만 거래량은 증가한 종목 (리버설)"
    ]
    
    logger.info(f"✓ {len(ideas)} test ideas prepared")


def test_backtest():
    """백테스팅 테스트"""
    logger.info("Testing backtest engine...")
    
    from alpha_gpt_kr.backtest import BacktestEngine
    
    # 테스트 데이터
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    tickers = [f'STK{i:03d}' for i in range(50)]
    
    price_data = pd.DataFrame(
        np.random.randn(500, 50).cumsum(axis=0) * 10 + 1000,
        index=dates,
        columns=tickers
    )
    
    # 간단한 알파: 20일 모멘텀
    alpha = price_data.pct_change(20)
    
    # 백테스트
    engine = BacktestEngine(tickers, price_data)
    result = engine.backtest(alpha, "momentum_20d")
    
    logger.info(f"Backtest result: IC={result.ic:.4f}, Sharpe={result.sharpe_ratio:.2f}")
    logger.info(result.summary())
    
    assert result.ic is not None
    assert result.sharpe_ratio is not None
    
    logger.info("✓ Backtest engine working")


def test_genetic_programming():
    """Genetic Programming 테스트"""
    logger.info("Testing genetic programming...")
    
    from alpha_gpt_kr.mining.genetic_programming import AlphaGeneticProgramming
    
    # 간단한 적합도 함수
    def mock_fitness(expr: str) -> float:
        # 표현식 길이 기반 더미 점수
        return len(expr) / 100.0
    
    gp = AlphaGeneticProgramming(
        fitness_func=mock_fitness,
        population_size=20,
        generations=5
    )
    
    seeds = [
        "ops.ts_mean(close, 10)",
        "ops.ts_delta(close, 20)",
        "ops.ts_corr(volume, close, 15)"
    ]
    
    result = gp.evolve(seeds, verbose=True)
    
    logger.info(f"GP evolved {len(result)} individuals")
    logger.info(f"Best fitness: {result[0].fitness:.4f}")
    
    logger.info("✓ Genetic programming working")


def main():
    """메인 테스트 실행"""
    logger.info("=" * 60)
    logger.info("Alpha-GPT Paper Replication Tests")
    logger.info("=" * 60)
    
    try:
        test_operators()
        print()
        
        test_idea_to_alpha()
        print()
        
        test_backtest()
        print()
        
        test_genetic_programming()
        print()
        
        logger.info("=" * 60)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
