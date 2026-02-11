"""
한국 증시 벤치마크 실험
실제 KRX 데이터로 Alpha-GPT 성능 검증
"""

import sys
sys.path.insert(0, '..')

import argparse
from datetime import datetime
from loguru import logger

from alpha_gpt_kr import AlphaGPT


def run_benchmark(start_date: str = "2020-01-01",
                 end_date: str = "2024-12-31",
                 llm_provider: str = "openai",
                 model: str = "gpt-4-turbo-preview"):
    """
    벤치마크 실행
    
    Args:
        start_date: 백테스트 시작일
        end_date: 백테스트 종료일
        llm_provider: LLM 제공자
        model: 모델명
    """
    logger.info("=" * 60)
    logger.info("Alpha-GPT Korean Market Benchmark")
    logger.info(f"Period: {start_date} ~ {end_date}")
    logger.info(f"LLM: {llm_provider}/{model}")
    logger.info("=" * 60)
    
    # AlphaGPT 초기화
    gpt = AlphaGPT(
        market="KRX",
        llm_provider=llm_provider,
        model=model
    )
    
    # 데이터 로드
    logger.info("\n[1] Loading Korean market data...")
    try:
        gpt.load_data(
            universe=None,  # KOSPI200
            start_date=start_date,
            end_date=end_date
        )
        logger.info("✓ Data loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.warning("Please make sure you have installed FinanceDataReader:")
        logger.warning("  pip install FinanceDataReader")
        return
    
    # 테스트 아이디어들
    test_ideas = [
        """
        거래량이 20일 평균 대비 2배 이상 증가하면서 
        가격이 전일 대비 2% 이상 상승한 종목을 포착하고 싶습니다.
        """,
        
        """
        최근 5일간 지속적으로 하락했지만 
        거래량은 증가하는 반전 신호를 찾고 싶습니다.
        """,
        
        """
        주가가 20일 이동평균선을 상향 돌파하는 
        골든크로스 시그널을 구현해주세요.
        """
    ]
    
    results = []
    
    for i, idea in enumerate(test_ideas, 1):
        logger.info(f"\n[{i+1}] Mining alpha for idea {i}...")
        logger.info(f"Idea: {idea.strip()[:100]}...")
        
        try:
            result = gpt.mine_alpha(
                idea=idea,
                num_seeds=5,
                enhancement_rounds=10,
                top_n=3
            )
            
            results.append((idea, result))
            
            logger.info(f"✓ Best IC: {result.best_ic:.4f}")
            logger.info(f"✓ Best Sharpe: {result.best_sharpe:.2f}")
            
            # Top 3 알파 출력
            print("\nTop 3 Alphas:")
            for j, (expr, bt_result) in enumerate(result.top_alphas, 1):
                print(f"\n{j}. {expr[:80]}{'...' if len(expr) > 80 else ''}")
                print(f"   IC: {bt_result.ic:.4f} | Sharpe: {bt_result.sharpe_ratio:.2f} | "
                      f"Return: {bt_result.annual_return:.2%}")
            
        except Exception as e:
            logger.error(f"Failed to mine alpha: {e}")
            import traceback
            traceback.print_exc()
    
    # 최종 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)
    
    if results:
        best_overall = max(results, key=lambda x: x[1].best_ic)
        logger.info(f"\nBest Overall IC: {best_overall[1].best_ic:.4f}")
        logger.info(f"Best Overall Sharpe: {best_overall[1].best_sharpe:.2f}")
        
        logger.info(f"\n✅ Benchmark completed successfully!")
        logger.info(f"Total alphas generated: {sum(len(r[1].top_alphas) for r in results)}")
    else:
        logger.warning("No results generated")


def main():
    parser = argparse.ArgumentParser(description="Alpha-GPT Korean Market Benchmark")
    parser.add_argument("--start", default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--llm", default="openai", choices=["openai", "anthropic"],
                       help="LLM provider")
    parser.add_argument("--model", default="gpt-4-turbo-preview", help="Model name")
    
    args = parser.parse_args()
    
    run_benchmark(
        start_date=args.start,
        end_date=args.end,
        llm_provider=args.llm,
        model=args.model
    )


if __name__ == "__main__":
    main()
