"""
Simple Example: Alpha-GPT 기본 사용법
"""

import sys
sys.path.insert(0, '..')

from alpha_gpt_kr import AlphaGPT
from loguru import logger

def main():
    """간단한 예제"""
    
    logger.info("Alpha-GPT Simple Example")
    logger.info("=" * 60)
    
    # 1. AlphaGPT 초기화
    print("\n[1] Initializing Alpha-GPT...")
    
    gpt = AlphaGPT(
        market="KRX",
        llm_provider="openai",  # or "anthropic"
        model="gpt-4-turbo-preview"
    )
    
    # 2. 데이터 로드 (소규모 테스트)
    print("\n[2] Loading test data...")
    
    # 소수의 종목으로 테스트
    test_universe = ["005930", "000660", "035420", "051910", "035720"]  # 삼성전자, SK하이닉스 등
    
    try:
        gpt.load_data(
            universe=test_universe,
            start_date="2023-01-01",
            end_date="2024-01-01"
        )
        print("✓ Data loaded")
    except Exception as e:
        print(f"⚠️  Failed to load real data: {e}")
        print("   Continuing with mock mode...")
        return
    
    # 3. 트레이딩 아이디어 입력
    print("\n[3] Trading idea:")
    
    idea = """
    최근 5일간 거래량이 증가하면서 주가도 상승하는 
    모멘텀이 강한 종목을 찾고 싶습니다.
    """
    
    print(idea)
    
    # 4. 알파 마이닝
    print("\n[4] Mining alphas...")
    print("   (This may take a few minutes...)")
    
    try:
        result = gpt.mine_alpha(
            idea=idea,
            num_seeds=3,
            enhancement_rounds=5,
            top_n=2
        )
        
        # 5. 결과 출력
        print("\n[5] Results:")
        print("=" * 60)
        
        for i, (expr, bt_result) in enumerate(result.top_alphas, 1):
            print(f"\n알파 #{i}:")
            print(f"표현식: {expr}")
            print(f"IC: {bt_result.ic:.4f}")
            print(f"Sharpe Ratio: {bt_result.sharpe_ratio:.2f}")
            print(f"연간 수익률: {bt_result.annual_return:.2%}")
            print(f"최대 낙폭: {bt_result.max_drawdown:.2%}")
        
        print("\n" + "=" * 60)
        print("✅ Complete!")
        
    except Exception as e:
        print(f"\n❌ Error during mining: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
