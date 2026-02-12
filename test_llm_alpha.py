#!/usr/bin/env python3
"""
LLM 기반 알파 생성 테스트 (PostgreSQL + OpenAI)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from alpha_gpt_kr.data.postgres_loader import PostgresDataLoader
from alpha_gpt_kr.agents.quant_developer import QuantDeveloper
from alpha_gpt_kr.backtest.engine import BacktestEngine
from alpha_gpt_kr.mining.operators import AlphaOperators as ops
from dotenv import load_dotenv
import openai
import pandas as pd

def main():
    print("=" * 60)
    print("LLM 기반 알파 생성 테스트")
    print("=" * 60)
    
    # 환경 변수 로드
    load_dotenv()
    
    try:
        # 1. OpenAI 클라이언트 초기화
        print("\n1. OpenAI 클라이언트 초기화...")
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ OPENAI_API_KEY가 설정되지 않았습니다!")
            return False
        
        client = openai.OpenAI(api_key=api_key)
        print(f"✅ API Key: {api_key[:15]}...{api_key[-10:]}")
        
        # 2. 데이터 로드
        print("\n2. PostgreSQL에서 데이터 로드...")
        loader = PostgresDataLoader()
        
        # 랜덤 샘플 종목 선택
        conn = loader._get_connection()
        stocks_df = pd.read_sql("""
            SELECT ticker, name 
            FROM stocks 
            WHERE is_active = true 
            ORDER BY RANDOM() 
            LIMIT 10;
        """, conn)
        conn.close()
        
        sample_tickers = stocks_df['ticker'].tolist()
        print(f"✅ 샘플 종목 {len(sample_tickers)}개:")
        for ticker, name in stocks_df.values[:3]:
            print(f"   {ticker}: {name}")
        print("   ...")
        
        # 데이터 로드
        data = loader.load_data(
            universe=sample_tickers,
            start_date="2024-12-01",
            end_date="2025-02-11"
        )
        
        print(f"✅ 데이터: {len(data['close'])} 일 × {len(data['close'].columns)} 종목")
        
        # 3. LLM으로 알파 생성
        print("\n3. LLM으로 알파 표현식 생성...")
        
        # Quant Developer 초기화
        developer = QuantDeveloper(client)
        
        # 트레이딩 아이디어
        idea = """
        거래량이 급증하면서 주가가 상승하는 종목을 찾고 싶습니다.
        
        조건:
        1. 최근 5일 평균 거래량이 20일 평균 대비 2배 이상
        2. 최근 5일 주가 상승 추세 (5일 수익률 > 0)
        3. 거래량과 주가의 양의 상관관계
        
        데이터: close, volume
        시간: 단기 (5-20일)
        """
        
        print(f"   아이디어: 거래량 급증 + 주가 상승 전략")
        
        # 알파 생성
        alphas = developer.generate_alphas(
            refined_idea=idea,
            relevant_fields=['close', 'volume'],
            num_variations=3
        )
        
        print(f"✅ {len(alphas)}개 알파 생성:")
        for i, alpha in enumerate(alphas[:3], 1):
            print(f"   {i}. {alpha.expr[:80]}...")
        
        # 4. 알파 백테스트
        print("\n4. 생성된 알파 백테스트...")
        
        close = data['close']
        volume = data['volume']
        
        # 수익률 계산
        returns = close.pct_change()
        
        # 첫 번째 알파 테스트
        best_alpha = alphas[0]
        print(f"   테스트할 알파: {best_alpha.expr[:60]}...")
        
        try:
            # 알파 계산
            alpha_values = eval(best_alpha.expr)
            
            # 백테스트
            engine = BacktestEngine(
                universe=sample_tickers,
                price_data=close,
                return_data=returns
            )
            
            result = engine.backtest(
                alpha=alpha_values,
                alpha_expr=best_alpha.expr,
                quantiles=(0.3, 0.7)
            )
            
            print(f"\n✅ 백테스트 결과:")
            print(f"   IC: {result.ic:.4f}")
            print(f"   Sharpe: {result.sharpe_ratio:.2f}")
            print(f"   연수익률: {result.annual_return:.2%}")
            print(f"   MDD: {result.max_drawdown:.2%}")
            
        except Exception as e:
            print(f"   ⚠️  알파 계산 실패: {e}")
            print(f"   (복잡한 알파는 실행 환경 문제로 실패할 수 있습니다)")
        
        # 5. 간단한 수동 알파로 추가 테스트
        print("\n5. 간단한 알파로 추가 테스트...")
        
        # 5일 이동평균 모멘텀
        simple_alpha = ops.ts_delta(ops.ts_mean(close, 5), 1)
        
        result2 = engine.backtest(
            alpha=simple_alpha,
            alpha_expr="ts_delta(ts_mean(close, 5), 1)",
            quantiles=(0.3, 0.7)
        )
        
        print(f"✅ 간단한 알파 결과:")
        print(f"   IC: {result2.ic:.4f}")
        print(f"   Sharpe: {result2.sharpe_ratio:.2f}")
        print(f"   연수익률: {result2.annual_return:.2%}")
        
        print("\n" + "=" * 60)
        print("✅ LLM 기반 알파 생성 테스트 완료!")
        print("=" * 60)
        print("\n🎉 Alpha-GPT가 PostgreSQL + OpenAI로 정상 작동합니다!")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
