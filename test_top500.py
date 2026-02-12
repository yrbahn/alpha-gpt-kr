#!/usr/bin/env python3
"""
시가총액 상위 500개 종목으로 알파 테스트
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
    print("시가총액 상위 500개 종목 알파 테스트")
    print("=" * 60)
    
    load_dotenv()
    
    try:
        # 1. 시가총액 상위 500개 종목 조회
        print("\n1. 시가총액 상위 500개 종목 조회...")
        loader = PostgresDataLoader()
        
        conn = loader._get_connection()
        stocks_df = pd.read_sql("""
            SELECT ticker, name, market_cap
            FROM stocks 
            WHERE is_active = true 
                AND market_cap IS NOT NULL
            ORDER BY market_cap DESC 
            LIMIT 500;
        """, conn)
        conn.close()
        
        top500_tickers = stocks_df['ticker'].tolist()
        print(f"✅ 상위 500개 종목 조회 완료")
        print(f"   1위: {stocks_df.iloc[0]['name']} (시총: {stocks_df.iloc[0]['market_cap']:,.0f})")
        print(f"   2위: {stocks_df.iloc[1]['name']} (시총: {stocks_df.iloc[1]['market_cap']:,.0f})")
        print(f"   3위: {stocks_df.iloc[2]['name']} (시총: {stocks_df.iloc[2]['market_cap']:,.0f})")
        print(f"   ...")
        print(f"   500위: {stocks_df.iloc[499]['name']} (시총: {stocks_df.iloc[499]['market_cap']:,.0f})")
        
        # 2. 데이터 로드 (최근 3개월)
        print("\n2. 데이터 로드 (최근 3개월, 500개 종목)...")
        print("   (로딩 중... 1-2분 소요)")
        
        data = loader.load_data(
            universe=top500_tickers,
            start_date="2024-11-01",
            end_date="2025-02-11",
            include_technical=False
        )
        
        print(f"✅ 데이터 로드 완료:")
        print(f"   기간: {data['close'].index[0].date()} ~ {data['close'].index[-1].date()}")
        print(f"   일수: {len(data['close'])} 일")
        print(f"   종목: {len(data['close'].columns)} 개")
        
        # 3. OpenAI로 알파 생성
        print("\n3. LLM으로 알파 생성...")
        api_key = os.getenv('OPENAI_API_KEY')
        client = openai.OpenAI(api_key=api_key)
        developer = QuantDeveloper(client)
        
        idea = """
        거래량이 급증하면서 주가가 상승하는 종목을 찾습니다.
        
        조건:
        1. 최근 5일 평균 거래량이 20일 평균 대비 1.5배 이상
        2. 최근 5일 주가 수익률이 양수
        3. 거래량과 주가의 상관관계가 양수
        
        시간: 단기 (5-20일)
        """
        
        alphas = developer.generate_alphas(
            refined_idea=idea,
            relevant_fields=['close', 'volume'],
            num_variations=3
        )
        
        print(f"✅ {len(alphas)}개 알파 생성:")
        for i, alpha in enumerate(alphas, 1):
            print(f"   {i}. {alpha.description}")
            print(f"      {alpha.expr[:100]}...")
        
        # 4. 백테스트
        print("\n4. 생성된 알파 백테스트 (상위 500개 종목)...")
        
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()
        
        best_results = []
        
        for i, alpha in enumerate(alphas, 1):
            print(f"\n   [{i}/{len(alphas)}] 백테스트 중...")
            print(f"   알파: {alpha.description}")
            
            try:
                # 알파 계산
                alpha_values = eval(alpha.expr)
                
                # 백테스트
                engine = BacktestEngine(
                    universe=top500_tickers,
                    price_data=close,
                    return_data=returns
                )
                
                result = engine.backtest(
                    alpha=alpha_values,
                    alpha_expr=alpha.expr,
                    quantiles=(0.2, 0.8),  # 상위 20%, 하위 20%
                    rebalance_freq='1D'
                )
                
                print(f"   ✅ IC: {result.ic:.4f}, Sharpe: {result.sharpe_ratio:.2f}, 연수익: {result.annual_return:.2%}")
                
                best_results.append({
                    'alpha': alpha,
                    'result': result
                })
                
            except Exception as e:
                print(f"   ⚠️  실패: {str(e)[:100]}")
        
        # 5. 최고 알파 선택
        if best_results:
            print("\n" + "=" * 60)
            print("✅ 백테스트 완료 - 최고 알파:")
            print("=" * 60)
            
            best = max(best_results, key=lambda x: x['result'].ic)
            result = best['result']
            
            print(f"\n알파: {best['alpha'].description}")
            print(f"표현식: {best['alpha'].expr[:150]}...")
            print(f"\n성과:")
            print(f"  IC (Information Coefficient): {result.ic:.4f}")
            print(f"  IC 표준편차: {result.ic_std:.4f}")
            print(f"  IR (Information Ratio): {result.ir:.2f}")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  연평균 수익률: {result.annual_return:.2%}")
            print(f"  누적 수익률: {result.total_return:.2%}")
            print(f"  최대 낙폭 (MDD): {result.max_drawdown:.2%}")
            print(f"  회전율: {result.turnover:.2%}")
            print(f"  승률: {result.win_rate:.2%}")
            
            # 요약 판단
            print(f"\n평가:")
            if result.ic > 0.05:
                print("  🎉 우수한 알파! (IC > 0.05)")
            elif result.ic > 0.02:
                print("  ✅ 괜찮은 알파 (IC > 0.02)")
            elif result.ic > 0:
                print("  ⚠️  약한 알파 (IC > 0)")
            else:
                print("  ❌ 개선 필요 (IC < 0)")
            
            if result.sharpe_ratio > 1.5:
                print("  🎉 훌륭한 샤프비율! (> 1.5)")
            elif result.sharpe_ratio > 1.0:
                print("  ✅ 좋은 샤프비율 (> 1.0)")
            elif result.sharpe_ratio > 0.5:
                print("  ⚠️  보통 샤프비율 (> 0.5)")
        
        print("\n" + "=" * 60)
        print("✅ 시가총액 상위 500개 종목 테스트 완료!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
