#!/usr/bin/env python3
"""
LLM 생성 알파의 2년 장기 백테스트
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
    print("LLM 알파 장기 백테스트 (2023-2025)")
    print("=" * 60)
    
    load_dotenv()
    
    try:
        # 1. 종목
        print("\n1. 시총 상위 500개 종목...")
        loader = PostgresDataLoader()
        conn = loader._get_connection()
        stocks_df = pd.read_sql("""
            SELECT ticker, name FROM stocks 
            WHERE is_active = true AND market_cap IS NOT NULL
            ORDER BY market_cap DESC LIMIT 500;
        """, conn)
        conn.close()
        
        tickers = stocks_df['ticker'].tolist()
        print(f"✅ {len(tickers)}개 종목")
        
        # 2. 2년 데이터
        print("\n2. 2년 데이터 로드...")
        print("   로딩 중... (1-2분)")
        
        data = loader.load_data(
            universe=tickers,
            start_date="2023-01-01",
            end_date="2025-02-11"
        )
        
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()
        
        print(f"✅ {len(close)}일, {len(close.columns)}개 종목")
        
        # 3. LLM 알파 생성
        print("\n3. LLM으로 알파 생성...")
        api_key = os.getenv('OPENAI_API_KEY')
        client = openai.OpenAI(api_key=api_key)
        developer = QuantDeveloper(client)
        
        idea = """
        2023-2025 한국 증시 특성을 고려한 알파 전략
        
        관찰:
        - 단순 모멘텀은 역효과 (IC < 0)
        - 시장이 변동성이 컸음
        
        전략:
        - 리버설 (역행) 전략 시도
        - 변동성 조정 필요
        - 거래량 급감 후 반등 포착
        - 상대 강도 활용
        
        데이터: close, volume
        """
        
        alphas = developer.generate_alphas(
            refined_idea=idea,
            relevant_fields=['close', 'volume'],
            num_variations=3
        )
        
        print(f"✅ {len(alphas)}개 알파 생성")
        
        # 4. 백테스트
        print("\n4. 장기 백테스트...")
        
        results = []
        
        for i, alpha in enumerate(alphas, 1):
            print(f"\n   [{i}/{len(alphas)}] {alpha.description[:60]}...")
            
            try:
                alpha_values = eval(alpha.expr)
                
                engine = BacktestEngine(
                    universe=tickers,
                    price_data=close,
                    return_data=returns
                )
                
                result = engine.backtest(
                    alpha=alpha_values,
                    alpha_expr=alpha.description,
                    quantiles=(0.2, 0.8)
                )
                
                print(f"        IC: {result.ic:>7.4f} | Sharpe: {result.sharpe_ratio:>6.2f} | 연수익: {result.annual_return:>7.2%}")
                
                results.append({'alpha': alpha, 'result': result})
                
            except Exception as e:
                print(f"        ⚠️  실패: {str(e)[:60]}")
        
        # 5. 최고 알파
        if results:
            best = max(results, key=lambda x: x['result'].ic)
            r = best['result']
            a = best['alpha']
            
            print("\n" + "=" * 60)
            print("🥇 최고 성과 알파 (2년)")
            print("=" * 60)
            
            print(f"\n전략: {a.description}")
            print(f"표현식: {a.expr[:150]}...")
            
            print(f"\n성과:")
            print(f"  IC:              {r.ic:>8.4f}")
            print(f"  Sharpe:          {r.sharpe_ratio:>8.2f}")
            print(f"  연수익:          {r.annual_return:>8.2%}")
            print(f"  2년 누적:        {r.total_return:>8.2%}")
            print(f"  MDD:             {r.max_drawdown:>8.2%}")
            
            print(f"\n평가:")
            if r.ic > 0.03:
                print("  ✅ IC 우수!")
            elif r.ic > 0.01:
                print("  ✅ IC 양호")
            elif r.ic > 0:
                print("  ⚠️  IC 약함")
            else:
                print("  ❌ IC 음수")
            
            if r.sharpe_ratio > 1.0:
                print("  ✅ Sharpe 우수!")
            elif r.sharpe_ratio > 0.5:
                print("  ⚠️  Sharpe 보통")
            else:
                print("  ❌ Sharpe 약함")
        
        print("\n✅ 완료!")
        return True
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
