#!/usr/bin/env python3
"""
장기 백테스트: 시가총액 상위 500개 종목 (2-3년)
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
    print("장기 백테스트: 시가총액 상위 500개 (2023~2025)")
    print("=" * 60)
    
    load_dotenv()
    
    try:
        # 1. 시가총액 상위 500개 종목 조회
        print("\n1. 시가총액 상위 500개 종목 조회...")
        loader = PostgresDataLoader()
        
        conn = loader._get_connection()
        stocks_df = pd.read_sql("""
            SELECT ticker, name, market_cap, sector
            FROM stocks 
            WHERE is_active = true 
                AND market_cap IS NOT NULL
            ORDER BY market_cap DESC 
            LIMIT 500;
        """, conn)
        conn.close()
        
        top500_tickers = stocks_df['ticker'].tolist()
        print(f"✅ 상위 500개 종목:")
        print(f"   1위: {stocks_df.iloc[0]['name']} ({stocks_df.iloc[0]['sector']})")
        print(f"   2위: {stocks_df.iloc[1]['name']} ({stocks_df.iloc[1]['sector']})")
        print(f"   3위: {stocks_df.iloc[2]['name']} ({stocks_df.iloc[2]['sector']})")
        
        # 2. 장기 데이터 로드 (2023-01-01 ~ 2025-02-11)
        print("\n2. 장기 데이터 로드 (2023-01-01 ~ 2025-02-11)...")
        print("   📊 약 2년 데이터 로딩 중... (1-3분 소요)")
        
        data = loader.load_data(
            universe=top500_tickers,
            start_date="2023-01-01",
            end_date="2025-02-11",
            include_technical=False
        )
        
        print(f"✅ 데이터 로드 완료:")
        print(f"   기간: {data['close'].index[0].date()} ~ {data['close'].index[-1].date()}")
        print(f"   거래일수: {len(data['close'])} 일")
        print(f"   종목수: {len(data['close'].columns)} 개")
        print(f"   총 데이터: {len(data['close']) * len(data['close'].columns):,} 포인트")
        
        # 3. LLM으로 알파 생성 (여러 개)
        print("\n3. LLM으로 다양한 알파 생성...")
        api_key = os.getenv('OPENAI_API_KEY')
        client = openai.OpenAI(api_key=api_key)
        developer = QuantDeveloper(client)
        
        idea = """
        거래량 급증과 주가 모멘텀을 결합한 전략
        
        핵심:
        - 거래량이 평소보다 많이 증가한 종목
        - 단기 상승 추세인 종목
        - 거래량과 주가가 동반 상승
        
        데이터: close, volume
        시간: 5-20일 단기
        """
        
        print(f"   아이디어: 거래량 급증 + 모멘텀 전략")
        
        alphas = developer.generate_alphas(
            refined_idea=idea,
            relevant_fields=['close', 'volume'],
            num_variations=5  # 5개 생성
        )
        
        print(f"✅ {len(alphas)}개 알파 생성 완료")
        
        # 4. 각 알파 백테스트
        print("\n4. 장기 백테스트 실행...")
        print("   (각 알파당 10-30초 소요)")
        
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()
        
        results = []
        
        for i, alpha in enumerate(alphas, 1):
            print(f"\n   [{i}/{len(alphas)}] {alpha.description[:80]}...")
            
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
                    alpha_expr=alpha.expr[:100],
                    quantiles=(0.2, 0.8),  # 상위/하위 20%
                    rebalance_freq='1D'
                )
                
                print(f"        IC: {result.ic:>7.4f} | Sharpe: {result.sharpe_ratio:>6.2f} | 연수익: {result.annual_return:>7.2%}")
                
                results.append({
                    'alpha': alpha,
                    'result': result
                })
                
            except Exception as e:
                print(f"        ⚠️  실패: {str(e)[:60]}")
        
        # 5. 결과 정리 및 순위
        if not results:
            print("\n❌ 성공한 알파가 없습니다.")
            return False
        
        print("\n" + "=" * 60)
        print("📊 장기 백테스트 결과 (2023-2025, 2년)")
        print("=" * 60)
        
        # IC순 정렬
        results.sort(key=lambda x: x['result'].ic, reverse=True)
        
        print("\n🏆 알파 순위 (IC 기준):")
        print("-" * 60)
        for i, item in enumerate(results, 1):
            r = item['result']
            alpha = item['alpha']
            
            print(f"\n{i}위. {alpha.description[:60]}")
            print(f"     IC: {r.ic:.4f} | Sharpe: {r.sharpe_ratio:.2f} | 연수익: {r.annual_return:.2%}")
            print(f"     MDD: {r.max_drawdown:.2%} | 회전율: {r.turnover:.2%} | 승률: {r.win_rate:.2%}")
        
        # 최고 알파 상세
        best = results[0]
        r = best['result']
        alpha = best['alpha']
        
        print("\n" + "=" * 60)
        print("🥇 최고 성과 알파 (상세)")
        print("=" * 60)
        
        print(f"\n알파: {alpha.description}")
        print(f"\n표현식:")
        print(f"  {alpha.expr[:200]}")
        if len(alpha.expr) > 200:
            print(f"  ...")
        
        print(f"\n카테고리: {alpha.category}")
        print(f"복잡도: {alpha.complexity}/10")
        print(f"연산자: {', '.join(alpha.operators_used[:5])}")
        
        print(f"\n📈 성과 지표 (2년):")
        print(f"  IC (Information Coefficient):  {r.ic:>8.4f}")
        print(f"  IC 표준편차:                  {r.ic_std:>8.4f}")
        print(f"  IR (Information Ratio):       {r.ir:>8.2f}")
        print(f"  Sharpe Ratio:                 {r.sharpe_ratio:>8.2f}")
        print(f"  연평균 수익률:                {r.annual_return:>8.2%}")
        print(f"  누적 수익률 (2년):            {r.total_return:>8.2%}")
        print(f"  최대 낙폭 (MDD):              {r.max_drawdown:>8.2%}")
        print(f"  평균 회전율:                  {r.turnover:>8.2%}")
        print(f"  승률:                         {r.win_rate:>8.2%}")
        
        print(f"\n💰 투자 시뮬레이션 (1억원 투자):")
        final_capital = 100_000_000 * (1 + r.total_return)
        profit = final_capital - 100_000_000
        print(f"  초기 자본:  100,000,000원")
        print(f"  최종 자본:  {final_capital:>13,.0f}원")
        print(f"  순이익:     {profit:>13,.0f}원")
        
        print(f"\n📊 평가:")
        
        # IC 평가
        if r.ic > 0.05:
            print(f"  IC {r.ic:.4f}:  🎉 우수 (> 0.05)")
        elif r.ic > 0.03:
            print(f"  IC {r.ic:.4f}:  ✅ 양호 (> 0.03)")
        elif r.ic > 0.01:
            print(f"  IC {r.ic:.4f}:  ⚠️  보통 (> 0.01)")
        else:
            print(f"  IC {r.ic:.4f}:  ❌ 약함")
        
        # Sharpe 평가
        if r.sharpe_ratio > 2.0:
            print(f"  Sharpe {r.sharpe_ratio:.2f}: 🎉 탁월 (> 2.0)")
        elif r.sharpe_ratio > 1.5:
            print(f"  Sharpe {r.sharpe_ratio:.2f}: 🎉 우수 (> 1.5)")
        elif r.sharpe_ratio > 1.0:
            print(f"  Sharpe {r.sharpe_ratio:.2f}: ✅ 양호 (> 1.0)")
        elif r.sharpe_ratio > 0.5:
            print(f"  Sharpe {r.sharpe_ratio:.2f}: ⚠️  보통 (> 0.5)")
        else:
            print(f"  Sharpe {r.sharpe_ratio:.2f}: ❌ 약함")
        
        # MDD 평가
        if abs(r.max_drawdown) < 0.15:
            print(f"  MDD {r.max_drawdown:.2%}:     ✅ 우수 (< 15%)")
        elif abs(r.max_drawdown) < 0.25:
            print(f"  MDD {r.max_drawdown:.2%}:     ⚠️  보통 (< 25%)")
        else:
            print(f"  MDD {r.max_drawdown:.2%}:     ❌ 높음 (> 25%)")
        
        print("\n" + "=" * 60)
        print("✅ 장기 백테스트 완료!")
        print("=" * 60)
        
        print("\n💡 다음 단계:")
        print("  1. 최고 알파를 Genetic Programming으로 더욱 개선")
        print("  2. 다양한 시장 상황에서 교차 검증")
        print("  3. 리스크 관리 규칙 추가 (손절, 포지션 사이징)")
        print("  4. 실거래 시뮬레이션 (모의투자)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
