#!/usr/bin/env python3
"""
장기 백테스트: 간단하고 검증된 알파들
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from alpha_gpt_kr.data.postgres_loader import PostgresDataLoader
from alpha_gpt_kr.backtest.engine import BacktestEngine
from alpha_gpt_kr.mining.operators import AlphaOperators as ops
import pandas as pd

def main():
    print("=" * 60)
    print("장기 백테스트: 시가총액 상위 500개 (2023~2025)")
    print("=" * 60)
    
    try:
        # 1. 시가총액 상위 500개 종목
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
        
        top500 = stocks_df['ticker'].tolist()
        print(f"✅ 종목: {len(top500)}개")
        print(f"   1위: {stocks_df.iloc[0]['name']}")
        print(f"   2위: {stocks_df.iloc[1]['name']}")
        print(f"   3위: {stocks_df.iloc[2]['name']}")
        
        # 2. 2년 데이터 로드
        print("\n2. 2년 데이터 로드 (2023-01-01 ~ 2025-02-11)...")
        print("   로딩 중... (1-2분)")
        
        data = loader.load_data(
            universe=top500,
            start_date="2023-01-01",
            end_date="2025-02-11"
        )
        
        close = data['close']
        volume = data['volume']
        returns = close.pct_change()
        
        print(f"✅ 데이터:")
        print(f"   기간: {close.index[0].date()} ~ {close.index[-1].date()}")
        print(f"   일수: {len(close)} 일")
        print(f"   종목: {len(close.columns)} 개")
        
        # 3. 여러 알파 테스트
        print("\n3. 다양한 알파 전략 백테스트...")
        
        alphas = [
            {
                'name': '거래량 급증 + 모멘텀',
                'desc': '5일 평균 거래량/20일 평균 거래량 × 5일 수익률',
                'expr': lambda: ops.cwise_mul(
                    ops.div(ops.ts_mean(volume, 5), ops.ts_mean(volume, 20)),
                    ops.ts_delta(close, 5) / close.shift(5)
                )
            },
            {
                'name': '단순 모멘텀',
                'desc': '10일 수익률',
                'expr': lambda: ops.ts_delta(close, 10) / close.shift(10)
            },
            {
                'name': '거래량-주가 상관관계',
                'desc': '20일 거래량-주가 상관계수',
                'expr': lambda: ops.ts_corr(volume, close, 20)
            },
            {
                'name': '변동성 조정 모멘텀',
                'desc': '10일 수익률 / 20일 변동성',
                'expr': lambda: ops.div(
                    ops.ts_delta(close, 10) / close.shift(10),
                    ops.ts_std(returns, 20)
                )
            },
            {
                'name': '거래량 가속도',
                'desc': '5일 평균 거래량 변화율',
                'expr': lambda: ops.ts_delta(ops.ts_mean(volume, 5), 5) / ops.ts_mean(volume, 5).shift(5)
            }
        ]
        
        results = []
        
        for i, alpha_def in enumerate(alphas, 1):
            print(f"\n   [{i}/{len(alphas)}] {alpha_def['name']}")
            print(f"        {alpha_def['desc']}")
            
            try:
                # 알파 계산
                alpha_values = alpha_def['expr']()
                
                # 백테스트
                engine = BacktestEngine(
                    universe=top500,
                    price_data=close,
                    return_data=returns
                )
                
                result = engine.backtest(
                    alpha=alpha_values,
                    alpha_expr=alpha_def['name'],
                    quantiles=(0.2, 0.8)
                )
                
                print(f"        IC: {result.ic:>7.4f} | Sharpe: {result.sharpe_ratio:>6.2f} | 연수익: {result.annual_return:>7.2%}")
                
                results.append({
                    'name': alpha_def['name'],
                    'desc': alpha_def['desc'],
                    'result': result
                })
                
            except Exception as e:
                print(f"        ⚠️  실패: {str(e)[:60]}")
        
        if not results:
            print("\n❌ 성공한 알파가 없습니다.")
            return False
        
        # 4. 결과 정리
        print("\n" + "=" * 60)
        print("📊 2년 장기 백테스트 결과")
        print("=" * 60)
        
        results.sort(key=lambda x: x['result'].ic, reverse=True)
        
        print("\n🏆 알파 순위 (IC 기준):\n")
        for i, item in enumerate(results, 1):
            r = item['result']
            print(f"{i}위. {item['name']}")
            print(f"     {item['desc']}")
            print(f"     IC: {r.ic:.4f} | Sharpe: {r.sharpe_ratio:.2f} | 연수익: {r.annual_return:.2%}")
            print(f"     MDD: {r.max_drawdown:.2%} | 회전율: {r.turnover:.2%} | 승률: {r.win_rate:.2%}\n")
        
        # 5. 최고 알파 상세
        best = results[0]
        r = best['result']
        
        print("=" * 60)
        print("🥇 최고 성과 알파")
        print("=" * 60)
        
        print(f"\n전략: {best['name']}")
        print(f"설명: {best['desc']}")
        
        print(f"\n📈 2년 성과:")
        print(f"  IC (정보계수):        {r.ic:>8.4f}")
        print(f"  IC 표준편차:          {r.ic_std:>8.4f}")
        print(f"  IR (정보비율):        {r.ir:>8.2f}")
        print(f"  Sharpe Ratio:         {r.sharpe_ratio:>8.2f}")
        print(f"  연평균 수익률:        {r.annual_return:>8.2%}")
        print(f"  누적 수익률 (2년):    {r.total_return:>8.2%}")
        print(f"  최대 낙폭 (MDD):      {r.max_drawdown:>8.2%}")
        print(f"  평균 회전율:          {r.turnover:>8.2%}")
        print(f"  승률:                 {r.win_rate:>8.2%}")
        
        print(f"\n💰 1억원 투자 시뮬레이션:")
        final = 100_000_000 * (1 + r.total_return)
        profit = final - 100_000_000
        print(f"  초기:  100,000,000원")
        print(f"  최종:  {final:>13,.0f}원")
        print(f"  수익:  {profit:>13,.0f}원 ({r.total_return:>6.2%})")
        
        print(f"\n📊 평가:")
        
        if r.ic > 0.05:
            print(f"  ✅ IC {r.ic:.4f}: 우수 (> 0.05)")
        elif r.ic > 0.03:
            print(f"  ✅ IC {r.ic:.4f}: 양호 (> 0.03)")
        elif r.ic > 0.01:
            print(f"  ⚠️  IC {r.ic:.4f}: 보통 (> 0.01)")
        else:
            print(f"  ❌ IC {r.ic:.4f}: 약함")
        
        if r.sharpe_ratio > 2.0:
            print(f"  🎉 Sharpe {r.sharpe_ratio:.2f}: 탁월 (> 2.0)")
        elif r.sharpe_ratio > 1.5:
            print(f"  ✅ Sharpe {r.sharpe_ratio:.2f}: 우수 (> 1.5)")
        elif r.sharpe_ratio > 1.0:
            print(f"  ✅ Sharpe {r.sharpe_ratio:.2f}: 양호 (> 1.0)")
        else:
            print(f"  ⚠️  Sharpe {r.sharpe_ratio:.2f}: 보통")
        
        if abs(r.max_drawdown) < 0.15:
            print(f"  ✅ MDD {r.max_drawdown:.2%}: 우수 (< 15%)")
        elif abs(r.max_drawdown) < 0.25:
            print(f"  ⚠️  MDD {r.max_drawdown:.2%}: 보통 (< 25%)")
        else:
            print(f"  ❌ MDD {r.max_drawdown:.2%}: 높음 (> 25%)")
        
        print("\n" + "=" * 60)
        print("✅ 장기 백테스트 완료!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
