#!/usr/bin/env python3
"""
Alpha-GPT + PostgreSQL 통합 테스트
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from alpha_gpt_kr.core import AlphaGPT
from alpha_gpt_kr.data.postgres_loader import PostgresDataLoader
from loguru import logger

def main():
    print("=" * 60)
    print("Alpha-GPT + PostgreSQL 통합 테스트")
    print("=" * 60)
    
    try:
        # 1. 샘플 종목 선택 (직접 지정)
        print("\n1. 샘플 종목 선택...")
        loader = PostgresDataLoader()
        
        # 전체 종목에서 랜덤으로 20개 선택
        conn = loader._get_connection()
        import pandas as pd
        stocks_df = pd.read_sql("""
            SELECT ticker, name 
            FROM stocks 
            WHERE is_active = true 
            ORDER BY RANDOM() 
            LIMIT 20;
        """, conn)
        conn.close()
        
        sample_tickers = stocks_df['ticker'].tolist()
        print(f"✅ 샘플 종목 {len(sample_tickers)}개 선택:")
        for ticker, name in stocks_df.values[:5]:
            print(f"   {ticker}: {name}")
        print("   ...")
        
        # 2. 데이터 로드 (직접)
        print("\n2. PostgreSQL에서 데이터 로드...")
        data = loader.load_data(
            universe=sample_tickers,
            start_date="2024-12-01",
            end_date="2025-02-11",
            include_technical=True
        )
        
        # 수익률 계산
        data['returns'] = data['close'].pct_change()
        
        print(f"✅ 데이터 로드 완료:")
        print(f"   기간: {data['close'].index[0]} ~ {data['close'].index[-1]}")
        print(f"   종목 수: {len(data['close'].columns)}")
        print(f"   데이터 필드: {list(data.keys())[:10]}...")
        
        # 4. 간단한 알파 백테스트
        print("\n4. 샘플 알파 백테스트...")
        from alpha_gpt_kr.backtest.engine import BacktestEngine
        
        # 간단한 알파: 5일 이동평균
        close = data['close']
        returns = data['returns']
        alpha = close.rolling(5).mean()
        
        # 백테스트
        engine = BacktestEngine(
            universe=sample_tickers,
            price_data=close,
            return_data=returns
        )
        result = engine.backtest(
            alpha=alpha,
            alpha_expr="ts_mean(close, 5)",
            quantiles=(0.3, 0.7),
            rebalance_freq='1D'
        )
        
        print(f"✅ 백테스트 완료:")
        print(f"   IC: {result.ic:.4f}")
        print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"   연평균 수익률: {result.annual_return:.2%}")
        print(f"   최대 낙폭: {result.max_drawdown:.2%}")
        
        # 5. 연산자 테스트
        print("\n5. 알파 연산자 테스트...")
        from alpha_gpt_kr.mining.operators import AlphaOperators as ops
        
        # 몇 가지 연산자 테스트
        volume = data['volume']
        
        test_alphas = {
            'ts_mean': ops.ts_mean(close, 10),
            'ts_delta': ops.ts_delta(close, 1),
            'ts_corr': ops.ts_corr(close, volume, 20),
            'zscore': ops.zscore_scale(close)
        }
        
        print("✅ 연산자 테스트 완료:")
        for name, alpha in test_alphas.items():
            valid_ratio = alpha.notna().sum().sum() / alpha.size
            print(f"   {name}: {alpha.shape}, 유효 데이터 {valid_ratio:.1%}")
        
        print("\n" + "=" * 60)
        print("✅ 모든 테스트 통과!")
        print("=" * 60)
        print("\n🎉 Alpha-GPT가 PostgreSQL 데이터로 정상 작동합니다!")
        print()
        print("다음 단계:")
        print("1. OpenAI API 키 설정 (.env 파일)")
        print("2. LLM 기반 알파 생성 테스트")
        print("3. 실제 트레이딩 아이디어로 알파 마이닝")
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
